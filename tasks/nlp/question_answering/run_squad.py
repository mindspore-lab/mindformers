# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

'''
Bert finetune and evaluation script.
'''
import argparse
import collections
import os

import mindspore.common.dtype as mstype
from mindspore import context
from mindspore.common import set_seed
from mindspore.common.tensor import Tensor
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from tasks.nlp.question_answering.src.bert_for_finetune import BertSquadCell, BertSquad
from tasks.nlp.data.dataset import create_squad_dataset
from tasks.nlp.utils import make_directory, LossCallBack
from transformer.build_parallel_config import build_parallel_config
from transformer.learning_rate import build_lr
from transformer.logger import get_logger
from transformer.models.build_model import get_downstream_config
from transformer.modules import override_attention
from transformer.optim.optimizer import build_optimizer
from transformer.utils import parse_with_config, _convert_dtype_class, get_newest_ckpt

_cur_dir = os.getcwd()


def set_context_env(config):
    """Set the context env"""
    context_args = config.context
    context_args['enable_graph_kernel'] = False
    if context_args['device_target'] != "GPU":
        context_args['enable_graph_kernel'] = False
        config.logger.info("Disable graph kernel.")
    context.set_context(**context_args)


def set_fused_kernel(config):
    if config.speed_up.get('fused_kernel', False):
        pwd = os.path.dirname(os.path.abspath(__file__))
        softmax_kernel_path = os.path.join(pwd, 'modules/fused_kernel/aot_scale_masked_softmax.cu')
        config.logger.info(f"Detect the fused_kernel True, "
                           f"start to compile the cuda code. Cuda code path {softmax_kernel_path}. "
                           f"The attention in the mindspore will be replaced with softmax fused attention.")
        override_attention(softmax_kernel_path)


def modify_args(opt):
    # maps fp16 to mstype.float16 and fp32 to mstype.float32
    for k, v in opt.__dict__.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                v[sub_k] = _convert_dtype_class(sub_v)
        else:
            opt.__dict__[k] = _convert_dtype_class(v)


def do_train(opt, dataset=None, network=None, load_checkpoint_path="", save_checkpoint_path="", epoch_num=1):
    """ do train """
    if load_checkpoint_path == "":
        raise ValueError("Pretrain model missed, finetune task must load pretrain model!")
    steps_per_epoch = dataset.get_dataset_size()
    # optimizer
    steps_per_epoch = dataset.get_dataset_size()
    actual_epoch_num = int(epoch_num * steps_per_epoch / opt.sink_size)
    callback_size = opt.sink_size
    # lr
    lr = build_lr(opt, epoch_num, steps_per_epoch)
    # optimizer
    optimizer = build_optimizer(net=network,
                                lr=lr,
                                optimizer_name=opt.optimizer,
                                args=None,
                                stage_num=1,
                                fused=True,
                                opt_offload=opt.opt_offload)

    # load checkpoint into network
    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=1)
    ckpoint_cb = ModelCheckpoint(prefix="squad",
                                 directory=None if save_checkpoint_path == "" else save_checkpoint_path,
                                 config=ckpt_config)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(network, param_dict)

    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2 ** 32, scale_factor=2, scale_window=1000)
    netwithgrads = BertSquadCell(network, optimizer=optimizer, scale_update_cell=update_cell)
    model = Model(netwithgrads)
    callbacks = [TimeMonitor(dataset.get_dataset_size()), LossCallBack(dataset.get_dataset_size()), ckpoint_cb]
    model.train(actual_epoch_num, dataset, callbacks=callbacks, sink_size=callback_size)


def do_eval(dataset=None, load_checkpoint_path="", eval_batch_size=1, model_config=None):
    """ do eval """
    if load_checkpoint_path == "":
        raise ValueError("Finetune model missed, evaluation task must load finetune model!")
    model_config.is_training = False
    net = BertSquad(model_config)
    net.set_train(False)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(net, param_dict)
    model = Model(net)
    output = []
    RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])
    columns_list = ["input_ids", "input_mask", "segment_ids", "unique_ids"]
    for data in dataset.create_dict_iterator(num_epochs=1):
        input_data = []
        for i in columns_list:
            input_data.append(data[i])
        input_ids, input_mask, segment_ids, unique_ids = input_data
        start_positions = Tensor([1], mstype.float32)
        end_positions = Tensor([1], mstype.float32)
        is_impossible = Tensor([1], mstype.float32)
        logits = model.predict(input_ids, input_mask, segment_ids, start_positions,
                               end_positions, unique_ids, is_impossible)
        ids = logits[0].asnumpy()
        start = logits[1].asnumpy()
        end = logits[2].asnumpy()

        for i in range(eval_batch_size):
            unique_id = int(ids[i])
            start_logits = [float(x) for x in start[i].flat]
            end_logits = [float(x) for x in end[i].flat]
            output.append(RawResult(
                unique_id=unique_id,
                start_logits=start_logits,
                end_logits=end_logits))
    return output


def run_squad(args_opt):
    """run squad task"""
    if args_opt.do_train.lower() == "false" and args_opt.do_eval.lower() == "false":
        raise ValueError("At least one of 'do_train' or 'do_eval' must be true")
    if args_opt.do_train.lower() == "true" and args_opt.train_data_file_path == "":
        raise ValueError("'train_data_file_path' must be set when do finetune task")
    if args_opt.do_eval.lower() == "true":
        if args_opt.vocab_file_path == "":
            raise ValueError("'vocab_file_path' must be set when do evaluation task")
        if args_opt.eval_json_path == "":
            raise ValueError("'tokenization_file_path' must be set when do evaluation task")
    set_context_env(args_opt)
    parallel_config = build_parallel_config(args_opt)
    model_config = get_downstream_config(args_opt)
    model_config.parallel_config = parallel_config
    epoch_num = args_opt.epoch_num
    load_pretrain_checkpoint_path = args_opt.load_pretrain_checkpoint_path
    save_finetune_checkpoint_path = args_opt.save_finetune_checkpoint_path
    load_finetune_checkpoint_path = args_opt.load_finetune_checkpoint_path
    target = args_opt.context["device_target"]
    if target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args_opt.device_id)
    elif target == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
        context.set_context(enable_graph_kernel=True)

    else:
        raise Exception("Target error, GPU or Ascend is supported.")

    model_config.is_training = True
    netwithloss = BertSquad(model_config)

    if args_opt.do_train.lower() == "true":
        ds = create_squad_dataset(batch_size=args_opt.model['train_batch_size'],
                                  data_file_path=args_opt.train_data_file_path,
                                  do_shuffle=(args_opt.train_data_shuffle.lower() == "true"))
        do_train(args_opt, ds, netwithloss, load_pretrain_checkpoint_path, save_finetune_checkpoint_path, epoch_num)
        if args_opt.do_eval.lower() == "true":
            if save_finetune_checkpoint_path == "":
                load_finetune_checkpoint_dir = _cur_dir
            else:
                load_finetune_checkpoint_dir = make_directory(save_finetune_checkpoint_path)
            load_finetune_checkpoint_path = get_newest_ckpt(load_finetune_checkpoint_dir, "squad")

    if args_opt.do_eval.lower() == "true":
        from tasks.nlp import tokenization
        from tasks.nlp.question_answering.src.create_squad_data import read_squad_examples, convert_examples_to_features
        from tasks.nlp.question_answering.src.squad_get_predictions import write_predictions
        from tasks.nlp.question_answering.src.squad_postprocess import squad_postprocess
        tokenizer = tokenization.FullTokenizer(vocab_file=args_opt.vocab_file_path, do_lower_case=True)
        eval_examples = read_squad_examples(args_opt.eval_json_path, False)
        eval_features = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=args_opt.model['seq_length'],
            doc_stride=128,
            max_query_length=64,
            is_training=False,
            output_fn=None,
            vocab_file=args_opt.vocab_file_path)
        ds = create_squad_dataset(batch_size=args_opt.model['eval_batch_size'],
                                  data_file_path=eval_features,
                                  schema_file_path=args_opt.schema_file_path, is_training=False,
                                  do_shuffle=(args_opt.eval_data_shuffle.lower() == "true"))
        model_config.batch_size = args_opt.model['eval_batch_size']
        outputs = do_eval(ds, load_finetune_checkpoint_path, args_opt.eval_batch_size, model_config)
        all_predictions = write_predictions(eval_examples, eval_features, outputs, 20, 30, True)
        squad_postprocess(args_opt.eval_json_path, all_predictions, output_metrics="output.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="transformer/configs/bert/task_squad_config.yaml", help='YAML config files')
    args = parse_with_config(parser)
    args.logger = get_logger()
    modify_args(args)
    set_seed(args.seed)
    run_squad(args)
