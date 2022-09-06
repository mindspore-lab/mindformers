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
import os

from mindspore import context
from mindspore.common import set_seed
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from tasks.nlp.bert_downstream.src.assessment_method import Accuracy, F1, MCC, SpearmanCorrelation
from tasks.nlp.bert_downstream.src.bert_for_finetune import BertFinetuneCell, BertCLS
from tasks.nlp.bert_downstream.src.dataset import create_classification_dataset
from tasks.nlp.bert_downstream.src.utils import make_directory, LossCallBack, load_newest_ckpt
from transformer.build_parallel_config import build_parallel_config
from transformer.learning_rate import build_lr
from transformer.logger import get_logger
from transformer.models.build_model import get_downstream_config
from transformer.modules import override_attention
from transformer.optim.optimizer import build_optimizer
from transformer.utils import parse_with_config, _convert_dtype_class

_cur_dir = os.getcwd()


def set_context_env(config):
    """Set the context env"""
    context_args = config.context
    if context_args['device_target'] != "GPU":
        context_args['enable_graph_kernel'] = False
        config.logger.info("Disable graph kernel.")
    context.set_context(**context_args)


def set_fused_kernel(config):
    if config.speed_up.get('fused_kernel', False):
        pwd = os.path.dirname(os.path.abspath(__file__))
        softmax_kernel_path = os.path.join(pwd, 'modules/fused_kernel/aot_scale_masked_softmax.cu')
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
    print("steps_per_epoch is", steps_per_epoch)
    # lr
    lr = build_lr(opt, epoch_num, steps_per_epoch)
    actual_epoch_num = int(epoch_num * steps_per_epoch / opt.sink_size)
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
    ckpoint_cb = ModelCheckpoint(prefix="classifier",
                                 directory=None if save_checkpoint_path == "" else save_checkpoint_path,
                                 config=ckpt_config)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(network, param_dict)

    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2 ** 32, scale_factor=2, scale_window=1000)
    netwithgrads = BertFinetuneCell(network, optimizer=optimizer, scale_update_cell=update_cell)
    model = Model(netwithgrads)
    callbacks = [TimeMonitor(dataset.get_dataset_size()), LossCallBack(dataset.get_dataset_size()), ckpoint_cb]
    callback_size = opt.sink_size
    model.train(actual_epoch_num, dataset, callbacks=callbacks, sink_size=callback_size)


def eval_result_print(assessment_method="accuracy", callback=None):
    """ print eval result """
    if assessment_method == "accuracy":
        print("acc_num {} , total_num {}, accuracy {:.6f}".format(callback.acc_num, callback.total_num,
                                                                  callback.acc_num / callback.total_num))
    elif assessment_method == "f1":
        print("Precision {:.6f} ".format(callback.TP / (callback.TP + callback.FP)))
        print("Recall {:.6f} ".format(callback.TP / (callback.TP + callback.FN)))
        print("F1 {:.6f} ".format(2 * callback.TP / (2 * callback.TP + callback.FP + callback.FN)))
    elif assessment_method == "mcc":
        print("MCC {:.6f} ".format(callback.cal()))
    elif assessment_method == "spearman_correlation":
        print("Spearman Correlation is {:.6f} ".format(callback.cal()[0]))
    else:
        raise ValueError("Assessment method not supported, support: [accuracy, f1, mcc, spearman_correlation]")


def do_eval(dataset=None, network=None, num_class=2, assessment_method="accuracy", load_checkpoint_path="",
            model_config=None):
    """ do eval """
    if load_checkpoint_path == "":
        raise ValueError("Finetune model missed, evaluation task must load finetune model!")
    net_for_pretraining = network(model_config, False, num_class)
    net_for_pretraining.set_train(False)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(net_for_pretraining, param_dict)
    model = Model(net_for_pretraining)

    if assessment_method == "accuracy":
        callback = Accuracy()
    elif assessment_method == "f1":
        callback = F1(False, num_class)
    elif assessment_method == "mcc":
        callback = MCC()
    elif assessment_method == "spearman_correlation":
        callback = SpearmanCorrelation()
    else:
        raise ValueError("Assessment method not supported, support: [accuracy, f1, mcc, spearman_correlation]")

    columns_list = ["input_ids", "input_mask", "segment_ids", "label_ids"]
    for data in dataset.create_dict_iterator(num_epochs=1):
        input_data = []
        for i in columns_list:
            input_data.append(data[i])
        input_ids, input_mask, token_type_id, label_ids = input_data
        logits = model.predict(input_ids, input_mask, token_type_id, label_ids)
        callback.update(logits, label_ids)
    print("==============================================================")
    eval_result_print(assessment_method, callback)
    print("==============================================================")


def run_classifier(args_opt):
    """run classifier task"""
    if args_opt.do_train.lower() == "false" and args_opt.do_eval.lower() == "false":
        raise ValueError("At least one of 'do_train' or 'do_eval' must be true")
    if args_opt.do_train.lower() == "true" and args_opt.train_data_file_path == "":
        raise ValueError("'train_data_file_path' must be set when do finetune task")
    if args_opt.do_eval.lower() == "true" and args_opt.eval_data_file_path == "":
        raise ValueError("'eval_data_file_path' must be set when do evaluation task")
    set_context_env(args_opt)
    parallel_config = build_parallel_config(args_opt)
    model_config = get_downstream_config(args_opt)
    model_config.parallel_config = parallel_config
    set_fused_kernel(args_opt)
    epoch_num = args_opt.epoch_num
    assessment_method = args_opt.assessment_method.lower()
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

    netwithloss = BertCLS(model_config, True, num_labels=args_opt.num_class, dropout_prob=0.1,
                          assessment_method=assessment_method)

    if args_opt.do_train.lower() == "true":
        ds = create_classification_dataset(batch_size=args_opt.model['train_batch_size'],
                                           assessment_method=assessment_method,
                                           data_file_path=args_opt.train_data_file_path,
                                           dataset_format="tfrecord",
                                           schema_file_path=args_opt.schema_file_path,
                                           do_shuffle=(args_opt.train_data_shuffle.lower() == "true"))
        do_train(args_opt, ds, netwithloss, load_pretrain_checkpoint_path, save_finetune_checkpoint_path, epoch_num)

        if args_opt.do_eval.lower() == "true":
            if save_finetune_checkpoint_path == "":
                load_finetune_checkpoint_dir = _cur_dir
            else:
                load_finetune_checkpoint_dir = make_directory(save_finetune_checkpoint_path)
            load_finetune_checkpoint_path = load_newest_ckpt(load_finetune_checkpoint_dir, "classifier")

    if args_opt.do_eval.lower() == "true":
        print("do evaluation")
        ds = create_classification_dataset(batch_size=args_opt.model['eval_batch_size'],
                                           assessment_method=assessment_method,
                                           data_file_path=args_opt.eval_data_file_path,
                                           dataset_format="tfrecord",
                                           schema_file_path=args_opt.schema_file_path,
                                           do_shuffle=(args_opt.eval_data_shuffle.lower() == "true"))
        do_eval(ds, BertCLS, args_opt.num_class, assessment_method, load_finetune_checkpoint_path, model_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="transformer/configs/bert/task_classifier_config.yaml",
                        help='YAML config files')
    args = parse_with_config(parser)
    args.logger = get_logger()
    modify_args(args)
    set_seed(args.seed)
    run_classifier(args)
