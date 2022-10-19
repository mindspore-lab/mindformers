# Copyright 2020 Huawei Technologies Co., Ltd
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
"""
GPT-2 finetune and evaluation script for Language Modeling task.
"""
import argparse
import math
import os
import time

import mindspore.communication.management as D
from mindspore import context
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from transformer.models.gpt.gpt_lm import GPT2FinetuneCell, GPT2LM
from transformer.build_parallel_config import build_parallel_config
from transformer.data import build_downstream_dataset
from transformer.learning_rate import build_lr
from transformer.logger import get_logger
from transformer.models.bert.utils import LossCallBack
from transformer.models.build_model import get_downstream_config
from transformer.modules import override_attention
from transformer.optim.optimizer import build_optimizer
from transformer.utils import parse_with_config, _convert_dtype_class, get_newest_ckpt, make_directory

_cur_dir = os.getcwd()


def set_context_env(config):
    """Set the context env"""
    context_args = config.context
    if context_args['device_target'] != "GPU":
        context_args['enable_graph_kernel'] = False
        config.logger.info("Disable graph kernel.")
    context.set_context(**context_args)


def check_args(opt, device_num):
    """Validate the dp and mp"""
    dp = opt.parallel_config['data_parallel']
    mp = opt.parallel_config['model_parallel']
    if mp < 1:
        raise ValueError("The model parallel must be equal or larger than 1. "
                         f"You can fix this by setting --model_parallel=1, for example.")
    if mp > device_num:
        raise ValueError(f"The model parallel must be less or equal to the device_num {device_num}. "
                         f"You can fix this by setting --model_parallel=1, for example")
    if opt.parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL) and dp * mp != device_num:
        opt.logger.warn(f"The data_parallel * model_parallel must be equal to the {device_num}. "
                        f"You can remove this warning by setting --data_parallel={device_num // mp}. "
                        f"Now the full_batch will be set False.")
        opt.full_batch = False

    # If the user runs the data_parallel and set full_batch to be true
    if opt.parallel_mode in (ParallelMode.DATA_PARALLEL,) and opt.full_batch:
        raise ValueError("full_batch doesn't support DATA_PARALLEL mode, you can fix it by setting --full_batch=False")


def set_auto_parallel_context_env(config):
    """Set the auto parallel env"""
    if config.parallel_mode != context.ParallelMode.STAND_ALONE:
        config.logger.info(f"Enabling the parallel mode: {config.parallel_mode} for multi card training.")
        D.init()
        device_num = D.get_group_size()
        rank_id = D.get_rank()
        context.reset_auto_parallel_context()
        check_args(config, device_num)
        context.set_auto_parallel_context(parallel_mode=config.parallel_mode, gradients_mean=True,
                                          full_batch=config.full_batch,
                                          device_num=device_num, grad_accumulation_step=config.acc_step)

    else:
        config.logger.info(f"Enabling the parallel mode: {config.parallel_mode} for stand alone training.")
        rank_id = 0
        device_num = 1
    if config.full_batch:
        config.logger.info("Enabling the full batch import.")
    return rank_id, device_num


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


def do_train(opt, rank_id, dataset=None, network=None, load_checkpoint_path="", save_checkpoint_path="", epoch_num=1):
    """
    Do train
    Args:
        dataset: the train dataset.
        network:  the network with loss
        load_checkpoint_path: the file path which saved pretrained model checkpoint.
        save_checkpoint_path:  the file path which will save finetuned model checkpoint.
        epoch_num: the number of epoch.
    """
    if load_checkpoint_path == "":
        raise ValueError("Pretrain model missed, finetune task must load pretrain model!")

    steps_per_epoch = dataset.get_dataset_size()
    print("steps_per_epoch is", steps_per_epoch)
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
    if rank_id == 0:
        ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=1)
        ckpoint_cb = ModelCheckpoint(prefix="gpt2_language_model",
                                     directory=None if save_checkpoint_path == "" else save_checkpoint_path,
                                     config=ckpt_config)

    param_dict = load_checkpoint(load_checkpoint_path)
    for param in param_dict.keys():
        print("param_dict:", param, param_dict[param])

    # print("param_dict", param_dict.keys())
    final_param_dict = {}
    for name, _ in param_dict.items():
        final_param_dict['gpt2.' + name] = param_dict[name]

    final_param_dict['gpt2.dense1.weight'] = param_dict['backbone.word_embedding.embedding_table']

    load_param_into_net(network, final_param_dict)
    print("Load pretrained parameter successfully!\n")

    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2 ** 32, scale_factor=2, scale_window=1000)
    netwithgrads = GPT2FinetuneCell(network, optimizer=optimizer, scale_update_cell=update_cell)
    netwithgrads.set_train(True)

    model = Model(netwithgrads)
    if rank_id == 0:
        callbacks = [TimeMonitor(dataset.get_dataset_size()), LossCallBack(dataset.get_dataset_size()), ckpoint_cb]
    else:
        callbacks = [TimeMonitor(dataset.get_dataset_size()), LossCallBack(dataset.get_dataset_size())]
    print("==================== Starting Finetuning ====================")
    callback_size = opt.sink_size
    model.train(actual_epoch_num, dataset, callbacks=callbacks, sink_size=callback_size)
    print("==================== Finetuning Success  ====================")


def do_eval(model_config, dataset=None, network=None, metric=None, load_checkpoint_path="", eval_type=None):
    """
    Do eval
    Args:
        dataset: the eval dataset.
        network:  the network with loss.
        metric: the evaluation method.
        load_checkpoint_path: the file path which saved finetuned model checkpoint.
        eval_type: option for "zero-shot" or "finetuned"
    """
    if load_checkpoint_path == "":
        raise ValueError("Finetune model missed, evaluation task must load finetune model!")

    if metric.lower() == "ppl":
        print("Prepare to calculate the ppl score ...")
        model_config.is_training = False
        gpt2_loss = network(config=model_config)

        gpt2_loss.set_train(False)
        param_dict = load_checkpoint(load_checkpoint_path)

        if eval_type == "zero-shot":
            final_param_dict = {}
            for name, _ in param_dict.items():
                final_param_dict['gpt2.' + name] = param_dict[name]
            # final_param_dict['gpt2.dense1.weight'] = param_dict['gpt2_embedding_lookup.embedding_table']
            final_param_dict['gpt2.dense1.weight'] = param_dict['backbone.word_embedding.embedding_table']
            load_param_into_net(gpt2_loss, final_param_dict)
            print("load pretrained parameter successfully!\n")
        elif eval_type == "finetuned":
            load_param_into_net(gpt2_loss, param_dict)
            print("load finetuned parameter successfully!\n")
        else:
            raise ValueError("Evaluation type missed, eval_type should be [zero-shot, finetuned]")

        model = Model(gpt2_loss)
        columns_list = ["input_ids", "input_mask", "label_ids"]
        print("==================== [PPL] Testing ====================")
        num_data = 1
        total_loss = 0.0
        avg_loss = 0.0
        for data in dataset.create_dict_iterator():
            input_data = []
            for i in columns_list:
                input_data.append(data[i])
            input_ids, input_mask, label_ids = input_data
            loss = model.predict(input_ids, input_mask, label_ids)
            print("loss is", loss)
            loss = float(loss.asnumpy())
            total_loss += loss
            avg_loss = float(total_loss / num_data)
            print(" | Current Loss: {:.6f}".format(avg_loss))
            print(" | Current PPL: {}\n\n".format(math.exp(avg_loss)))
            num_data += 1

        print("\n\n")
        print("**************************************************************")
        print("Average Loss: {:.6f}".format(avg_loss))
        print("Average PPL: {:.6f}".format(math.exp(avg_loss)))
        print("********************** Testing Finished **********************")
    else:
        raise ValueError("metric method not supported, support: [ppl]")


def run_languagemodel(args_opt):
    """
    run Language Modeling task
    """
    set_context_env(args_opt)
    rank_id, device_num = set_auto_parallel_context_env(args_opt)
    parallel_config = build_parallel_config(args_opt)
    model_config = get_downstream_config(args_opt)
    model_config.parallel_config = parallel_config
    set_fused_kernel(args_opt)
    epoch_num = args_opt.epoch_num
    metric = args_opt.metric_method
    save_finetune_ckpt_path = args_opt.save_finetune_ckpt_path
    load_finetune_ckpt_path = args_opt.load_pretrain_ckpt_path
    load_pretrain_ckpt_path = args_opt.load_pretrain_ckpt_path

    if args_opt.do_train.lower() == "false" and args_opt.do_eval.lower() == "false":
        raise ValueError("At least one of 'do_train' or 'do_eval' must be true")
    if args_opt.do_train.lower() == "true" and args_opt.train_data_path == "":
        raise ValueError("'train_data_path' must be set when do finetune task")
    if args_opt.do_eval.lower() == "true" and args_opt.eval_data_path == "":
        raise ValueError("'eval_data_path' must be set when do evaluation task")

    target = args_opt.context["device_target"]
    if target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args_opt.device_id)
    elif target == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
        context.set_context(enable_graph_kernel=True)
    else:
        raise Exception("Target error, GPU or Ascend is supported.")
    model_config.is_training = True
    gpt2_loss = GPT2LM(model_config)

    if args_opt.do_train.lower() == "true":
        print("====================    Start Loading Train Dataset   ==================")
        print(" | Train Dataset: {}".format(args_opt.train_data_path))
        print(" | Checkpoint: {}".format(args_opt.load_pretrain_ckpt_path))
        train_dataset = build_downstream_dataset(args_opt, rank_id, device_num,
                                                 batch_size=args_opt.model['train_batch_size'],
                                                 do_shuffle=(args_opt.train_data_shuffle.lower() == "true"),
                                                 dataset_format="mindrecord", task_name="language_model",
                                                 data_file_path=args_opt.train_data_path)
        do_train(args_opt, rank_id, train_dataset, gpt2_loss, load_pretrain_ckpt_path, save_finetune_ckpt_path,
                 epoch_num)

        if args_opt.do_eval.lower() == "true" and rank_id == 0:
            if save_finetune_ckpt_path == "":
                load_finetune_checkpoint_dir = _cur_dir
            else:
                load_finetune_checkpoint_dir = make_directory(save_finetune_ckpt_path)
            load_finetune_ckpt_path = get_newest_ckpt(load_finetune_checkpoint_dir, "gpt")

    if args_opt.do_eval.lower() == "true" and rank_id == 0:
        # get_model_setting(cfg, gpt2_net_cfg)
        print("==================== Start Loading Evaluation Dataset ==================")
        print(" | Eval Dataset: {}".format(args_opt.eval_data_path))
        print(" | Checkpoint: {}".format(load_finetune_ckpt_path))
        eval_dataset = build_downstream_dataset(args_opt, 0, 1, batch_size=args_opt.model['eval_batch_size'],
                                                do_shuffle=(args_opt.train_data_shuffle.lower() == "true"),
                                                is_training=False,
                                                dataset_format="mindrecord", task_name="language_model",
                                                data_file_path=args_opt.eval_data_path)

        eval_data_size = eval_dataset.get_dataset_size()
        print(" | Eval Data Size: {}".format(eval_data_size))
        model_config.batch_size = args_opt.model['eval_batch_size']
        do_eval(model_config, eval_dataset, GPT2LM, metric, load_finetune_ckpt_path, args_opt.eval_type)


if __name__ == "__main__":
    print("Start Time: \n", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="transformer/configs/gpt/language_model.yaml", help='YAML config files')
    args = parse_with_config(parser)
    args.logger = get_logger()
    modify_args(args)
    set_seed(args.seed)
    run_languagemodel(args)
    print("End Time: \n", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
