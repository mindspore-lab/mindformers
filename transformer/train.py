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

"""
Basic model train script
"""
import argparse
import os.path

import mindspore
from mindspore import context, DynamicLossScaleManager, FixedLossScaleManager
from mindspore.train.model import Model
import mindspore.communication.management as D
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.common import set_seed
from mindspore.nn.wrap.cell_wrapper import MicroBatchInterleaved
from transformer.data import build_dataset
from transformer.optim.optimizer import build_optimizer
from transformer.models import build_model
from transformer.build_parallel_config import build_parallel_config
from transformer.utils import parse_with_config, print_model_size
from transformer.trainer.grad_accu_model import AccModel
from transformer.trainer import build_trainer
from transformer.learning_rate import build_lr
from transformer.modules import override_attention
from transformer.callback import LossCallBack
from transformer.logger import get_logger


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
        softmax_kernel_path = os.path.join(pwd, 'models/fused_kernel/aot_scale_masked_softmax.cu')
        config.logger.info(f"Detect the fused_kernel True, "
                           f"start to compile the cuda code. Cuda code path {softmax_kernel_path}. "
                           f"The attention in the mindspore will be replaced with softmax fused attention.")

        override_attention(softmax_kernel_path)


def set_auto_parallel_context_env(config):
    """Set the auto parallel env"""
    if config.parallel_mode != context.ParallelMode.STAND_ALONE:
        config.logger.info(f"Enabling the parallel mode: {config.parallel_mode} for multi card training.")
        D.init()
        device_num = D.get_group_size()
        rank_id = D.get_rank()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=config.parallel_mode, gradients_mean=True,
                                          enable_parallel_optimizer=config.enable_parallel_optimizer,
                                          device_num=device_num, grad_accumulation_step=config.acc_step)
    else:
        config.logger.info(f"Enabling the parallel mode: {config.parallel_mode} for stand alone training.")
        rank_id = 0
        device_num = 1
    if config.parallel_mode in (context.ParallelMode.SEMI_AUTO_PARALLEL, context.ParallelMode.AUTO_PARALLEL):
        context.set_auto_parallel_context(full_batch=True)
        config.logger.info("Enable the full batch import.")
    return rank_id, device_num


def run_train(opt):
    """Main training process"""
    set_context_env(opt)
    rank_id, device_num = set_auto_parallel_context_env(opt)
    parallel_config = build_parallel_config(opt)
    # This should be called before any cell construction
    set_fused_kernel(opt)
    # Build the model with loss
    net_with_loss = build_model(opt, parallel_config)
    micro_batch_num = opt.speed_up['micro_batch_num']
    flatten_weights = opt.speed_up['flatten_weights']
    if micro_batch_num > 1:
        net_with_loss = MicroBatchInterleaved(net_with_loss, micro_batch_num)
        opt.logger.info(f"Enabling the micro batch interleaved, the batch num is : {micro_batch_num}.")
    if flatten_weights:
        net_with_loss.flatten_weights()
        opt.logger.info("Enabling the flatten_weights.")
    ds = build_dataset(opt, rank_id, device_num)

    epoch_num = opt.epoch_size
    step_per_epoch = ds.get_dataset_size()

    lr = build_lr(opt, epoch_num, step_per_epoch)

    optimizer = build_optimizer(net=net_with_loss,
                                lr=lr,
                                optimizer_name=opt.optimizer,
                                args=None,
                                stage_num=1,
                                fused=True,
                                opt_offload=opt.opt_offload,
                                flatten_weights=flatten_weights)

    callback_size = opt.sink_size if opt.acc_step <= 1 else opt.acc_step
    actual_epoch_num = int(epoch_num * step_per_epoch / callback_size)
    callback = [LossCallBack(callback_size)]

    config_ck = CheckpointConfig(save_checkpoint_steps=step_per_epoch, keep_checkpoint_max=1)
    ckpoint_cb = ModelCheckpoint(prefix=opt.arch,
                                 directory=opt.ckpt_save_dir + './ckpt_{}'.format(rank_id),
                                 config=config_ck)
    callback.append(ckpoint_cb)

    # CPU doest not support overflow check, so should use fixed loss scale.
    if mindspore.get_context('device_target').lower() == 'cpu':
        loss_scale_manager = FixedLossScaleManager(loss_scale=opt.init_loss_scale_value, drop_overflow_update=False)
    else:
        loss_scale_manager = DynamicLossScaleManager(init_loss_scale=opt.init_loss_scale_value,
                                                     scale_factor=opt.scale_factor, scale_window=opt.scale_window)

    print_model_size(net_with_loss, opt.logger)
    # Build the TrainOneStepCell
    net_wrapper = build_trainer(opt, net_with_loss, optim=optimizer,
                                update_cell=loss_scale_manager.get_update_cell())

    opt.logger.info("Start to compile the net and run.")
    if opt.acc_step > 1:
        opt.logger.info("Start to run gradient accumulation.")
        model = AccModel(net_wrapper)
        # Note: If accumulation is enabled, it only supports dataset sink mode
        model.train(actual_epoch_num, ds, callbacks=callback, dataset_sink_mode=True)
    else:
        model = Model(net_wrapper)
        model.train(actual_epoch_num, ds, callbacks=callback, sink_size=callback_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/gpt/gpt_base.yaml", help='YAML config files')
    args = parse_with_config(parser)
    args.logger = get_logger()
    set_seed(args.seed)
    run_train(args)
