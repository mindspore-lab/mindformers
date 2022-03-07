# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""Transformer training script."""

import os

from easydict import EasyDict as edict
import mindspore as ms
from mindspore.common.tensor import Tensor
from mindspore.nn.optim import Adam
from mindspore.train.model import Model
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.train.callback import TimeMonitor
from mindspore.nn.transformer import CrossEntropyLoss
import mindspore.communication.management as D
from mindspore.communication.management import get_rank
from mindspore.common import set_seed

from mindspore.nn.transformer import TransformerOpParallelConfig, MoEConfig
from src.callback import LossCallBack
from src.t5 import TransformerNetworkWithLoss, TransformerModel
from src.dataset import create_transformer_dataset
from src.lr_schedule import create_dynamic_lr
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper


def modelarts_pre_process():
    config.save_checkpoint_path = config.output_path
    config.data_path = os.path.join(config.data_path, 'ende-l128-mindrecord')


def set_context():
    ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target)
    ms.set_context(reserve_class_name_in_scope=False)
    # Set mempool block size in PYNATIVE_MODE for improving memory utilization, which will not take effect in GRAPH_MODE
    ms.set_context(mempool_block_size="31GB")
    if config.device_target == "GPU":
        # Enable graph kernel
        ms.set_context(enable_graph_kernel=True, graph_kernel_flags="--enable_parallel_fusion")


def set_auto_parallel_context():
    """Set the contexdt of auto_parallel"""
    if config.distribute == "true":
        D.init()
        device_num = D.get_group_size()
        rank = get_rank()
        config.device_id = rank
        ms.reset_auto_parallel_context()
        ms.set_auto_parallel_context(parallel_mode=config.parallel_mode, gradients_mean=True,
                                     device_num=device_num)
        rank_id = config.device_id % device_num
        save_ckpt_path = os.path.join(config.save_checkpoint_path, 'ckpt_' + str(get_rank()) + '/')
    else:
        device_num = 1
        rank_id = 0
        save_ckpt_path = os.path.join(config.save_checkpoint_path, 'ckpt_0/')
    return device_num, rank_id, save_ckpt_path


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_transformer_train():
    """
    Transformer training.
    """
    set_context()
    device_num, rank_id, save_ckpt_path = set_auto_parallel_context()
    dataset = create_transformer_dataset(global_batch_size=config.batch_size,
                                         rank_size=device_num,
                                         rank_id=rank_id,
                                         do_shuffle=config.do_shuffle,
                                         dataset_path=config.data_path,
                                         bucket_boundaries=config.bucket_boundaries)

    network = TransformerModel(config=config)
    loss = CrossEntropyLoss(parallel_config=config.parallel_config.dp_mp_config)
    netwithloss = TransformerNetworkWithLoss(network=network, loss=loss)
    netwithloss.set_train(True)

    if config.checkpoint_path:
        parameter_dict = ms.load_checkpoint(config.checkpoint_path)
        ms.load_param_into_net(netwithloss, parameter_dict)

    hidden_size = config.hidden_size
    learning_rate = config.lr_schedule.learning_rate if config.device_target == "Ascend" else 1.0
    lr = Tensor(create_dynamic_lr(schedule="constant*rsqrt_hidden*linear_warmup*rsqrt_decay",
                                  training_steps=dataset.get_dataset_size()*config.epoch_size,
                                  learning_rate=learning_rate,
                                  warmup_steps=config.lr_schedule.warmup_steps,
                                  hidden_size=hidden_size,
                                  start_decay_step=config.lr_schedule.start_decay_step,
                                  min_lr=config.lr_schedule.min_lr), ms.float32)

    optimizer = Adam(netwithloss.trainable_params(), lr)

    callbacks = [TimeMonitor(dataset.get_dataset_size()), LossCallBack(rank_id=rank_id)]
    if config.enable_save_ckpt == "true":
        if rank_id % 8 == 0:
            ckpt_config = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps,
                                           keep_checkpoint_max=config.save_checkpoint_num)
            ckpoint_cb = ModelCheckpoint(prefix='transformer', directory=save_ckpt_path, config=ckpt_config)
            callbacks.append(ckpoint_cb)

    scale_manager = ms.DynamicLossScaleManager(init_loss_scale=config.init_loss_scale_value,
                                               scale_factor=config.scale_factor,
                                               scale_window=config.scale_window)

    model = Model(netwithloss, optimizer=optimizer, loss_scale_manager=scale_manager)

    model.train(config.epoch_size, dataset, callbacks=callbacks, dataset_sink_mode=False)


def set_config(train_config):
    """Manually modify the config"""
    train_config.dtype = ms.float32
    train_config.compute_type = ms.float16
    train_config.lr_schedule = edict({
        'learning_rate': 2.0,
        'warmup_steps': 8000,
        'start_decay_step': 16000,
        'min_lr': 0.0,
    })

    train_config.bucket_boundaries = [16]
    train_config.seq_length = 16
    train_config.max_decode_length = 16

    train_config.parallel_config = TransformerOpParallelConfig(data_parallel=train_config.data_parallel,
                                                               model_parallel=train_config.model_parallel,
                                                               recompute=train_config.recompute,
                                                               pipeline_stage=train_config.pipeline_stage,
                                                               micro_batch_num=train_config.micro_batch_num,
                                                               optimizer_shard=train_config.optimizer_shard,
                                                               expert_parallel=train_config.expert_parallel_num,
                                                               vocab_emb_dp=train_config.vocab_emb_dp,
                                                               gradient_aggregation_group=
                                                               train_config.gradient_aggregation_group)

    train_config.moe_config = MoEConfig(expert_num=train_config.expert_num,
                                        capacity_factor=2.0,
                                        num_experts_chosen=train_config.per_token_num_experts_chosen)

if __name__ == '__main__':
    set_seed(1)
    set_config(config)
    run_transformer_train()
