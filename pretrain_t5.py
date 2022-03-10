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
Transformer train script
"""
from easydict import EasyDict as edict

from mindspore import context, DynamicLossScaleManager, FixedLossScaleManager, Tensor
from mindspore.train.model import Model
import mindspore.communication.management as D
from mindspore.train.callback import TimeMonitor, LossMonitor, ModelCheckpoint, CheckpointConfig
from mindspore.nn.transformer.loss import CrossEntropyLoss
from mindspore.nn.transformer import TransformerRecomputeConfig, TransformerOpParallelConfig, MoEConfig
import mindspore.common.dtype as mstype
from mindspore.common import set_seed

from transformer.optimizer import get_optimizer
from transformer.callback import LossSummaryCallback
from transformer.models.t5 import TransformerNetworkWithLoss, TransformerModel, TransformerConfig
from transformer.config_parser.parser import get_config
from transformer.learning_rate import create_dynamic_lr
from transformer.utils import print_mode_size

from transformer.data.t5_dataset import create_dataset


def set_context_env(opt):
    """Set the context env"""
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=opt.device_target)
    if opt.device_target == "GPU":
        # Enable graph kernel
        context.set_context(enable_graph_kernel=True, graph_kernel_flags="--enable_parallel_fusion")


def set_auto_parallel_context_env(opt):
    """Set the auto parallel env"""
    if opt.distribute == "true":
        D.init()
        device_num = opt.device_num
        rank_id = D.get_rank()
        print("rank_id is {}, device_num is {}".format(rank_id, device_num))

        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=opt.parallel_mode, gradients_mean=True,
                                          device_num=device_num)
    else:
        rank_id = 0
        device_num = 1

    return rank_id, device_num


def get_parallel_config(opt):
    """Get the parallel config from the yaml file"""
    recompute_config = TransformerRecomputeConfig(recompute=opt.recompute,
                                                  parallel_optimizer_comm_recompute=True,
                                                  mp_comm_recompute=True,
                                                  recompute_slice_activation=True)
    parallel_config = TransformerOpParallelConfig(data_parallel=opt.data_parallel,
                                                  model_parallel=opt.model_parallel,
                                                  recompute=recompute_config,
                                                  pipeline_stage=opt.pipeline_stage,
                                                  micro_batch_num=opt.micro_batch_num,
                                                  optimizer_shard=opt.optimizer_shard,
                                                  expert_parallel=opt.expert_parallel_num,
                                                  vocab_emb_dp=opt.vocab_emb_dp,
                                                  gradient_aggregation_group=
                                                  opt.gradient_aggregation_group)

    parallel_config.moe_config = MoEConfig(expert_num=opt.expert_num,
                                           capacity_factor=2.0,
                                           num_experts_chosen=opt.per_token_num_experts_chosen)

    return parallel_config


def get_model_config(opt):
    """Get the model config from the yaml files"""
    config = TransformerConfig(batch_size=opt.global_batch_size,
                               seq_length=opt.max_seq_length,
                               max_decode_length=opt.max_decode_length,
                               vocab_size=opt.vocab_size,
                               hidden_size=opt.hidden_size,
                               num_hidden_layers=opt.num_hidden_layers,
                               num_attention_heads=opt.num_attention_heads,
                               intermediate_size=4 * opt.hidden_size,
                               hidden_dropout_prob=opt.hidden_dropout_prob,
                               attention_probs_dropout_prob=opt.attention_probs_dropout_prob,
                               compute_type=mstype.float16)
    return config


def run_train():
    """Main training process"""
    opt = get_config()
    set_context_env(opt)
    rank_id, device_num = set_auto_parallel_context_env(opt)
    parallel_config = get_parallel_config(opt)

    model_config = get_model_config(opt)
    model_config.parallel_config = parallel_config

    network = TransformerModel(config=model_config)
    loss = CrossEntropyLoss(parallel_config=parallel_config.dp_mp_config)
    net_with_loss = TransformerNetworkWithLoss(network=network, loss=loss)
    net_with_loss.set_train(True)

    print_mode_size(net_with_loss)
    ds = create_dataset(opt.batch_size, data_path=opt.data_path, device_num=device_num, rank=rank_id,
                        bucket_boundaries=opt.bucket_boundaries)

    epoch_num = opt.epoch_size
    step_per_epoch = ds.get_dataset_size()

    opt.lr_schedule = edict({
        'learning_rate': 2.0,
        'warmup_steps': 8000,
        'start_decay_step': 16000,
        'min_lr': 0.0,
    })

    learning_rate = opt.lr_schedule.learning_rate if opt.device_target == "Ascend" else 1.0
    lr = Tensor(create_dynamic_lr(schedule="constant*rsqrt_hidden*linear_warmup*rsqrt_decay",
                                  training_steps=step_per_epoch*opt.epoch_size,
                                  learning_rate=learning_rate,
                                  warmup_steps=opt.lr_schedule.warmup_steps,
                                  hidden_size=opt.hidden_size,
                                  start_decay_step=opt.lr_schedule.start_decay_step,
                                  min_lr=opt.lr_schedule.min_lr), mstype.float32)

    optimizer = get_optimizer(net_with_loss, lr, opt.optimizer)

    callback_size = opt.sink_size
    actual_epoch_num = int(epoch_num * step_per_epoch/callback_size)
    callback = [TimeMonitor(callback_size), LossMonitor(callback_size)]

    config_ck = CheckpointConfig(save_checkpoint_steps=step_per_epoch, keep_checkpoint_max=1)
    ckpoint_cb = ModelCheckpoint(prefix="t5",
                                 directory=opt.ckpt_save_dir + './ckpt_{}'.format(rank_id),
                                 config=config_ck)
    callback.append(ckpoint_cb)

    loss_summary_callback = LossSummaryCallback(summary_dir=f'./summary_dir_{rank_id}')
    callback.append(loss_summary_callback)

    # CPU doest not support overflow check, so should use fixed loss scale.
    if opt.device_target == 'CPU':
        loss_scale_manager = FixedLossScaleManager(loss_scale=opt.init_loss_scale_value, drop_overflow_update=False)
    else:
        loss_scale_manager = DynamicLossScaleManager(init_loss_scale=opt.init_loss_scale_value,
                                                     scale_factor=opt.scale_factor, scale_window=opt.scale_window)

    model = Model(net_with_loss, optimizer=optimizer, loss_scale_manager=loss_scale_manager)
    model.train(actual_epoch_num, ds, callbacks=callback, sink_size=callback_size)


if __name__ == "__main__":
    set_seed(12315)
    run_train()
