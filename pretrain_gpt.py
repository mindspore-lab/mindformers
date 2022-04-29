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
GPT train script
"""

from mindspore import context, DynamicLossScaleManager, FixedLossScaleManager
from mindspore.train.model import Model
import mindspore.communication.management as D
from mindspore.train.callback import TimeMonitor, LossMonitor, ModelCheckpoint, CheckpointConfig
from mindspore.nn.transformer.loss import CrossEntropyLoss
from mindspore.nn.transformer import TransformerRecomputeConfig, MoEConfig, TransformerOpParallelConfig
import mindspore.common.dtype as mstype
from mindspore.common import set_seed

from transformer.data.dataset import create_dataset
from transformer.optimizer import get_optimizer
from transformer.models.gpt import GPTWithLoss, GPTConfig, GPT
from transformer.config_parser.parser import get_config
from transformer.learning_rate import LearningRate
from transformer.utils import download_data


def set_context_env(opt):
    """Set the context env"""
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=opt.device_target)
    if opt.device_target == "GPU":
        # Enable graph kernel
        context.set_context(enable_graph_kernel=True)
        context.set_context(graph_kernel_flags="--enable_parallel_fusion=true\
                            --parallel_ops_level=1\
                            --disable_cluster_ops=ReduceMax \
                            --disable_expand_ops=SoftmaxCrossEntropyWithLogits,Softmax,LogSoftmax")


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
    if opt.parallel_mode != "data_parallel":
        context.set_auto_parallel_context(full_batch=True)
    return rank_id, device_num


def get_parallel_config(opt):
    """Get the parallel config from the yaml file"""
    model_parallel_num = opt.model_parallel
    data_parallel_num = opt.data_parallel
    recompute_config = TransformerRecomputeConfig(recompute=opt.recompute,
                                                  parallel_optimizer_comm_recompute=True,
                                                  mp_comm_recompute=True,
                                                  recompute_slice_activation=True)
    parallel_config = TransformerOpParallelConfig(data_parallel=data_parallel_num,
                                                  model_parallel=model_parallel_num,
                                                  vocab_emb_dp=bool(opt.vocab_emb_dp),
                                                  recompute=recompute_config)

    parallel_config.moe_config = MoEConfig(expert_num=opt.expert_num,
                                           capacity_factor=2.0,
                                           num_experts_chosen=opt.per_token_num_experts_chosen)

    return parallel_config


def get_model_config(opt):
    """Get the model config from the yaml files"""
    config = GPTConfig(batch_size=opt.global_batch_size,
                       seq_length=opt.max_seq_length,
                       vocab_size=opt.vocab_size,
                       embedding_size=opt.hidden_size,
                       num_layers=opt.num_hidden_layers,
                       num_heads=opt.num_attention_heads,
                       expand_ratio=4,
                       post_layernorm_residual=False,
                       dropout_rate=0.1,
                       compute_dtype=mstype.float16)
    return config


def get_dataset(args_opt, rank_id, device_num):
    """get dataset from local or obs"""
    if args_opt.data_from_obs == "false":
        cache_url = args_opt.data_path
    else:
        # copy data from the cloud to the /cache/Data
        cache_url = '/cache/Data/'
        download_data(src_data_url=args_opt.data_url, tgt_data_path=cache_url, rank=rank_id)
    ds = create_dataset(args_opt.global_batch_size, data_path=cache_url, device_num=device_num, rank=rank_id)
    return ds


def run_train():
    """Main training process"""
    opt = get_config()
    set_context_env(opt)
    rank_id, device_num = set_auto_parallel_context_env(opt)
    parallel_config = get_parallel_config(opt)

    model_config = get_model_config(opt)
    model_config.parallel_config = parallel_config

    net = GPT(model_config)
    loss = CrossEntropyLoss(parallel_config.dp_mp_config)
    net_with_loss = GPTWithLoss(net, loss, model_config)

    ds = get_dataset(opt, rank_id, device_num)

    epoch_num = opt.epoch_size
    step_per_epoch = ds.get_dataset_size()

    lr = LearningRate(learning_rate=opt.start_lr,
                      end_learning_rate=opt.end_lr,
                      warmup_steps=opt.warmup_step,
                      decay_steps=epoch_num * step_per_epoch)

    optimizer = get_optimizer(net_with_loss, lr, opt.optimizer)

    callback_size = opt.sink_size
    actual_epoch_num = int(epoch_num * step_per_epoch / callback_size)
    callback = [TimeMonitor(callback_size), LossMonitor(callback_size)]

    config_ck = CheckpointConfig(save_checkpoint_steps=step_per_epoch, keep_checkpoint_max=1)
    ckpoint_cb = ModelCheckpoint(prefix="GPT3",
                                 directory=opt.ckpt_save_dir + './ckpt_{}'.format(rank_id),
                                 config=config_ck)
    callback.append(ckpoint_cb)

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
