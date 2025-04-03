# Copyright 2025 Huawei Technologies Co., Ltd
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
"""run MoE"""

import argparse
import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.communication.management import get_rank, init
from mindspore.nn import AdamWeightDecay, SoftmaxCrossEntropyWithLogits

from mindformers import MoEConfig as GoldenMoEConfig
from mindformers.experimental.parallel_core.pynative.config import (
    LoraConfig,
    ModelParallelConfig,
    TransformerConfig,
    TrainingConfig
)
from mindformers.modules.transformer.op_parallel_config import default_dpmp_config
from mindformers.modules.transformer.moev3 import MoEV3
from tests.st.test_distri_core.utils import train
import numpy as np

class TestData:
    """
    generate a test dataset
    """
    def __init__(self, data_size=None, input_data=None, label_data=None):
        super().__init__()
        _ = data_size
        self.input_data = input_data
        self.label_data = label_data

    def __getitem__(self, index):
        return (Tensor(self.input_data[index]), Tensor(self.label_data[index]))

    def __len__(self):
        return self.input_data.shape[0]


class MoEV3Net(nn.Cell):
    """
    define a graph MoE net
    """
    def __init__(self,
                 hidden_size,
                 moe_config,
                 compute_dtype=mstype.float16,
                 param_init_type=mstype.float32,
                 parallel_config=default_dpmp_config,):
        super(MoEV3Net, self).__init__()
        self.feed_forward = MoEV3(dim=hidden_size,
                                  intermediate_size=4*hidden_size,
                                  compute_dtype=compute_dtype,
                                  param_init_type=param_init_type,
                                  moe_config=moe_config,
                                  parallel_config=parallel_config)

        self.loss = SoftmaxCrossEntropyWithLogits()
        self.cast = ops.Cast()
        self.rank_id = get_rank()
        if moe_config.expert_num == 1:
            self.feed_forward.shard(parallel_config)
        elif moe_config.shared_expert_num == 0:
            self.feed_forward.ffn.shard(parallel_config)
        else:
            self.feed_forward.shard(parallel_config)

    def construct(self, hidden_states, label):
        """define a forward process"""
        output = self.cast(hidden_states, mstype.float16)
        output = self.feed_forward(output)
        output = self.cast(output, mstype.float32)

        output = ops.reshape(output, label.shape)
        loss = ops.dist(output, label.to(ms.float32), p=1) / (16*8)

        return loss


def generate_golden(model_config, args):
    """
    run graph mode moe to generate golden ckpt and loss
    """
    parallel_config = model_config.parallel_config
    moe_config = model_config.moe_config

    dp = parallel_config.data_parallel
    mp = parallel_config.model_parallel
    ms.set_context(device_target="Ascend", mode=ms.GRAPH_MODE, deterministic='ON', pynative_synchronize=True)
    init()
    rank_id = get_rank()
    ms.reset_auto_parallel_context()
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.STAND_ALONE)
    ms.set_seed(1921)
    hidden_states = np.load("data/golden_moe_input_and_loss.npz")['input']
    dataset = TestData(data_size=args.dataset_size, input_data=hidden_states, label_data=hidden_states)
    dataset = ds.GeneratorDataset(dataset, column_names=['hidden_states', 'label'],
                                  num_shards=dp, shard_id=rank_id // mp)
    dataset = dataset.batch(args.batch_size)
    network = MoEV3Net(hidden_size=model_config.hidden_size,
                       moe_config=moe_config,
                       compute_dtype=model_config.compute_dtype,
                       param_init_type=model_config.param_init_type,
                       parallel_config=parallel_config)
    optimizer = AdamWeightDecay(params=network.get_parameters())
    train(1, dataset, network, optimizer, None, reduce_grad=False)

def run_moe(model_config, args):
    """
    run graph mode moe to generate golden ckpt and loss
    """
    parallel_config = model_config.parallel_config
    moe_config = model_config.moe_config

    dp = parallel_config.data_parallel
    mp = parallel_config.model_parallel
    ms.set_context(device_target="Ascend", mode=ms.GRAPH_MODE, deterministic='ON', pynative_synchronize=True)
    init()
    rank_id = get_rank()
    ms.reset_auto_parallel_context()
    ms.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    ms.set_seed(1921)
    hidden_states = np.load("data/golden_moe_input_and_loss.npz")['input']
    dataset = TestData(data_size=args.dataset_size, input_data=hidden_states, label_data=hidden_states)
    dataset = ds.GeneratorDataset(dataset, column_names=['hidden_states', 'label'],
                                  num_shards=dp, shard_id=rank_id // mp)
    dataset = dataset.batch(args.batch_size)
    network = MoEV3Net(hidden_size=model_config.hidden_size,
                       moe_config=moe_config,
                       compute_dtype=model_config.compute_dtype,
                       param_init_type=model_config.param_init_type,
                       parallel_config=parallel_config)
    optimizer = AdamWeightDecay(params=network.get_parameters())
    loss_list = train(1, dataset, network, optimizer, None, reduce_grad=False)

    # save golden input and loss
    golden_loss = np.load("data/golden_moe_input_and_loss.npz")['loss']
    loss_list = np.array([x for x in loss_list])
    assert np.allclose(golden_loss, loss_list, rtol=1e-3), \
            f"relative error between pynative loss and golden loss exceeds 1e-3, please check your code."


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_golden', action='store_true', help="Generate golden data for test.")
    parser.add_argument('--dp', type=int, default=2, help="data_parallel")
    parser.add_argument('--mp', type=int, default=1, help="model_parallel")
    parser.add_argument('--ep', type=int, default=2, help="expert_model_parallel_size")
    parser.add_argument('--batch_size', type=int, default=1, help="batch_size")
    parser.add_argument('--hidden_act', type=str, default='swiglu', help="activation")
    parser.add_argument('--dataset_size', type=int, default=32, help="dataset_size")
    parser.add_argument('--aux_loss', action='store_true', help="use aux_loss load balancing type.")
    parser.add_argument('--aux_loss_free', action='store_true', help="use aux_loss_free load balancing type.")
    parser.add_argument('--z_loss_coeff', type=float, default=None, help="use aux_loss load balancing type.")
    parser.add_argument('--use_seq_parallel', default=False, help="use short sequence parallel.", action='store_true')
    parser.add_argument('--comp_comm_parallel', default=False, help="use comp comm parallel.", action='store_true')
    parser.add_argument('--comp_comm_parallel_degree', type=int, default=2, help="comp comm parallel degree.")
    parser.add_argument('--use_allgather_dispatcher', type=bool, default=False, help="use allgather dispatcher")
    parser.add_argument('--use_gmm', type=bool, default=False, help="use GroupedMatmul")


    cli_args, rest_args = parser.parse_known_args()
    balancing_type = 'none'
    if cli_args.aux_loss:
        balancing_type = 'aux_loss'

    parallel_cfg = ModelParallelConfig(
        data_parallel=cli_args.dp,
        model_parallel=cli_args.mp,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=cli_args.ep,
        use_seq_parallel=False,
        seq_split_num=1
    )
    if cli_args.use_seq_parallel or cli_args.comp_comm_parallel:
        parallel_cfg.model_parallel = cli_args.mp
        parallel_cfg.expert_parallel = cli_args.ep
    if cli_args.use_seq_parallel:
        parallel_cfg.use_seq_parallel = True

    moe_cfg_golden = GoldenMoEConfig(
        expert_num=16,
        capacity_factor=-1,
        aux_loss_factor=0.05,
        num_experts_chosen=1,
        routing_policy="TopkRouterV2",
        enable_sdrop=False,
        router_dense_type='float32'
    )
    training_cfg = TrainingConfig(parallel_config=parallel_cfg)
    model_cfg = TransformerConfig(
        vocab_size=1,
        num_layers=1,
        num_attention_heads=1,
        seq_length=8,
        hidden_size=16,
        ffn_hidden_size=64,
        gated_linear_unit=True,
        hidden_act=cli_args.hidden_act,
        qkv_has_bias=True,
        mlp_has_bias=False,
        params_dtype='float32',
        param_init_type=mstype.float32,
        compute_dtype='float32',
        fp32_residual_connection=False,
        parallel_config=parallel_cfg,
        moe_config=None,
        lora_config=LoraConfig(use_lora=False),
        training_config=training_cfg
    )
    if cli_args.use_gmm:
        moe_cfg_golden.use_gmm = True
        model_cfg.parallel_config.expert_parallel = model_cfg.parallel_config.expert_model_parallel_size

    if cli_args.generate_golden:
        model_cfg.parallel_config.context_parallel = model_cfg.parallel_config.context_parallel_size
        if cli_args.aux_loss_free:
            moe_cfg_golden.balance_via_topk_bias = True
            moe_cfg_golden.topk_bias_update_rate = 0.001
            model_cfg.parallel_config.expert_parallel = model_cfg.parallel_config.expert_model_parallel_size
        if cli_args.comp_comm_parallel or cli_args.use_seq_parallel:
            moe_cfg_golden.capacity_factor = 1.1
        if cli_args.comp_comm_parallel:
            moe_cfg_golden.comp_comm_parallel = cli_args.comp_comm_parallel
            moe_cfg_golden.comp_comm_parallel_degree = cli_args.comp_comm_parallel_degree
        if cli_args.use_allgather_dispatcher:
            moe_cfg_golden.use_allgather_dispatcher = True
            model_cfg.parallel_config.expert_parallel = model_cfg.parallel_config.expert_model_parallel_size
        model_cfg.moe_config = moe_cfg_golden
        generate_golden(model_cfg, cli_args)
    else:
        model_cfg.moe_config = moe_cfg_golden
        run_moe(model_cfg, cli_args)
