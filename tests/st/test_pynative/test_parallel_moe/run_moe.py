# Copyright 2024 Huawei Technologies Co., Ltd
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
import os

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.communication.management import get_rank, init
from mindspore.nn import AdamWeightDecay, SoftmaxCrossEntropyWithLogits

from mindformers import MoEConfig as GoldenMoEConfig
from mindformers.experimental.distri_cores.config import (
    LoraConfig,
    ModelParallelConfig,
    MoEConfig,
    TransformerConfig
)
from mindformers.experimental.distri_cores.create_comm import (
    get_ep_group,
    get_dp_group,
    get_ep_rank,
    get_pp_group,
    get_tp_group,
    get_cp_group,
    initialize_model_parallel,
)
from mindformers.experimental.distri_cores.transformer.moe.moe_layer import MoELayer
from mindformers.models.llama.llama_layer import LlamaFeedForward
from mindformers.modules.transformer.moe import MoEV2
from mindformers.modules.transformer.op_parallel_config import default_dpmp_config

from tests.st.test_pynative.utils import transform_moe_golden_params_to_pynative_params, train

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


class GoldenMoENet(nn.Cell):
    """
    define a graph MoE net
    """
    def __init__(self,
                 hidden_size,
                 moe_config,
                 intermediate_size=None,
                 expert_num=1,
                 hidden_act=None,
                 ffn_dim_multiplier=None,
                 compute_dtype=mstype.float16,
                 param_init_type=mstype.float32,
                 parallel_config=default_dpmp_config,):
        super(GoldenMoENet, self).__init__()
        ffn = LlamaFeedForward(dim=hidden_size,
                               intermediate_size=intermediate_size,
                               expert_num=expert_num,
                               hidden_act=hidden_act,
                               ffn_dim_multiplier=ffn_dim_multiplier,
                               compute_dtype=compute_dtype,
                               param_init_type=param_init_type,
                               is_dynamic=False,
                               parallel_config=parallel_config)
        self.feed_forward = MoEV2(ffn=ffn,
                                  dim=hidden_size,
                                  moe_config=moe_config,
                                  parallel_config=parallel_config)

        self.loss = SoftmaxCrossEntropyWithLogits()
        self.cast = ops.Cast()
        self.rank_id = get_rank()
        self.all_gather = ops.AllGather()

    def construct(self, hidden_states, label):
        """define a forward process"""
        output = self.cast(hidden_states, mstype.float16)
        output = self.feed_forward(output)
        output = self.cast(output, mstype.float32)

        output = ops.reshape(output, label.shape)
        loss = ops.dist(output, label.to(ms.float32), p=1) / (16 * 8)

        return loss


class PynativeMoENet(nn.Cell):
    """
    define a pynative MoE net
    """
    def __init__(self, config):
        super(PynativeMoENet, self).__init__()
        self.moe = MoELayer(config=config)
        self.loss = SoftmaxCrossEntropyWithLogits()
        self.cast = ops.Cast()
        self.rank_id = get_ep_rank()
        self.all_gather = ops.AllGather(group=get_dp_group())

    def construct(self, hidden_states, label):
        """define a forward process"""
        output = self.cast(hidden_states, mstype.float16)

        output, _ = self.moe(output)
        output = self.cast(output, mstype.float32)

        output = ops.reshape(output, label.shape)
        loss = ops.dist(output, label.to(ms.float32), p=1) / (16 * 8)

        output_all = self.all_gather(output)
        label_all = self.all_gather(label)
        loss_all = ops.dist(output_all, label_all.to(ms.float32), p=1) / (16 * 8)
        print(f"loss_all is {loss_all}")
        return loss


def generate_golden(model_config, args):
    """
    run graph mode moe to generate golden ckpt and loss
    """
    parallel_config = model_config.parallel_config
    moe_config = model_config.moe_config

    dp = parallel_config.data_parallel

    ms.set_context(device_target="Ascend", mode=ms.GRAPH_MODE, deterministic='ON', pynative_synchronize=True)
    init()
    rank_id = get_rank()
    ms.reset_auto_parallel_context()
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.STAND_ALONE)

    ms.set_seed(1921)
    hidden_states = np.random.normal(size=(args.dataset_size, model_config.seq_length, model_config.hidden_size))
    hidden_states = hidden_states.astype(np.float32)
    dataset = TestData(data_size=args.dataset_size, input_data=hidden_states, label_data=hidden_states)
    dataset = ds.GeneratorDataset(dataset, column_names=['hidden_states', 'label'], num_shards=dp, shard_id=rank_id)
    dataset = dataset.batch(args.batch_size)

    network = GoldenMoENet(hidden_size=model_config.hidden_size,
                           moe_config=moe_config,
                           intermediate_size=model_config.ffn_hidden_size,
                           expert_num=moe_config.expert_num,
                           hidden_act=model_config.hidden_act,
                           compute_dtype=model_config.compute_dtype,
                           param_init_type=model_config.param_init_type,
                           parallel_config=parallel_config)
    ms.save_checkpoint(network, "./data/golden_moe.ckpt")
    optimizer = AdamWeightDecay(params=network.get_parameters())
    loss_list = train(1, dataset, network, optimizer, None, reduce_grad=False)

    # save golden input and loss
    loss_list = np.array([x for x in loss_list])
    input_and_loss = {"input": hidden_states,
                      "loss": loss_list}
    np.save("./data/golden_moe_input_and_loss.npy", input_and_loss)


def run_moe_pynative(model_config, args):
    """
    run pynative mode moe and load golden ckpt to generate pynative loss
    """
    # prepare some config
    parallel_config = model_config.parallel_config
    moe_config = model_config.moe_config

    dp = parallel_config.data_parallel
    tp = parallel_config.model_parallel
    ep = parallel_config.expert_parallel
    en = moe_config.num_experts

    # init parallel env
    init()
    print("data_parallel {}, tensor_parallel {}, expert_parallel {}, num_experts {}".format(dp, tp, ep, en))
    initialize_model_parallel(expert_model_parallel_size=ep, order='tp-ep-dp-pp-cp')
    print("dp group {}, tp group {}, pp group {}, ep group {}, cp group {}".format \
         (get_dp_group(), get_tp_group(), get_pp_group(), get_ep_group(), get_cp_group()))
    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE)
    rank_id = get_rank()
    local_expert_idx = np.arange(en).reshape(ep, -1)[rank_id].tolist()

    ms.set_seed(1921)

    # load data
    golden_input_and_loss_path = "./data/golden_moe_input_and_loss.npy"
    assert os.path.exists(golden_input_and_loss_path), \
           f"'{golden_input_and_loss_path}' did not exits, please run generate_golden() to " + \
            "generate one by running below command: \n`pytest -sv test_moe.py::TestMoE::test_moe_golden`"

    input_and_loss = np.load(golden_input_and_loss_path, allow_pickle=True).tolist()
    hidden_states = input_and_loss['input']
    # generate data
    dataset = TestData(data_size=args.dataset_size, input_data=hidden_states, label_data=hidden_states)
    dataset = ds.GeneratorDataset(dataset, column_names=['hidden_states', 'label'], num_shards=dp, shard_id=rank_id)
    dataset = dataset.batch(args.batch_size)

    network = PynativeMoENet(config=model_config)

    # load golden ckpt
    golden_ckpt_path = "./data/golden_moe.ckpt"
    assert os.path.exists(golden_ckpt_path), \
           "'./data/golden_moe.ckpt' did not exits, please run generate_golden() to " + \
           "generate one by running below command: \n`pytest -sv test_moe.py::TestMoE::test_moe_golden`"
    golden_params = ms.load_checkpoint(golden_ckpt_path)
    pynative_params = transform_moe_golden_params_to_pynative_params(golden_params, local_expert_idx)
    param_not_load, _ = ms.load_param_into_net(network, pynative_params)
    assert not param_not_load, f"{param_not_load} was not loaded in this net, test failed."
    print("load ckpt competele.", flush=True)

    # perform train
    optimizer = AdamWeightDecay(params=network.get_parameters())
    if args.aux_loss or args.z_loss_coeff:
        network.set_train(True)
    train(1, dataset, network, optimizer, None, reduce_grad=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_golden', action='store_true', help="Generate golden data for test.")
    parser.add_argument('--dp', type=int, default=2, help="data_parallel")
    parser.add_argument('--ep', type=int, default=2, help="expert_parallel")
    parser.add_argument('--batch_size', type=int, default=1, help="batch_size")
    parser.add_argument('--dataset_size', type=int, default=20, help="dataset_size")
    parser.add_argument('--aux_loss', action='store_true', help="use aux_loss load balancing type.")
    parser.add_argument('--z_loss_coeff', type=float, default=None, help="use aux_loss load balancing type.")

    cli_args, rest_args = parser.parse_known_args()
    balancing_type = 'none'
    if cli_args.aux_loss:
        balancing_type = 'aux_loss'

    parallel_cfg = ModelParallelConfig(
        data_parallel=cli_args.dp,
        model_parallel=1,
        pipeline_stage=1,
        expert_parallel=cli_args.ep,
        use_seq_parallel=False
        )
    moe_cfg = MoEConfig(
        num_experts=4,
        moe_router_topk=1,
        add_bias_linear=False,
        moe_token_dispatcher_type='alltoall',
        moe_z_loss_coeff=cli_args.z_loss_coeff,
        moe_aux_loss_coeff=1e-2,
        moe_router_load_balancing_type=balancing_type, # ['none', 'aux_loss'],
        moe_input_noise_eps=None,
        moe_expert_capacity_factor=None,
        moe_token_drop_policy=None,
        moe_pad_expert_input_to_capacity=False,
        use_self_defined_alltoall=False,
        )
    moe_cfg_golden = GoldenMoEConfig(
        expert_num=moe_cfg.num_experts,
        capacity_factor=-1,
        aux_loss_factor=0.05,
        num_experts_chosen=1,
        routing_policy="TopkRouterV2",
        enable_sdrop=False,
        router_dense_type='float32'
        )
    model_cfg = TransformerConfig(
        vocab_size=1,
        num_layers=1,
        num_heads=1,
        seq_length=8,
        hidden_size=16,
        ffn_hidden_size=64,
        mlp_has_gate=True,
        hidden_act="gelu",
        qkv_has_bias=True,
        mlp_has_bias=False,
        param_init_dtype='float32',
        param_init_type=mstype.float32,
        compute_dtype='float32',
        fp32_residual_connection=False,
        parallel_config=parallel_cfg,
        moe_config=moe_cfg,
        lora_config=LoraConfig(use_lora=False)
        )

    if cli_args.generate_golden:
        model_cfg.moe_config = moe_cfg_golden
        generate_golden(model_cfg, cli_args)
    else:
        run_moe_pynative(model_cfg, cli_args)
