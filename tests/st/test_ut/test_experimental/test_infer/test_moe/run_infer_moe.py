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
"""run transformer block in infer mode"""
import argparse
import os

import numpy as np

import mindspore as ms
from mindspore import Tensor
from mindspore.communication import get_group_size, init

from mindformers.models.utils import convert_mstype
from research.deepseek3.deepseek3_config import DeepseekV3Config

from tests.st.test_ut.test_experimental.test_infer.test_moe.utils import ExpertParallelMoENet

def set_config(expert_parallel):
    """Set config for mlp ut test."""
    config_ = DeepseekV3Config()
    config_.batch_size = 1
    config_.seq_length = 32
    config_.hidden_size = 64
    config_.intermediate_size = 128
    config_.ffn_concat = True
    config_.compute_dtype = convert_mstype("bfloat16")
    config_.param_init_dtype = convert_mstype("bfloat16")
    config_.param_init_type = convert_mstype("bfloat16")
    config_.layernorm_compute_type = convert_mstype("bfloat16")
    config_.pad_token_id = 1
    config_.mlp_has_bias = False

    config_.moe_config.expert_num = 256
    config_.moe_config.num_experts_chosen = 8
    config_.moe_config.routing_policy = "TopkRouterV2"
    config_.moe_config.shared_expert_num = 1
    config_.moe_config.routed_scaling_factor = 2.5
    config_.moe_config.first_k_dense_replace = 3
    config_.moe_config.moe_intermediate_size = 2048
    config_.moe_config.topk_group = 4
    config_.moe_config.n_group = 8
    config_.moe_config.router_dense_type = "bfloat16"

    world_size = get_group_size()
    config_.parallel_config.use_alltoall = expert_parallel == world_size
    config_.parallel_config.expert_parallel = expert_parallel
    config_.parallel_config.use_sequence_parallel = False
    return config_

def generate_inputs(config):
    """Generate input tensors for transformer block or layer inference test."""
    bs = config.batch_size
    seq_len = config.seq_length
    hidden_size = config.hidden_size

    input_shape = (bs * seq_len, hidden_size)
    hidden_states = Tensor(np.random.standard_normal(input_shape).astype(np.bfloat16))
    return hidden_states

def _test_parallel_moe(expert_parallel):
    """Test the Transformer Block module in inference mode."""
    os.environ["RUN_MODE"] = "predict"
    jit_level = "O0"
    infer_boost = "on"
    ms.set_context(device_target="Ascend",
                   mode=ms.GRAPH_MODE,
                   jit_config={"jit_level": jit_level, "infer_boost": infer_boost})
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.STAND_ALONE, full_batch=False)
    init()

    seed_value = 16
    ms.set_seed(seed_value)
    np.random.seed(seed_value)

    config = set_config(expert_parallel)
    hidden_states = generate_inputs(config)
    net = ExpertParallelMoENet(config)

    output = net(hidden_states)
    assert output.shape == (config.batch_size * config.seq_length, config.hidden_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_parallel', type=str, help='test moe module in inference mode')

    args = parser.parse_args()
    ep_size = int(args.expert_parallel)
    _test_parallel_moe(ep_size)
