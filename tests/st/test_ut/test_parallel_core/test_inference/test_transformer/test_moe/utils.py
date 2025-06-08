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
"""test infer moe utils"""
import numpy as np

from mindspore import nn, Tensor, Parameter
from mindspore.ops import operations as P

from mindformers.parallel_core.transformer_config import TransformerConfig

from mindformers.parallel_core.utils.spec_utils import build_module
from mindformers.parallel_core.inference.base_models.gpt.moe_module_spec import get_moe_module_spec

from research.deepseek3.moe import RoutedParallelMLP, SharedParallelMLP, ParallelMoEV2


def get_init_params(hidden_size, num_experts, moe_intermediate_size):
    """Generate initialization parameters"""
    np.random.seed(2025)
    gate_weight_shape = (num_experts, hidden_size)
    gate_e_score_correction_bias_shape = (num_experts,)
    experts_fc1_weight_shape = (num_experts, hidden_size, moe_intermediate_size * 2)
    experts_fc2_weight_shape = (num_experts, moe_intermediate_size, hidden_size)
    shared_experts_fc1_weight_shape = (moe_intermediate_size * 2, hidden_size)
    shared_experts_fc2_weight_shape = (hidden_size, moe_intermediate_size)
    return {
        "input": np.random.rand(2 * 2, hidden_size),
        "moe.router.weight.weight": Parameter(0.01 * np.random.rand(*gate_weight_shape)),
        "moe.router.expert_bias": Parameter(0.01 * np.random.rand(*gate_e_score_correction_bias_shape)),
        "moe.experts.weight1": Parameter(0.01 * np.random.rand(*experts_fc1_weight_shape)),
        "moe.experts.weight2": Parameter(0.01 * np.random.rand(*experts_fc2_weight_shape)),
        "moe.shared_experts.linear_fc1.weight": Parameter(0.01 * np.random.rand(*shared_experts_fc1_weight_shape)),
        "moe.shared_experts.linear_fc2.weight": Parameter(0.01 * np.random.rand(*shared_experts_fc2_weight_shape)),
    }


def convert_weight_name(param_dict):
    """Convert weight name."""

    for name, param in list(param_dict.items()):
        weight_name = name.replace('moe.router.expert_bias', 'moe.routed_experts.router.e_score_correction_bias')
        weight_name = weight_name.replace('moe.router.weight.weight', 'moe.routed_experts.router.dense.weight')
        weight_name = weight_name.replace('moe.experts.weight1', 'moe.routed_experts.ffn.w_gate_hidden.weight')
        weight_name = weight_name.replace('moe.experts.weight2', 'moe.routed_experts.ffn.w2.weight')
        weight_name = weight_name.replace('moe.shared_experts.linear_fc1.weight',
                                          'moe.shared_experts.w_gate_hidden.weight')
        new_name = weight_name.replace('moe.shared_experts.linear_fc2.weight', 'moe.shared_experts.w2.weight')

        param.name = new_name
        if new_name != name:
            param_dict[new_name] = param_dict.pop(name)

    return param_dict


class NewMoENet(nn.Cell):
    """A model class of new moe."""
    def __init__(self, config: TransformerConfig):
        super(NewMoENet, self).__init__()
        self.moe = build_module(
            get_moe_module_spec(num_experts=config.num_moe_experts),
            config=config,
            layer_number=1,
        )

    def construct(self, hidden_states: Tensor):
        output = self.moe(hidden_states)
        return output


class OldMoENet(nn.Cell):
    """A model class of old moe."""
    def __init__(self, config):
        super(OldMoENet, self).__init__()
        config.param_init_dtype = config.param_init_type
        config.parallel_config.use_sequence_parallel = False
        self.moe = OldMoELayer(config=config)

    def construct(self, hidden_states: Tensor):
        output = self.moe(hidden_states)
        return output


class OldMoELayer(nn.Cell):
    """A class of Old Moe Layer."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.parallel_config = config.parallel_config
        self.moe_config = config.moe_config
        self.moe_config.router_dense_type = config.router_dense_type
        intermediate_size = self.moe_config.moe_intermediate_size

        ffn = RoutedParallelMLP(config)
        self.routed_experts = ParallelMoEV2(ffn, self.config.hidden_size, self.moe_config)
        intermediate_size = intermediate_size * self.moe_config.shared_expert_num
        self.shared_experts = SharedParallelMLP(config, intermediate_size)
        self.add = P.Add()

    def construct(self, hidden_states: Tensor):
        output = self.routed_experts(hidden_states)
        output = self.add(output, self.shared_experts(hidden_states))

        return output
