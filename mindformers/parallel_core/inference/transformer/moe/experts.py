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
"""Expert GrouedMLP."""

__all__ = ["GroupedMLP"]

from typing import Optional

from mindspore import mint, nn

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.utils.spec_utils import build_module
from mindformers.parallel_core.inference.transformer.mlp import MLPSubmodules
from mindformers.parallel_core.inference.tensor_parallel.gemm_layers import UnquantizedGroupedLinearMethod
from mindformers.parallel_core.inference.tensor_parallel.quantization import QuantizationConfig
from mindformers.parallel_core.inference.tensor_parallel.quantization.base_config import QuantizeMethodBase
from mindformers.parallel_core.inference.transformer.activation import get_act_func
from mindformers.parallel_core.inference.utils import divide
from mindformers.parallel_core.process_group_config import ModelCommProcessGroups, default_model_comm_pgs
from mindformers.parallel_core.inference.weights_utils import set_weight_attrs


class GroupedMLP(nn.Cell):
    """An implementation of the Experts layer.

    Executes multiple experts in parallel to maximize computational efficiency.
    """

    def __init__(
            self,
            num_local_experts: int,
            config: TransformerConfig,
            submodules: MLPSubmodules,
            model_comm_pgs: Optional[ModelCommProcessGroups] = default_model_comm_pgs,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
    ):
        super().__init__()
        self.config: TransformerConfig = config
        self.num_local_experts = num_local_experts
        self.input_size = self.config.hidden_size
        # use model_comm_pgs.moe_tp_group as tensor parallel group in this module.
        self.tp_group = model_comm_pgs.moe_tp
        self.tp_group_size = self.tp_group.size

        ffn_hidden_size = self.config.moe_ffn_hidden_size
        self.ffn_hidden_size_per_partition = divide(ffn_hidden_size, self.tp_group_size)
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2

        self.activation_type = self.config.hidden_act

        # create weights
        # Note: Currently does not support quantization, only use UnquantizedGroupedLinearMethod.
        self.quant_method: Optional[QuantizeMethodBase] = UnquantizedGroupedLinearMethod()
        if quant_config is not None:
            self.quant_method = quant_config.get_quant_method(self, prefix)
        self.weight1 = self.quant_method.create_weights(
            layer=None,
            num_local_experts=self.num_local_experts,
            input_size_per_partition=self.input_size,
            output_partition_sizes=[divide(ffn_hidden_size, self.tp_group_size)],
            params_dtype=self.config.params_dtype,
        )
        self.weight2 = self.quant_method.create_weights(
            layer=None,
            num_local_experts=self.num_local_experts,
            input_size_per_partition=self.ffn_hidden_size_per_partition,
            output_partition_sizes=[self.input_size],
            params_dtype=self.config.params_dtype,
        )

        # linear fc1
        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.num_local_experts,
            self.input_size,
            ffn_hidden_size,
            config=self.config,
            bias=self.config.add_bias_linear,
            gather_output=False,
            weight=self.weight1, # Skip creating weights and use weight1 for gemm linear calculation
            is_expert=True,
            tp_group=self.tp_group,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_fc1",
        )

        if self.activation_type is not None:
            self.activation_func = get_act_func(self.activation_type)
        else:
            self.activation_func = None

        # linear fc2
        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.num_local_experts,
            self.config.moe_ffn_hidden_size,
            self.input_size,
            config=self.config,
            bias=self.config.add_bias_linear,
            weight=self.weight2, # Skip creating weights and use weight2 for gemm linear calculation
            is_expert=True,
            tp_group=self.tp_group,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_fc2",
        )

        set_weight_attrs(self.weight1, {"weight_loader": self.linear_fc1.weight_loader})
        set_weight_attrs(self.weight2, {"weight_loader": self.linear_fc2.weight_loader})

    def construct(self, hidden_states, group_list=None):
        """Forward process of GroupedMLP"""
        # [T, H] -> [T, ffn_H]
        intermediate_parallel = self.linear_fc1(hidden_states, group_list=group_list)

        if self.config.gated_linear_unit:
            gate, hidden = mint.split(intermediate_parallel,
                                      (self.ffn_hidden_size_per_partition,
                                       self.ffn_hidden_size_per_partition), -1)
            gate = self.activation_func(gate) if self.activation_type else gate
            intermediate_parallel = mint.mul(hidden, gate)
        else:
            intermediate_parallel = self.activation_func(
                intermediate_parallel) if self.activation_type else intermediate_parallel

        # [T, ffn_H] -> [T, H]
        output = self.linear_fc2(intermediate_parallel, group_list=group_list)
        return output

    def sharded_state_dict(self):
        """Provide the sharded state dict."""
        expert_parallel_group_size = self.config.num_moe_experts // self.num_local_experts
        w1_shard = (expert_parallel_group_size, 1, self.tensor_parallel_group_size)
        w2_shard = (expert_parallel_group_size, self.tensor_parallel_group_size, 1)

        state_dict = {}
        state_dict[self.weight1.name] = {'shape': self.weight1.shape,
                                         'shard': w1_shard}
        state_dict[self.weight2.name] = {'shape': self.weight2.shape,
                                         'shard': w2_shard}
        return state_dict
