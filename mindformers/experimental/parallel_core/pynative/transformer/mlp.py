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
"""MLP Module"""
from mindspore import mint

from mindformers.experimental.parallel_core.pynative.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from mindformers.experimental.parallel_core.pynative.tensor_parallel.lora_layers import (
    ColumnParallelLoRA,
    RowParallelLoRA,
)

from .activation import get_act_func, get_act_func_gated_version
from .module import Module

__all__ = [
    "ParallelMLP",
]


class ParallelMLP(Module):
    r"""
    Implementation of parallel feedforward block.

    Args:
        config (dict): Configuration.
        is_expert (book): This block is an expert block. Default: False.

    Inputs:
        - **hidden_states** (Tensor) - Tensor of shape :math:`(B, S, H)`.

    Outputs:
        - **output** (Tensor) - Output tensor of shape :math:`(B, S, H)`.
        - **output_bias** (Parameter) - Output projection bias weight when `projection.skip_bias_add=True`.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(self, config, is_expert=False):
        super(ParallelMLP, self).__init__(config)
        self.config = config
        self.has_bias = self.config.mlp_has_bias
        self.hidden_size = self.config.hidden_size
        self.ffn_hidden_size = self.config.ffn_hidden_size
        self.mlp_has_gate = self.config.mlp_has_gate
        self.use_lora = config.lora_config.use_lora
        self.act_type = self.config.hidden_act
        self.is_expert = is_expert

        self._init_mapping()
        self.bias_gelu_fusion = False
        self.act_func = get_act_func(self.act_type)

        self._init_projection()

    def _init_mapping(self):
        """ initialize mapping cell """
        mapping_output_size = self.ffn_hidden_size
        if self.config.mlp_has_gate:
            gated_act_type = get_act_func_gated_version(self.act_type)
            if gated_act_type is not None:
                self.mapping_gate_fusion = True
                self.act_type = gated_act_type
                mapping_output_size *= 2
            else:
                self.mapping_gate_fusion = False
                self.gating = ColumnParallelLinear(
                    self.hidden_size,
                    self.ffn_hidden_size,
                    config=self.config,
                    init_method=self.config.init_method,
                    bias=self.has_bias,
                    gather_output=False,
                    is_expert=self.is_expert,
                    bias_init=self.config.bias_init,
                )
        self.mapping = ColumnParallelLinear(
            self.hidden_size,
            mapping_output_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.has_bias,
            gather_output=False,
            is_expert=self.is_expert,
            param_init_dtype=self.config.param_init_dtype,
            compute_dtype=self.config.compute_dtype,
            bias_init=self.config.bias_init,
        )
        if self.use_lora:
            mapping_lora = self._get_cell_lora_config(self.config, 'mapping')
            if mapping_lora is not None:
                self.mapping = ColumnParallelLoRA(
                    self.hidden_size,
                    mapping_output_size,
                    config=self.config,
                    init_method=self.config.init_method,
                    bias=self.has_bias,
                    gather_output=False,
                    is_expert=self.is_expert,
                    param_init_dtype=self.config.param_init_dtype,
                    compute_dtype=self.config.compute_dtype,
                    bias_init=self.config.bias_init,
                    lora_rank=mapping_lora['rank'],
                    lora_alpha=mapping_lora['alpha'],
                    lora_dropout=mapping_lora['dropout'],
                )
            gating_lora = self._get_cell_lora_config(self.config, 'gating')
            if self.config.mlp_has_gate and not self.mapping_gate_fusion and gating_lora is not None:
                self.gating = ColumnParallelLoRA(
                    self.hidden_size,
                    self.ffn_hidden_size,
                    config=self.config,
                    init_method=self.config.init_method,
                    bias=self.has_bias,
                    gather_output=False,
                    is_expert=self.is_expert,
                    bias_init=self.config.bias_init,
                    lora_rank=gating_lora['rank'],
                    lora_alpha=gating_lora['alpha'],
                    lora_dropout=gating_lora['dropout'],
                )

    def _init_projection(self):
        """ initialize projection cell """
        if self.config.out_hidden_size is None:
            out_hidden_size = self.hidden_size
        else:
            out_hidden_size = self.config.out_hidden_size
        self.projection = RowParallelLinear(
            self.ffn_hidden_size,
            out_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.has_bias,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=self.is_expert,
            param_init_dtype=self.config.param_init_dtype,
            compute_dtype=self.config.compute_dtype,
            bias_init=self.config.bias_init,
        )
        if self.use_lora:
            projection_lora = self._get_cell_lora_config(self.config, 'projection')
            if projection_lora is not None:
                self.projection = RowParallelLoRA(
                    self.ffn_hidden_size,
                    out_hidden_size,
                    config=self.config,
                    init_method=self.config.init_method,
                    bias=self.has_bias,
                    input_is_parallel=True,
                    skip_bias_add=False,
                    is_expert=self.is_expert,
                    param_init_dtype=self.config.param_init_dtype,
                    compute_dtype=self.config.compute_dtype,
                    bias_init=self.config.bias_init,
                    lora_rank=projection_lora['rank'],
                    lora_alpha=projection_lora['alpha'],
                    lora_dropout=projection_lora['dropout'],
                )

    def construct(self, hidden_states):
        """ Construct function of mlp block. """
        # [B, S, H] -> [B, S, ffn_H]
        if self.config.mlp_has_gate and not self.mapping_gate_fusion:
            gate, _ = self.gating(hidden_states)
            gate = self.act_func(gate)
            intermediate_parallel, _ = self.mapping(hidden_states)
            intermediate_parallel = mint.mul(intermediate_parallel, gate)
        else:
            intermediate_parallel, _ = self.mapping(hidden_states)
            intermediate_parallel = self.act_func(intermediate_parallel)

        # [B, S, ffn_H] -> [B, S, H]
        output, output_bias = self.projection(intermediate_parallel)
        return output, output_bias
