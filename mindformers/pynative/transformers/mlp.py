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
"""Transformer MLP"""
__all__ = [
    "MLPSubmodules",
    "MLP"
]

from dataclasses import dataclass
from typing import Union

from mindspore import nn, Tensor, mint
from mindformers.pynative.layers.activation import get_activation
from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.transformer_config import TransformerConfig


@dataclass
class MLPSubmodules:
    linear_fc1: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None


class MLP(nn.Cell):
    """
    Parallel feed forward block implementation, with interleaved weight layout.

    Args:
        config (TransformerConfig): Configuration for the transformer model.
        submodules (MLPSubmodules): The submodules used to construct the MLP, such as activation and linear layers.
        input_size (int, optional): Input hidden size. If None, will use config.hidden_size. Default: None.

    Inputs:
        - **hidden_states** (Tensor) - Input tensor with shape :math:`(S, B, H)`, where
          :math:`S` is the sequence length, :math:`B` is the batch size, and :math:`H` is the hidden size.

    Outputs:
        - **output** (Tensor) - Output tensor of shape :math:`(S, B, H)`.
        - **output_bias** (Tensor) - Bias output tensor of shape :math:`(S, B, H)`.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(
            self,
            config: TransformerConfig,
            submodules: MLPSubmodules,
            input_size: int = None,
    ):
        super().__init__()
        self.config = config

        self.input_size = input_size if input_size is not None else config.hidden_size

        # Normal MLPs read ffn_hidden_size from config.ffn_hidden_size
        map_ffn_hidden_size = config.ffn_hidden_size

        self.gated_linear_unit = config.gated_linear_unit
        if self.gated_linear_unit:
            map_ffn_hidden_size *= 2
            self.mul = mint.mul

        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.input_size,
            map_ffn_hidden_size,
            compute_dtype=self.config.compute_dtype,
            params_dtype=self.config.params_dtype,
            init_method=self.config.init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=True
        )

        # Handle activation function
        self.activation_type = self.config.hidden_act
        self.activation_func = get_activation(self.activation_type)

        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.config.ffn_hidden_size,
            self.config.hidden_size,
            compute_dtype=self.config.compute_dtype,
            params_dtype=self.config.params_dtype,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=True
        )

        self.split = mint.split
        self.reshape = mint.reshape
        self.add = mint.add
        self.transpose = mint.transpose

    def construct(self, hidden_states: Tensor, extra_loss=0.) -> tuple[Tensor, Tensor, float]:
        """ Construct function of mlp block. """
        # [seq_len, bs, hidden_size] -> [seq_len, bs, ffn_hidden_size]
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)

        if bias_parallel is not None:
            intermediate_parallel = self.add(intermediate_parallel, bias_parallel)
        if self.gated_linear_unit:
            seq, bs, ffn_hidden_size = intermediate_parallel.shape
            intermediate_parallel = self.reshape(intermediate_parallel, (seq, bs, ffn_hidden_size // 2, 2))
            if self.activation_type == 'fusedswiglu':
                intermediate_parallel = self.transpose(intermediate_parallel, 2, 3)
                intermediate_parallel = self.activation_func(intermediate_parallel, -2)
                intermediate_parallel = self.reshape(intermediate_parallel, (seq, bs, ffn_hidden_size // 2))
            else:
                x0, x1 = self.split(intermediate_parallel, (1, 1), -1)
                x0 = self.reshape(x0, (seq, bs, ffn_hidden_size // 2))
                x1 = self.reshape(x1, (seq, bs, ffn_hidden_size // 2))
                act_out = self.activation_func(x0)
                intermediate_parallel = self.mul(act_out, x1)
        else:
            intermediate_parallel = self.activation_func(intermediate_parallel)
        # [seq_len, bs, hidden_size] -> [seq_len, bs, ffn_hidden_size]
        output, output_bias = self.linear_fc2(intermediate_parallel)
        return output, output_bias, extra_loss
