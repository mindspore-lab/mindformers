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
from mindspore import nn, Tensor
from mindspore.context import ParallelMode
from mindspore.ops.auto_generate import Mul, AddExt, SplitWithSize, Reshape
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindformers.parallel_core.training_graph.activation import get_activation
from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.transformer_config import TransformerConfig


@dataclass
class MLPSubmodules:
    linear_fc1: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None


class MLP(nn.Cell):
    r"""
    Parallel feedforward block implementation.

    Args:
        config (TransformerConfig): Configuration for the transformer model.
        submodules (MLPSubmodules): The submodules used to construct the MLP, such as activation and linear layers.
        is_expert (bool, optional): Whether this block is used as an expert in MoE. Default: False.
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
            is_expert: bool = False,
            input_size: int = None,
    ):
        super(MLP, self).__init__()
        self.hidden_size = input_size if input_size is not None else config.hidden_size
        self.ffn_hidden_size = config.ffn_hidden_size
        self.mlp_has_bias = config.add_bias_linear
        self.gated_linear_unit = config.gated_linear_unit
        self.init_method = config.init_method
        self.output_layer_init_method = config.output_layer_init_method
        self.activation_type = config.hidden_act
        self.compute_dtype = config.compute_dtype
        self.config = config
        cp = 1 if config is None else config.context_parallel_size
        self.compute_2d = (config.sequence_parallel and cp == 1)
        self.mapping_ffn_hidden_size = self.ffn_hidden_size
        self.split = SplitWithSize()
        self.reshape = Reshape()

        if self.gated_linear_unit:
            self.mapping_ffn_hidden_size *= 2
            self.mul = Mul()

        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.hidden_size,
            self.mapping_ffn_hidden_size,
            self.config,
            bias=self.mlp_has_bias,
            compute_dtype=self.compute_dtype,
            is_expert=is_expert,
            skip_bias_add=True,
            init_method=self.init_method
        )

        if self.activation_type is not None:
            self.activation_func = get_activation(self.activation_type, config=self.config)
        else:
            self.activation_func = None

        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.ffn_hidden_size,
            self.hidden_size,
            self.config,
            bias=self.mlp_has_bias,
            compute_dtype=self.compute_dtype,
            is_expert=is_expert,
            skip_bias_add=True,
            init_method=self.output_layer_init_method
        )

        self.add = AddExt()

        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.sharding_propagation(self.config)
        else:
            self.shard(self.config)

    def construct(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        """ Construct function of mlp block. """
        # [seq_len, bs, hidden_size] -> [seq_len, bs, ffn_hidden_size]
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)

        if bias_parallel is not None:
            intermediate_parallel = self.add(intermediate_parallel, bias_parallel)
        if self.gated_linear_unit:
            if self.activation_type == 'swiglu':
                intermediate_parallel = self.activation_func(intermediate_parallel)
            else:
                _, _, hidden_size = intermediate_parallel.shape
                gate, hidden = self.split(intermediate_parallel, (hidden_size // 2, hidden_size // 2), -1)
                gate = self.activation_func(gate) if self.activation_func else gate
                intermediate_parallel = self.mul(hidden, gate)
        else:
            intermediate_parallel = self.activation_func(
                intermediate_parallel) if self.activation_type else intermediate_parallel
        # [seq_len, bs, hidden_size] -> [seq_len, bs, ffn_hidden_size]
        output, output_bias = self.linear_fc2(intermediate_parallel)
        return output, output_bias

    def shard(self, config: TransformerConfig):
        """ shard function of mlp block. """
        if self.gated_linear_unit:
            dp = config.data_parallel_size if config.data_parallel_size is not None else 1
            cp = config.context_parallel_size if config.context_parallel_size is not None else 1
            tp = config.tensor_model_parallel_size if config.tensor_model_parallel_size is not None else 1
            if self.compute_2d:
                mul_in_strategy = ((dp, tp), (dp, tp))
                self.mul.shard(in_strategy=mul_in_strategy)
                self.add.shard(((dp, tp), (tp,)))
            else:
                mul_in_strategy = ((cp, dp, tp), (cp, dp, tp))
                self.mul.shard(in_strategy=mul_in_strategy)
                self.add.shard(((cp, dp, tp), (tp,)))

            if config.sequence_parallel and cp == 1:
                self.linear_fc2.matmul.shard(in_strategy=((dp, tp), (1, tp)), out_strategy=((dp * tp, 1),))

            if self.gated_linear_unit and self.activation_type != 'swiglu':
                if self.compute_2d:
                    self.split.shard(((dp, 1),))
                else:
                    self.split.shard(((1, dp, tp),)).add_prim_attr("skip_redistribution", True)

    def sharding_propagation(self, config: TransformerConfig):
        pass
