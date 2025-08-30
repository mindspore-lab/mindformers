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
from mindspore.ops.auto_generate import Mul, AddExt, SplitWithSize, Reshape, Transpose
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindformers.parallel_core.training_graph.activation import get_activation
from mindformers.parallel_core.training_graph.device_matrix import layout
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
        self.config = config

        self.input_size = input_size if input_size is not None else config.hidden_size

        # If this is a gated linear unit we double the output width
        if is_expert and config.moe_ffn_hidden_size is not None:
            # Experts read ffn_hidden_size from config.moe_ffn_hidden_size
            map_ffn_hidden_size = config.moe_ffn_hidden_size
        else:
            # Normal MLPs read ffn_hidden_size from config.ffn_hidden_size
            map_ffn_hidden_size = config.ffn_hidden_size

        self.gated_linear_unit = config.gated_linear_unit
        if self.gated_linear_unit:
            map_ffn_hidden_size *= 2
            self.mul = Mul()

        self.mlp_has_bias = config.add_bias_linear
        self.init_method = config.init_method
        self.output_layer_init_method = config.output_layer_init_method

        if config.hidden_act == 'swiglu' and config.bias_swiglu_fusion:
            self.activation_type = 'fusedswiglu'
        else:
            self.activation_type = config.hidden_act

        self.compute_dtype = config.compute_dtype

        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.input_size,
            map_ffn_hidden_size,
            config=self.config,
            bias=self.mlp_has_bias,
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
            self.config.ffn_hidden_size,
            self.config.hidden_size,
            config=self.config,
            bias=self.mlp_has_bias,
            is_expert=is_expert,
            skip_bias_add=True,
            init_method=self.output_layer_init_method
        )

        self.split = SplitWithSize()
        self.reshape = Reshape()
        self.add = AddExt()
        self.transpose = Transpose()

        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.sharding_propagation(self.config)
        else:
            self.shard(self.config)

    def construct(self, hidden_states: Tensor, per_token_scale=None, extra_loss=0.) -> tuple[Tensor, Tensor, float]:
        """ Construct function of mlp block. """
        if per_token_scale is not None:
            # bias_activation_fusion and per_token_scale configuration is currently not supported .
            raise NotImplementedError("For MLP, 'per_token_scale' is currently not supported.")

        # [seq_len, bs, hidden_size] -> [seq_len, bs, ffn_hidden_size]
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)

        if bias_parallel is not None:
            intermediate_parallel = self.add(intermediate_parallel, bias_parallel)
        if self.gated_linear_unit:
            if self.activation_type == 'swiglu' or self.activation_type == 'fusedswiglu':
                intermediate_parallel = self.activation_func(intermediate_parallel)
            else:
                hidden_size = intermediate_parallel.shape[-1]
                x0, x1 = self.split(intermediate_parallel, (hidden_size // 2, hidden_size // 2), -1)
                act_out = self.activation_func(x0) if self.activation_func else x0
                intermediate_parallel = self.mul(act_out, x1)
        else:
            intermediate_parallel = self.activation_func(
                intermediate_parallel) if self.activation_type else intermediate_parallel
        # [seq_len, bs, hidden_size] -> [seq_len, bs, ffn_hidden_size]
        output, output_bias = self.linear_fc2(intermediate_parallel)
        return output, output_bias, extra_loss

    def shard(self, config: TransformerConfig):
        """ shard function of mlp block. """
        self.add.shard((layout("cp", "dp", "tp"), layout("tp",)))
        if self.gated_linear_unit:
            self.split.shard((layout("cp", "dp", "tp"),)).add_prim_attr("skip_redistribution", True)
            self.mul.shard((layout("cp", "dp", "tp"), layout("cp", "dp", "tp")))
            if self.activation_type == 'swiglu':
                self.activation_func.split.add_prim_attr("skip_redistribution", True)
                self.activation_func.split.shard((layout("cp", "dp", "tp"),))
            if self.activation_type == 'fusedswiglu':
                self.activation_func.swiglu.add_prim_attr("skip_redistribution", True)
                self.activation_func.swiglu.shard((layout("cp", "dp", "tp"),))

    def sharding_propagation(self, config: TransformerConfig):
        pass


class MLPInterleaved(MLP):
    r"""
    Parallel feedforward block implementation, with interleaved weight layout.

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

    def construct(self, hidden_states: Tensor, per_token_scale=None, extra_loss=0.) -> tuple[Tensor, Tensor, float]:
        """ Construct function of mlp block. """
        if per_token_scale is not None:
            # bias_activation_fusion and per_token_scale configuration is currently not supported .
            raise NotImplementedError("For MLP, 'per_token_scale' is currently not supported.")

        # [seq_len, bs, hidden_size] -> [seq_len, bs, ffn_hidden_size]
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)

        if bias_parallel is not None:
            intermediate_parallel = self.add(intermediate_parallel, bias_parallel)
        if self.gated_linear_unit:
            seq, bs, ffn_hidden_size = intermediate_parallel.shape
            intermediate_parallel = self.reshape(intermediate_parallel, (seq, bs, ffn_hidden_size // 2, 2))
            if self.activation_type == 'fusedswiglu':
                intermediate_parallel = self.transpose(intermediate_parallel, (0, 1, 3, 2))
                intermediate_parallel = self.activation_func(intermediate_parallel, -2)
                intermediate_parallel = self.reshape(intermediate_parallel, (seq, bs, ffn_hidden_size // 2))
            elif self.activation_type == 'swiglu':
                intermediate_parallel = self.activation_func(intermediate_parallel)
                intermediate_parallel = self.reshape(intermediate_parallel, (seq, bs, ffn_hidden_size // 2))
            else:
                x0, x1 = self.split(intermediate_parallel, (1, 1), -1)
                x0 = self.reshape(x0, (seq, bs, ffn_hidden_size // 2))
                x1 = self.reshape(x1, (seq, bs, ffn_hidden_size // 2))
                act_out = self.activation_func(x0) if self.activation_func else x0
                intermediate_parallel = self.mul(act_out, x1)
        else:
            intermediate_parallel = self.activation_func(
                intermediate_parallel) if self.activation_type else intermediate_parallel
        # [seq_len, bs, hidden_size] -> [seq_len, bs, ffn_hidden_size]
        output, output_bias = self.linear_fc2(intermediate_parallel)
        return output, output_bias, extra_loss

    def shard(self, config: TransformerConfig):
        """ shard function of mlp block. """
        self.add.shard((layout("cp", "dp", "tp"), layout("tp",)))
        if self.gated_linear_unit:
            self.mul.shard((layout("cp", "dp", "tp"), layout("cp", "dp", "tp")))
            self.split.shard((layout("cp", "dp", "tp", "None"),))
            if self.activation_type == 'fusedswiglu':
                self.transpose.shard((layout("cp", "dp", "tp", "None"),))
                self.activation_func.swiglu.shard((layout("cp", "dp", "None", "tp"),))
            if self.activation_type == 'swiglu':
                self.activation_func.silu.shard((layout("cp", "dp", "tp", "None"),))
                self.activation_func.split.shard((layout("cp", "dp", "tp", "None"),))
                self.activation_func.mul.shard((layout("cp", "dp", "tp", "None"), layout("cp", "dp", "tp", "None")))

    def sharding_propagation(self, config: TransformerConfig):
        pass
