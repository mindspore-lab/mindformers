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
from typing import Union, Optional

from mindspore import nn, mint

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.inference.transformer.activation import get_act_func
from mindformers.parallel_core.inference.utils import divide
from mindformers.parallel_core.inference.parallel_state import ProcessGroup, default_pgs


@dataclass
class MLPSubmodules:
    """
    The MLPSubmodules class defines two submodules for a Multi-Layer Perceptron (MLP):

    Args:
        linear_fc1 (Union[ModuleSpec, type], optional): The module definition for the first fully connected layer.
            Defaults to None.
        linear_fc2 (Union[ModuleSpec, type], optional): The module definition for the second fully connected layer.
            Defaults to None.
    """

    linear_fc1: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None


class MLP(nn.Cell):
    r"""
    Implementation of parallel feedforward block.

    Args:
        config (dict): Configuration for the transformer model.
        submodules (MLPSubmodules): The submodules used to construct the MLP, such as activation and linear layers.
        is_expert (bool, optional): Whether this block is used as an expert in MoE. Default: False.
        input_size (int, optional): Input hidden size. If None, will use config.hidden_size. Default: None.
        delay_allreduce (bool): Whether to delay allreduce during linear_fc2 forward function. Default: False
        tp_group (ProcessGroup): The process_group that current layer used. Default: default_pgs.


    Inputs:
        - **hidden_states** (Tensor) - Input tensor

    Outputs:
        - **output** (Tensor) - Output tensor

    Supported Platforms:
        ``Ascend``
    """

    def __init__(
            self,
            config: TransformerConfig,
            submodules: MLPSubmodules,
            is_expert: bool = False,
            input_size: Optional[int] = None,
            delay_allreduce: bool = False,
            tp_group: Optional[ProcessGroup] = default_pgs,
    ):
        super().__init__(config)

        self.config: TransformerConfig = config

        self.input_size = input_size if input_size is not None else self.config.hidden_size
        self.tp_group = tp_group
        self.tp_group_size = tp_group.size if tp_group is not None else 1

        if is_expert and self.config.moe_ffn_hidden_size is not None:
            ffn_hidden_size = self.config.moe_ffn_hidden_size
        else:
            ffn_hidden_size = self.config.ffn_hidden_size
        self.ffn_hidden_size_per_partition = divide(ffn_hidden_size, self.tp_group_size)

        self.activation_type = self.config.hidden_act

        fc1_parallel_kwargs = {}
        fc2_parallel_kwargs = {}
        if self.tp_group is not None:
            # linear fc1 parallel kwargs
            fc1_parallel_kwargs.update({
                "gather_output": False,
                "tp_group": self.tp_group
            })
            # linear fc2 parallel kwargs
            fc2_parallel_kwargs.update({
                "delay_allreduce": delay_allreduce,
                "tp_group": self.tp_group
            })

        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.input_size,
            ffn_hidden_size,
            config=self.config,
            bias=self.config.mlp_linear_fc1_bias,
            is_expert=is_expert,
            transpose_b=True,
            compute_dtype=self.config.compute_dtype,
            **fc1_parallel_kwargs,
        )

        if self.activation_type is not None:
            self.activation_func = get_act_func(self.activation_type)
        else:
            self.activation_func = None

        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.config.ffn_hidden_size,
            self.config.hidden_size,
            config=self.config,
            bias=self.config.add_bias_linear,
            is_expert=is_expert,
            transpose_b=True,
            compute_dtype=self.config.compute_dtype,
            **fc2_parallel_kwargs,
        )

    def construct(self, hidden_states):
        """ Construct function of mlp block. """
        # [T, H] -> [T, ffn_H]
        intermediate_parallel = self.linear_fc1(hidden_states)

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
        output = self.linear_fc2(intermediate_parallel)
        return output
