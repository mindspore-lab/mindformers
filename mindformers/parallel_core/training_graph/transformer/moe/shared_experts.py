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
"""Transformer SharedExpertMLP"""
__all__ = [
    "SharedExpertMLP"
]
from copy import deepcopy
from mindspore import Tensor
from mindspore.nn.layer import Dense
from mindspore.ops.auto_generate import Cast, Mul, Sigmoid
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindspore.context import ParallelMode

from mindformers.parallel_core.training_graph.transformer.mlp import MLP, MLPSubmodules
from mindformers.parallel_core.transformer_config import TransformerConfig


class SharedExpertMLP(MLP):
    r"""
    Implementation of a shared expert feedforward block that inherits from MLP.

    This module extends the standard MLP to support shared expert logic, typically used in MoE settings.

    Args:
        config (TransformerConfig): Configuration for the transformer model.
        submodules (MLPSubmodules): The submodules used to construct the MLP, such as activation and linear layers.
        gate (bool): Whether gating mechanism is enabled for expert routing.

    Inputs:
        - **hidden_states** (Tensor) - Input tensor of shape :math:`(S, B, H)`, where
          :math:`S` is sequence length, :math:`B` is batch size, and :math:`H` is hidden size.

    Outputs:
        - **output** (Tensor) - Output tensor of shape :math:`(S, B, H)`.
        - **output_bias** (Tensor) - Bias tensor of shape :math:`(S, B, H)` (if applicable).

    Supported Platforms:
        ``Ascend``
    """

    def __init__(self, config: TransformerConfig, submodules: MLPSubmodules, gate: bool):
        config = deepcopy(config)
        config.ffn_hidden_size = config.moe_shared_expert_intermediate_size
        super().__init__(config, submodules)

        self.cast = Cast()
        self.use_seq_parallel = config.sequence_parallel
        self.router_dense_type = config.moe_router_dtype
        self.use_shared_expert_gate = gate
        if self.use_shared_expert_gate:
            self.shared_experts_gate = Dense(in_channels=config.hidden_size,
                                             out_channels=1,
                                             has_bias=False,
                                             dtype=self.router_dense_type)
            self.sigmoid = Sigmoid()
            self.mul_shared_gate = Mul()
            if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
                self.expert_sharding_propagation(self.parallel_config)
            else:
                self.expert_gate_shard(config)

    def construct(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        """ Construct function of shared_expert_mlp block. """
        if self.use_seq_parallel:
            shared_x = self.reshape(hidden_states, (-1, self.dp, hidden_states.shape[-1]))
            shared_experts_output, output_bias = super().construct(shared_x)
            shared_experts_output = self.reshape(shared_experts_output,
                                                 (hidden_states.shape[0], -1, hidden_states.shape[-1]))
        else:
            shared_experts_output, output_bias = super().construct(hidden_states)
        if self.use_shared_expert_gate:
            gate = self.sigmoid(self.shared_experts_gate(self.cast(hidden_states, self.router_dense_type)))
            shared_experts_output = self.mul_shared_gate(shared_experts_output, self.cast(gate, self.compute_dtype))
        return shared_experts_output, output_bias

    def expert_gate_shard(self, config: TransformerConfig):
        """ shard function of shared_expert_mlp block. """
        super().shard(config)
        dp = config.data_parallel_size if config.data_parallel_size is not None else 1
        if self.use_shared_expert_gate:
            self.sigmoid.shard(((1, dp, 1),))
            self.mul_shared_gate.shard(((1, dp, 1), (1, dp, 1)))
            self.shared_experts_gate.matmul.shard(((dp, 1), (1, 1)))

    def expert_sharding_propagation(self, config: TransformerConfig):
        super().sharding_propagation(config)
