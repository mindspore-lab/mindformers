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
"""Shared Experts Module."""
__all__ = ['SharedExpertMLP']

from copy import deepcopy
from typing import Optional

from mindspore import mint

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.inference.transformer.mlp import MLP, MLPSubmodules
from mindformers.parallel_core.inference.transformer.activation import get_act_func
from mindformers.parallel_core.inference.tensor_parallel.layers import ReplicatedLinear
from mindformers.parallel_core.process_group_config import ModelCommProcessGroups, default_model_comm_pgs


class SharedExpertMLP(MLP):
    """
    MLP layer for Shared Experts.
    """

    def __init__(
            self,
            config: TransformerConfig,
            submodules: MLPSubmodules,
            gate: bool,
            model_comm_pgs: Optional[ModelCommProcessGroups] = default_model_comm_pgs,
    ):
        config = deepcopy(config)
        if config.add_bias_linear:
            raise ValueError("bias is not supported in the shared experts")

        self.shared_expert_num = config.shared_expert_num
        if self.shared_expert_num == 0:
            raise ValueError("For SharedExpertMLP`, shared_expert_num` must be greater than 0.")

        config.ffn_hidden_size = config.moe_shared_expert_intermediate_size

        # In AlltoAll communication, shared expert will not be split in parallel
        super().__init__(
            config=config,
            submodules=submodules,
            is_expert=True,
            delay_allreduce=True,
            tp_group=model_comm_pgs.globals if not config.use_alltoall else None)

        self.use_shared_expert_gate = gate
        if self.use_shared_expert_gate:
            self.gate = ReplicatedLinear(
                input_size=1,
                output_size=self.config.hidden_size,
                config=self.config,
                bias=False,
                transpose_b=False,
                compute_dtype=self.config.compute_dtype
            )
            self.activation_func = get_act_func("silu")
        else:
            self.gate = None

    def construct(self, hidden_states):
        """ Construct function of SharedExpertMLP block. """
        # [T, H] -> [T, ffn_H]
        output = super().construct(hidden_states)

        if self.use_shared_expert_gate:
            logits = self.gate(hidden_states)
            gate_score = self.activation_func(logits)
            output = mint.matmul(output, gate_score)

        return output
