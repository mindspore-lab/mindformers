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
"""define experts"""

from mindspore import nn, ops

from mindformers.experimental.parallel_core.pynative.config import TransformerConfig
from mindformers.experimental.parallel_core.pynative.transformer.module import Module
from mindformers.experimental.parallel_core.pynative.transformer.mlp import ParallelMLP


class GroupedMLP(Module):
    """
    GroupedMLP is designed to execute experts in parallel
    Args:
        num_local_experts (int): how many local experts on this rank.
        config (TransformerConfig): configuration of the model.
        submodules: reserve arguments, not used now.
    """

    def __init__(self, num_local_experts: int, config: TransformerConfig, submodules=None):
        super(GroupedMLP, self).__init__()

        self.num_local_experts = num_local_experts
        self.config = config
        self.submodules = submodules

        raise NotImplementedError("GroupedMLP function not implemented.")


class SequentialMLP(Module):
    """SequentialMLP executes each expert one by one according to tokens indicises
        Args:
            num_local_experts (int): how many local experts on this rank.
            config (TransformerConfig): configuration of the model.
            submodules: reserve arguments, not used now.
    """
    def __init__(self, num_local_experts: int, config: TransformerConfig, submodules=None):
        super(SequentialMLP, self).__init__()
        if submodules is not None:
            raise NotImplementedError("For SequentialMLP, `submodules` is not supported for now.")
        self.config = config
        self.num_local_experts = num_local_experts
        self.local_experts = nn.SequentialCell()
        for _ in range(self.num_local_experts):
            expert = ParallelMLP(self.config, is_expert=True)
            self.local_experts.append(expert)
        self.cast = ops.Cast()

    def construct(self, permuted_local_hidden_states, tokens_per_expert):
        """forward process"""
        if not permuted_local_hidden_states.shape:
            return permuted_local_hidden_states
        output_local = ops.zeros_like(permuted_local_hidden_states)

        start_idx = 0
        end_idx = tokens_per_expert[0]
        # every expert rank expert_id starts from `0`
        for expert_id, expert in enumerate(self.local_experts):
            hidden_expert = permuted_local_hidden_states[start_idx:end_idx]
            if hidden_expert.shape[0] == 0:
                output = hidden_expert
            else:
                output, _ = expert(hidden_expert)
            output_local[start_idx:end_idx] = output
            if expert_id != len(self.local_experts) - 1:
                start_idx = end_idx
                end_idx += tokens_per_expert[expert_id + 1]
        return output_local
