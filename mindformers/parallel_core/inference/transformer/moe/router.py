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
"""Expert Router."""
import numpy as np

from mindspore import Tensor, nn, Parameter, ops, mint
import mindspore.common.dtype as mstype
from mindspore.common.initializer import initializer
from mindspore.ops.auto_generate import FusedAddTopKDiv

from mindformers.models.utils import convert_mstype
from mindformers.parallel_core.transformer_config import TransformerConfig


__all__ = [
    'Router',
    'TopKRouter',
]


class Router(nn.Cell):
    """Base Router class"""

    def __init__(self, config: TransformerConfig) -> None:
        """
        Initialize the Router module.

        Args:
            config (TransformerConfig): Configuration object for the Transformer model.
        """
        super(Router, self).__init__()
        self.config = config
        self.hidden = config.hidden_size
        self.router_dense_type = convert_mstype(config.moe_router_dtype)
        self.num_experts = self.config.num_moe_experts
        self.weight = nn.Dense(in_channels=self.hidden,
                               out_channels=self.num_experts,
                               has_bias=False,
                               dtype=self.router_dense_type)
        self.cast = ops.Cast()

    def gating(self, input_tensor: Tensor):
        """Forward pass of the router gate.

        Args:
            input_tensor (Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits tensor.
        """
        logits = self.weight(self.cast(input_tensor, self.router_dense_type))
        return logits

    def routing(self, logits: Tensor):
        """Routing function.

        Args:
            logits (Tensor): Logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing token assignment
            probabilities and mapping.
        """
        raise NotImplementedError("For Router, routing function not implemented.")

    def construct(self, input_tensor: Tensor):
        """
        Forward pass of the router.

        Args:
            input_tensor (Tensor): Input tensor.
        """
        raise NotImplementedError("For Router, construct function not implemented.")


class TopKRouter(Router):
    """Route each token to the top-k experts."""

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize the zero token dropping router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
        """
        super().__init__(config=config)
        self.config = config
        self.n_group = config.moe_router_num_groups
        self.topk_group = config.moe_router_group_topk
        self.group_topk_inner = 2
        self.num_experts_chosen = config.moe_router_topk

        self.idx_arr = Tensor(np.arange(1024, dtype=np.int32))

        self.group_topk = GroupTopkCell()

        self.moe_router_enable_expert_bias = config.moe_router_enable_expert_bias

        self.expert_bias = Parameter(initializer('zeros', (self.num_experts), mstype.float32))

        self.fused_add_topk_div = FusedAddTopKDiv()

    def routing(self, logits: Tensor):
        """Top-k routing function

        Args:
            logits (Tensor): Logits tensor after gating.

        Returns:
            probs (Tensor): The probabilities of token to experts assignment.
            routing_map (Tensor): The mapping of token to experts assignment,
                with shape [num_tokens, num_experts].
            origin_probs (Tensor): The probabilities of token to experts assignment before add bias.
        """
        input_dtype = logits.dtype
        gating_logits = self.gating(self.cast(logits, self.router_dense_type))
        gating_logits = self.cast(gating_logits, mstype.float32)
        if self.config.moe_router_group_topk:
            expert_weight, expert_index = \
                self.fused_add_topk_div(
                    gating_logits,
                    self.expert_bias,
                    self.num_experts_chosen,
                    self.topk_group,
                    self.group_topk_inner,
                    self.num_experts_chosen,
                    0,
                    True,
                    self.config.moe_router_topk_scaling_factor)
        else:
            score = mint.sigmoid(gating_logits)
            score = score + self.expert_bias
            expert_weight, expert_index = mint.topk(score, self.config.moe_router_topk, dim=-1)
            expert_index = self.cast(expert_index, mstype.int32)
            expert_weight = mint.div(expert_weight, mint.sum(expert_weight, -1, True))
        expert_weight = expert_weight.astype(input_dtype)
        return expert_weight, expert_index

    def construct(self, input_tensor: Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor with shape [num_tokens, hidden_size].
        """
        expert_weight, expert_index = self.routing(input_tensor)

        return expert_weight, expert_index


class GroupTopkCell(nn.Cell):
    r"""
        GroupTopkCell.

        Inputs:
            - **input_tensor** (Tensor) - should be `[batch, seq_length, hidden_size].

        Outputs:
            - **output_tensor** (Tensor) - should be `[batch, seq_length, hidden_size].
    """
    def __init__(self):
        super().__init__()
        self.group_topk = ops.GroupTopk()

    def construct(self, token, idx_arr, group_num, k, k_inner):
        self.group_topk(token, idx_arr, group_num, k, k_inner)
        return token
