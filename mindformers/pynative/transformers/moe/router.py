# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mixture of Experts (MoE) modules for pynative mode."""
from typing import Tuple, Optional

from mindspore import nn, Tensor, mint, ops
from mindspore.common.parameter import Parameter

from mindspore.common import dtype as mstype

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.utils.init_method import init_method_normal


class TopKRouter(nn.Cell):
    """This class implements token-choice routing. In token-choice top-K routing, each token is
    routed to top K experts based on the router scores.

    Optionally supports node-limited (group-limited) routing where experts are divided into groups
    (e.g., by node), and only num_limited_groups groups are considered before selecting top_k experts.
    This reduces cross-node communication in distributed settings.

    Args:
        config (TransformerConfig): Transformer configuration object containing:
            - hidden_size (int): Dimension of input tokens.
            - num_moe_experts (int): Number of experts in each moe layer.
            - moe_router_num_groups (int | None): Number of expert groups for node-limited routing. If None, standard
              top-k routing is used. Must be a divisor of num_experts.
            - moe_router_group_topk (int | None): Number of groups to select in node-limited routing. Required when
              moe_router_num_groups is set.
            - moe_router_topk (int): Number of experts each token will be routed to in token-choice routing.
            - moe_router_score_function (Literal["softmax", "sigmoid"]): Whether to use sigmoid or
              softmax for router scores.
            - norm_topk_prob (bool): Whether to normalize the routing scores when using sigmoid.
            - moe_router_topk_scaling_factor (float): Scaling factor applied to the routing scores.
            - moe_router_force_expert_balance (bool): Whether to force load balance via round-robin
              routing. Default: False.
    """

    def __init__(
        self,
        config: TransformerConfig,
    ):
        super().__init__()
        # Extract parameters from config
        dim = config.hidden_size
        num_experts = config.num_moe_experts
        num_expert_groups = config.moe_router_num_groups
        num_limited_groups = config.moe_router_group_topk
        top_k = config.moe_router_topk
        score_func = config.moe_router_score_function
        route_norm = config.norm_topk_prob
        route_scale = (
            config.moe_router_topk_scaling_factor
            if config.moe_router_topk_scaling_factor is not None
            else 1.0
        )
        self.weight = Parameter(init_method_normal(0.02)((num_experts, dim)), name='weight')
        self.num_experts = num_experts
        self.num_expert_groups = num_expert_groups
        self.num_limited_groups = num_limited_groups
        self.top_k = top_k
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale
        self._debug_force_load_balance = config.moe_router_force_expert_balance

        # Initialize operators in __init__
        self.linear = mint.nn.functional.linear
        self.sigmoid = mint.nn.functional.sigmoid
        self.softmax = mint.nn.functional.softmax
        self.cast = ops.cast
        self.arange = mint.arange
        self.gather = mint.gather
        self.reshape = mint.reshape
        self.topk = mint.topk
        self.sum = mint.sum
        self.ones_like = mint.ones_like
        self.div = mint.div
        self.mul = mint.mul
        self.histc = mint.histc

    def _debug_force_load_balance_routing(
        self, scores: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Balanced round-robin expert assignment.
        Returns (selected_experts_indices [N, K] LongTensor, top_scores [N, K] FloatTensor).
        """
        n_tokens = scores.shape[0]
        # Round-robin indices with exact balance
        selected_experts_indices = (
            self.reshape(
                self.arange(n_tokens * self.top_k, dtype=mstype.int64),
                (n_tokens, self.top_k)
            )
            % self.num_experts
        )
        top_scores = self.gather(scores, dim=1, index=selected_experts_indices)  # [N,K]
        return selected_experts_indices, top_scores

    def _get_node_limited_routing_scores(
        self,
        scores_for_choice: Tensor,
    ) -> Tensor:
        """Select num_limited_groups groups based on group scores,
        and set expert scores in non-selected groups as -inf

        Args:
            scores_for_choice: Router scores with expert_bias (if any), shape (bs*slen, num_experts)

        Returns:
            scores_for_choice: shape (bs*slen, num_experts)
        """
        if self.num_limited_groups is None:
            raise ValueError(
                "num_limited_groups must be set when num_expert_groups is set"
            )
        if self.num_expert_groups is None:
            raise ValueError(
                "num_expert_groups must be set when using node-limited routing"
            )
        if self.num_experts % self.num_expert_groups != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by "
                f"num_expert_groups ({self.num_expert_groups})"
            )
        experts_per_group = self.num_experts // self.num_expert_groups
        if experts_per_group < 2:
            raise ValueError(f"experts_per_group ({experts_per_group}) must be >= 2")
        scores_grouped = self.reshape(
            scores_for_choice,
            (-1, self.num_expert_groups, experts_per_group)
        )
        top2_scores_in_group, _ = self.topk(scores_grouped, 2, dim=-1)
        group_scores = self.sum(top2_scores_in_group, dim=-1)
        _, group_idx = self.topk(
            group_scores, k=self.num_limited_groups, dim=-1, sorted=False
        )
        group_mask = self.ones_like(group_scores, dtype=mstype.bool)
        group_mask.scatter_(1, group_idx, False)  # False = selected groups (keep)
        # Mask out experts from non-selected groups
        scores_for_choice = scores_grouped.masked_fill(
            group_mask.unsqueeze(-1), float("-inf")
        )
        scores_for_choice = self.reshape(
            scores_grouped, (-1, self.num_experts)
        )

        return scores_for_choice

    def construct(
        self, x: Tensor, expert_bias: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x (Tensor): Input tensor with shape ``(bs*slen, dim)``.
            expert_bias (Tensor | None, optional): Optional bias tensor for experts with shape ``(num_experts,)``.
                Used for load balancing. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - top_scores (Tensor):
                    Routing scores for selected experts with shape ``(bs*slen, top_k)``.
                - selected_experts_indices (Tensor):
                    Expert indices selected for each token with shape ``(bs*slen, top_k)``.
                - num_tokens_per_expert (Tensor):
                    Number of tokens assigned to each expert with shape ``(num_experts,)``.
        """
        # scores shape (bs*slen, num_experts)
        scores = self.linear(
            self.cast(x, mstype.float32), self.weight
        )

        # By default, sigmoid or softmax is performed in float32 to avoid loss explosion
        if self.score_func == "sigmoid":
            scores = self.sigmoid(self.cast(scores, mstype.float32))
        elif self.score_func == "softmax":
            scores = self.softmax(self.cast(scores, mstype.float32), dim=1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        scores_for_choice = scores if expert_bias is None else scores + expert_bias
        # Apply node-limited routing if configured
        if self.num_expert_groups is not None:
            scores_for_choice = self._get_node_limited_routing_scores(scores_for_choice)
        _, selected_experts_indices = self.topk(
            scores_for_choice, k=self.top_k, dim=-1, sorted=False
        )
        selected_experts_indices = self.cast(selected_experts_indices, mstype.int64)

        # top scores shape (bs*slen, top_k)
        # NOTE: The expert_bias is only used for routing. The gating value
        #       top_scores is still derived from the original scores.
        top_scores = self.gather(scores, dim=1, index=selected_experts_indices)

        # debug override: balanced round-robin routing
        if self._debug_force_load_balance:
            (
                selected_experts_indices,
                top_scores,
            ) = self._debug_force_load_balance_routing(scores)

        if self.route_norm:
            denominator = self.sum(top_scores, dim=-1, keepdim=True) + 1e-20
            top_scores = self.div(top_scores, denominator)

        top_scores = self.mul(top_scores, self.route_scale)
        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = self.histc(
            selected_experts_indices,
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )
        num_tokens_per_expert = self.cast(num_tokens_per_expert, mstype.float32)

        return top_scores, selected_experts_indices, num_tokens_per_expert
