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
from typing import Optional
import numpy as np

import mindspore as ms
from mindspore import Tensor, nn, Parameter, ops, mint
import mindspore.common.dtype as mstype
from mindspore.common.initializer import initializer
from mindspore.ops.auto_generate import FusedAddTopKDiv

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.process_group_config import ModelCommProcessGroups, default_model_comm_pgs
from mindformers.parallel_core.inference.weights_utils import set_weight_attrs


__all__ = [
    'Router',
    'TopKRouter',
]


class Router(nn.Cell):
    """Base Router class"""

    def __init__(
            self, config: TransformerConfig, model_comm_pgs: Optional[ModelCommProcessGroups] = default_model_comm_pgs
    ) -> None:
        """
        Initialize the Router module.

        Args:
            config (TransformerConfig): Configuration object for the Transformer model.
        """
        super(Router, self).__init__()
        self.config = config
        self.num_experts = self.config.num_moe_experts
        self.router_dense_type = self.config.moe_router_dtype
        self.ep_group = model_comm_pgs.moe_ep
        self.ep_group_size = self.ep_group.size
        self.ep_rank = self.ep_group.rank
        self.weight = nn.Dense(in_channels=self.config.hidden_size,
                               out_channels=self.num_experts,
                               has_bias=False,
                               dtype=self.router_dense_type)
        set_weight_attrs(self.weight.weight, {"weight_loader": self.weight_loader})

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

    def weight_loader(self, param, loaded_weight):
        """
        Load weights into the parameter, supporting both full tensor loading and re-layout loading

        Args:
            param: The target parameter to load weights into.
            loaded_weight: The weight tensor to be loaded.
        """
        loaded_weight = loaded_weight[:]
        if self.ep_group_size > 1 and not self.config.use_alltoall:
            expert_idx_list = [idx for idx in range(self.num_experts)]
            start_idx = self.num_experts // self.ep_group_size * self.ep_rank
            expert_idx_list = expert_idx_list[start_idx:] + expert_idx_list[:start_idx]
            loaded_weight = loaded_weight[expert_idx_list]

        if param.shape != loaded_weight.shape:
            raise ValueError(
                f"'param.data.shape' should be equal to 'loaded_weight.shape',"
                f" but got the shape of param is {param.shape} "
                f"and the shape of weight is{loaded_weight.shape}")
        param.set_data(ms.from_numpy(loaded_weight).astype(param.dtype))

    def construct(self, input_tensor: Tensor):
        """
        Forward pass of the router.

        Args:
            input_tensor (Tensor): Input tensor.
        """
        raise NotImplementedError("For Router, construct function not implemented.")


class TopKRouter(Router):
    """Route each token to the top-k experts."""

    def __init__(
            self, config: TransformerConfig, model_comm_pgs: Optional[ModelCommProcessGroups] = default_model_comm_pgs
    ) -> None:
        """Initialize the zero token dropping router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
        """
        super().__init__(config=config, model_comm_pgs=model_comm_pgs)
        self.config = config
        self.n_group = config.moe_router_num_groups
        self.topk_group = config.moe_router_group_topk
        self.group_topk_inner = 2
        self.num_experts_chosen = config.moe_router_topk
        self.moe_router_enable_expert_bias = config.moe_router_enable_expert_bias

        self.idx_arr = Tensor(np.arange(1024, dtype=np.int32))
        self.expert_bias = Parameter(initializer('zeros', (self.num_experts), mstype.float32))
        set_weight_attrs(self.expert_bias, {"weight_loader": self.weight_loader})

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
        gating_logits = self.gating(self.cast(logits, self.router_dense_type))
        gating_logits = self.cast(gating_logits, mstype.float32)
        if self.config.moe_router_group_topk:
            expert_weight, expert_index = \
                self.fused_add_topk_div(
                    gating_logits,
                    self.expert_bias,
                    self.n_group,
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
        return expert_weight, expert_index

    def construct(self, input_tensor: Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor with shape [num_tokens, hidden_size].
        """
        expert_weight, expert_index = self.routing(input_tensor)

        return expert_weight, expert_index
