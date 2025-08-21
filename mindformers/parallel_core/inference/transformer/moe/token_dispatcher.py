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
"""MoE Token Dispatcher."""
__all__ = ['MoEAllGatherTokenDispatcher', 'MoEAlltoAllTokenDispatcher']

from typing import Optional
from abc import abstractmethod

import mindspore.common.dtype as mstype
from mindspore import Tensor, ops, mint
from mindspore.ops.auto_generate import (MoeInitRoutingV2,
                                         MoeTokenUnpermute,
                                         MoeDistributeDispatch,
                                         MoeDistributeCombine)

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.process_group_config import ModelCommProcessGroups, default_model_comm_pgs
from mindformers.version_control import is_910b


class MoETokenDispatcher:
    """
    MoE Token Dispatcher

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
        model_comm_pgs (ModelCommProcessGroups, optional): Model communication process group.
            Default: default_model_comm_pgs.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(
            self, config: TransformerConfig, model_comm_pgs: Optional[ModelCommProcessGroups] = default_model_comm_pgs
    ) -> None:

        self.config = config
        self.num_experts = config.num_moe_experts

        self.tp_dp_group = model_comm_pgs.tp_dp
        self.ep_group = model_comm_pgs.moe_ep
        # use model_comm_pgs.moe_tp_group as tensor parallel group in this module.
        self.tp_group = model_comm_pgs.moe_tp

        self.tp_dp_rank = self.tp_dp_group.rank
        self.tp_size = self.tp_group.size
        self.tp_rank = self.tp_group.rank
        self.ep_size = self.ep_group.size
        self.ep_rank = self.ep_group.rank

    @abstractmethod
    def dispatch_preprocess(self, expert_weight: Tensor, routing_map: Tensor):
        """Prepares dispatch data for dispatch without inter-device communication."""
        raise NotImplementedError("dispatch_preprocess function not implemented.")

    @abstractmethod
    def token_dispatch(self, hidden_states: Tensor, routing_map: Tensor):
        """Performs the expert routing computation."""
        raise NotImplementedError("token_dispatch function not implemented.")

    @abstractmethod
    def token_combine(self, hidden_states: Tensor, expert_weight: Tensor, *args):
        """Performs expert output combine to restore original token order."""
        raise NotImplementedError("token_combine function not implemented.")


class MoEAllGatherTokenDispatcher(MoETokenDispatcher):
    """
    AllGather Based Token dispatcher.
    Note that this allgather spans the communication domain of TP*EP:

    Args:
        num_local_experts (int): Number of local experts.
        config (TransformerConfig): Configuration object for the transformer model.
        model_comm_pgs (ModelCommProcessGroups, optional): Model communication process group.
            Default: default_model_comm_pgs.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(
            self,
            num_local_experts: int,
            config: TransformerConfig,
            model_comm_pgs: Optional[ModelCommProcessGroups] = default_model_comm_pgs
    ) -> None:
        if not num_local_experts > 0:
            raise ValueError("Expected at least one expert in MoEAllGatherTokenDispatcher.")

        super().__init__(config=config, model_comm_pgs=model_comm_pgs)
        self.num_local_experts = num_local_experts
        self.group_list_index = Tensor([0,], mstype.int32)
        self.fill_value = Tensor(0, self.config.compute_dtype)

        self.cast = ops.Cast()
        self.moe_init_routing_v2 = MoeInitRoutingV2()
        self.moe_token_unpermute = MoeTokenUnpermute()

    def dispatch_preprocess(self, expert_weight, routing_map):
        """Preprocess expert weight by masking out invalid experts."""
        expert_weight_mask = routing_map >= self.num_local_experts
        expert_weight = ops.masked_fill(expert_weight, expert_weight_mask, self.fill_value)
        return expert_weight, routing_map

    def token_dispatch(self, hidden_states, routing_map):
        """Dispatch tokens to experts."""
        routing_map = self.cast(routing_map, mstype.int32)

        dispatch_input, tokens_per_expert, group_list, _ = \
            self.moe_init_routing_v2(
                hidden_states,
                routing_map,
                active_num=0,
                expert_capacity=0,
                expert_num=self.num_experts,
                drop_pad_mode=0,
                expert_tokens_count_or_cumsum_flag=2,
                expert_tokens_before_capacity_flag=True
            )

        # Avoid the problem of poor performance of the split(int32) operator
        group_list = group_list.reshape(self.ep_size, -1)
        group_list = mint.index_select(group_list, 0, self.group_list_index)
        group_list = self.cast(group_list.reshape(-1), mstype.int64)

        return dispatch_input, group_list, (tokens_per_expert,)

    def token_combine(self, hidden_states, expert_weight, *args):
        """Combines expert outputs."""
        (tokens_per_expert,) = args
        hidden_states = mint.nan_to_num(hidden_states, 0, 0, 0)
        expert_weight = expert_weight.astype(hidden_states.dtype)
        hidden_states = self.moe_token_unpermute(
            permuted_tokens=hidden_states,
            sorted_indices=tokens_per_expert,
            probs=expert_weight,
            padded_mode=False,
            restore_shape=None
        )
        return hidden_states


class MoEAlltoAllTokenDispatcher(MoETokenDispatcher):
    """
    AlltoAll Based Token dispatcher.
    Note that this allgather spans the communication domain of TP*EP:

    Args:
        num_local_experts (int): Number of local experts.
        config (TransformerConfig): Configuration object for the transformer model.
        model_comm_pgs (ModelCommProcessGroups, optional): Model communication process group.
            Default: default_model_comm_pgs.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(
            self,
            num_local_experts: int,
            config: TransformerConfig,
            model_comm_pgs: Optional[ModelCommProcessGroups] = default_model_comm_pgs
    ) -> None:
        if not num_local_experts > 0:
            raise ValueError("Expected at least one expert in MoEAllGatherTokenDispatcher.")

        super().__init__(config=config, model_comm_pgs=model_comm_pgs)
        self.num_local_experts = num_local_experts

        # Hardware-specific configurations
        self.dispatch_tp_world_size = 0 if is_910b() else 1
        self.dispatch_shared_expert_num = 0 if is_910b() else 1
        self.max_bs = 256 if is_910b() else 512
        self.dispatch_global_max_bs = min(config.dispatch_global_max_bs, self.max_bs)

        self.dispatch = MoeDistributeDispatch()
        self.combine = MoeDistributeCombine()

    def dispatch_preprocess(self, expert_weight, routing_map):
        """Identity preprocessing function that returns expert weight unchanged."""
        return expert_weight, routing_map

    def token_dispatch(self, hidden_states, routing_map):
        """Performs fused token dispatch to experts across parallel devices."""
        expand_x, _, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts, _ = self.dispatch(
            x=hidden_states,
            expert_ids=routing_map,
            ep_world_size=self.ep_size,
            ep_rank_id=self.ep_rank,
            moe_expert_num=self.num_experts,
            group_ep=self.ep_group.group,
            tp_world_size=self.dispatch_tp_world_size,
            shared_expert_num=self.dispatch_shared_expert_num,
            global_bs=self.dispatch_global_max_bs * self.ep_size,
            expert_token_nums_type=1
        )
        return expand_x, expert_token_nums, (routing_map, expand_idx, ep_recv_counts, tp_recv_counts)

    def token_combine(self, hidden_states, expert_weight, *args):
        """Combines expert outputs and restores original token ordering."""
        routing_map, expand_idx, ep_recv_counts, tp_recv_counts = args
        hidden_states = self.combine(
            expand_x=hidden_states,
            expert_ids=routing_map,
            expand_idx=expand_idx,
            ep_send_counts=ep_recv_counts,
            expert_scales=expert_weight,
            ep_world_size=self.ep_size,
            ep_rank_id=self.ep_rank,
            moe_expert_num=self.num_experts,
            tp_send_counts=tp_recv_counts,
            group_ep=self.ep_group.group,
            tp_world_size=self.dispatch_tp_world_size,
            shared_expert_num=self.dispatch_shared_expert_num,
            global_bs=self.dispatch_global_max_bs * self.ep_size)
        return hidden_states
