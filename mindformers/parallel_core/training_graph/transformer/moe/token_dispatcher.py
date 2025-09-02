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
"""MoETokenDispatcher for MoE."""

import hashlib
from abc import abstractmethod
from typing import Any
import numpy as np
import mindspore as ms
import mindspore.ops as ops
import mindspore.mint as mint
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.communication import create_group, get_rank
from mindspore.ops.operations.comm_ops import AlltoAllVC
from mindspore.ops.auto_generate import CumsumExt, FmodScalar, SortExt, IndexSelect, OneHotExt, Cast, Reshape, Zeros, Transpose, ReduceSum, MaskedSelect
from mindformers.parallel_core.transformer_config import TransformerConfig

_OEP_GROUP_NAME = {}
_IEP_GROUP_NAME = {}


class MoETokenDispatcher:
    """
    MoE Token Dispatcher
    """

    def __init__(self, config: TransformerConfig) -> None:
        """
        Initialize the MoE Token Dispatcher.
        """
        self.config = config
        self.use_pad_tokens = config.use_pad_tokens
        self.expert_num = config.num_moe_experts
        self.ep = config.expert_model_parallel_size

        self.num_out_tokens = None
        self.ep_group = self._ep_group()

        self.d2h = ops.MoveTo().add_prim_attr("recompute", False)

    def _ep_group(self):
        """Get expert model parallel group."""
        rank_id = get_rank()
        ep = self.config.expert_model_parallel_size

        rank_start = rank_id // ep * ep
        rand_end = rank_id // ep * ep + ep
        rank_list = [i for i in range(rank_start, rand_end)]

        rank_list_str = "-".join([str(i) for i in range(rank_start, rand_end)])
        hashed = hashlib.sha256(rank_list_str.encode()).hexdigest()[:48]
        ep_group_name = str(hashed)
        create_group(ep_group_name, rank_list)
        return ep_group_name

    @property
    def tp_group(self):
        """Get expert tensor parallel group."""
        return "tp_group"

    @property
    def tp_rank(self):
        """Get expert tensor parallel rank."""
        return "tp_rank"

    @property
    def tp_ep_group(self):
        """Get expert tensor and model parallel group."""
        return "tp_ep_group"

    @abstractmethod
    def token_permutation(
            self,
            tokens: Tensor,
            probs: Tensor,
            routing_map: Tensor
        ):
        """Dispatch tokens to experts.

        Args:
            tokens (torch.Tensor): Input tokens.
            probs (torch.Tensor): The routing probability tensor [num_tokens, num_experts].
            routing_map (torch.Tensor): Token to expert mapping tensor.

        Returns:
            torch.Tensor: Tokens tensor.
        """
        raise NotImplementedError("Dispatch function not implemented.")

    @abstractmethod
    def token_unpermutation(
            self,
            tokens: Tensor,
            ctx: Any
        ) -> Tensor:
        """Restores the expert output to its original ordering."""
        raise NotImplementedError("Restore function not implemented.")


class MoEAlltoAllZeroRedundancyTokenDispatcher(MoETokenDispatcher):
    """
    Implements a forward pass functionality without redundancy and  mainly used in processing input data x within
    an expert network (such as MoE, Mixture of Experts) through a series of operations including AllToAll communication,
    resorting, grouped matrix multiplication (GroupedMM), and its reverse operation.
    """

    def __init__(self, config):
        """
        Initialize the AlltoAll token dispatcher.

        Args:
            config (TransformerConfig): Configuration for the transformer model.
        """
        super().__init__(config)

        self.reshape_op = Reshape().add_prim_attr("recompute", False)
        self.mod_op = ops.Mod().add_prim_attr("recompute", False)  # aclnn operator
        self.zeros_op = Zeros().add_prim_attr("recompute", True)
        self.transpose_op = Transpose().add_prim_attr("recompute", False)
        self.cast_op = Cast().add_prim_attr("recompute", False)
        self.reduce_sum_op = ReduceSum().add_prim_attr("recompute", False)
        self.masked_select_op = MaskedSelect().add_prim_attr("recompute", False)
        self.sum_op = ReduceSum().add_prim_attr("recompute", False)
        self.cumsum_op = CumsumExt().add_prim_attr("recompute", False)
        self.index_add_op = ops.IndexAdd(axis=0).add_prim_attr("recompute", False)  # aclnn operator
        self.index_select_op = IndexSelect().add_prim_attr("recompute", False)

    def token_permutation(
            self,
            tokens: Tensor,
            probs: Tensor,
            routing_map: Tensor
    ):
        x_shape_origin = tokens.shape
        hidden_size = x_shape_origin[2]
        expert_num = self.expert_num
        local_expert_num = expert_num // self.ep
        indices = self.reshape_op(routing_map, (-1, routing_map.shape[-1]))
        probs = self.cast_op(self.reshape_op(probs, (-1, probs.shape[-1])), ms.bfloat16)
        num_tokens = indices.shape[0]  # (N, k)

        # 1 dispatch
        local_map_info = self.zeros_op((num_tokens, expert_num), probs.dtype)
        local_map_info = mint.scatter(local_map_info, 1, indices, probs)  # (N, E)
        local_map_info = self.reshape_op(local_map_info, (-1, self.ep, local_expert_num))  # (N, ep, local_E)
        local_map_info = self.transpose_op(local_map_info, (1, 0, 2))  # (ep, N, local_E)
        global_map_info = self.reshape_op(
            ops.AlltoAll(split_count=self.ep, split_dim=0, concat_dim=0, group=self.ep_group)(local_map_info),
            (-1, local_expert_num))  # (N*ep, local_E)

        # calculate send receive list
        send = mint.any(local_map_info, dim=-1)  # (ep, N) <-- (ep, N, local_E)
        receive = mint.any(self.reshape_op(global_map_info, (self.ep, num_tokens, local_expert_num)),
                           dim=-1)  # (ep, N) <-- (ep, N, local_E)
        send_list = self.cast_op(send.sum(dim=-1, keepdim=False), ms.int64)  # (ep) <-- (ep, N)
        receive_list = self.cast_op(receive.sum(dim=-1, keepdim=False), ms.int64)  # (ep) <-- (ep, N)

        send_list = ops.AllGather(group=self.ep_group)(send_list)
        receive_list = ops.AllGather(group=self.ep_group)(receive_list)
        send_list_reshape = self.reshape_op(send_list, (self.ep, -1))
        receive_list_reshape = self.reshape_op(receive_list, (self.ep, -1))
        send_list_reshape = self.d2h(send_list_reshape, "CPU", True)
        receive_list_reshape = self.d2h(receive_list_reshape, "CPU", True)

        # gather for all2allv
        send = ops.Depend()(send, (send_list_reshape, receive_list_reshape))
        select_index = self.mod_op(mint.nonzero(send.ravel()).ravel(), num_tokens)
        tokens = tokens.reshape((-1, tokens.shape[-1]))
        local_x = tokens.index_select(0, select_index)  # gather (-1, h)

        global_x = AlltoAllVC(group=self.ep_group, block_size=hidden_size)(local_x.reshape(-1),
                                                                           send_list_reshape)
        global_x_shape = global_x.shape

        # gather for gmm
        global_map_info = self.transpose_op(global_map_info, (1, 0))  # (local_E, N*ep) <-- (N*ep, local_E)
        mask_idx = mint.any(global_map_info, 0).broadcast_to((local_expert_num, num_tokens * self.ep))
        mask_idx_shape = self.reduce_sum_op(self.cast_op(mask_idx, ms.int32)) // local_expert_num
        global_map_info = self.masked_select_op(global_map_info, mask_idx)

        global_map_info = self.reshape_op(global_map_info, (local_expert_num, -1))  # (local_E, N*ep)
        token_id_recover = self.reshape_op(mint.nonzero(self.reshape_op(global_map_info, (-1,))), (-1,))
        probs = self.index_select_op(global_map_info.ravel(), 0, token_id_recover)
        token_id_recover = self.mod_op(token_id_recover, mask_idx_shape)
        global_map = self.cast_op(self.sum_op(self.cast_op(global_map_info.bool(), ms.int32), axis=1), ms.int32)
        group_list = self.cast_op(self.cumsum_op(global_map, 0), ms.int64)  # (Local_E, N) -> (local_E)

        global_x = global_x.reshape((-1, hidden_size))
        global_x = global_x.index_select(0, token_id_recover)  # gather

        ctx = (probs, global_x_shape, token_id_recover, receive_list_reshape, num_tokens, select_index, x_shape_origin)

        return global_x, group_list, ctx

    def token_unpermutation(
            self,
            tokens,
            ctx
    ):
        probs, global_x_shape, token_id_recover, receive_list_reshape, num_tokens, select_index, x_shape_origin = ctx
        hidden_size = tokens.shape[-1]
        tokens = tokens.reshape((-1, hidden_size))
        tokens = tokens * probs.unsqueeze(-1)
        # 3 combine
        permutated_local_input_tokens = self.zeros_op(global_x_shape, ms.float32)
        permutated_local_input_tokens = self.reshape_op(permutated_local_input_tokens,
                                                        (-1, hidden_size))
        permutated_local_input_tokens = self.index_add_op(permutated_local_input_tokens,
                                                          self.cast_op(token_id_recover, ms.int32),
                                                          self.cast_op(tokens, ms.float32))
        permutated_local_input_tokens = self.cast_op(permutated_local_input_tokens, tokens.dtype)
        permutated_global_input_tokens = AlltoAllVC(group=self.ep_group, block_size=hidden_size)(
            permutated_local_input_tokens.reshape(-1), receive_list_reshape)
        permutated_global_input_tokens = self.reshape_op(permutated_global_input_tokens, (-1, hidden_size))
        output = self.zeros_op((num_tokens, hidden_size), dtype=ms.float32)
        output = self.cast_op(
            self.index_add_op(output, self.cast_op(select_index, ms.int32),
                              self.cast_op(permutated_global_input_tokens, ms.float32)),
            tokens.dtype)
        output = output.reshape(x_shape_origin)
        return output


class MoEAlltoAllDeredundencyTokenDispatcher(MoETokenDispatcher):
    """
    Implements a forward pass functionality without redundecy and  mainly used in processing input data x within
    an expert network (such as MoE, Mixture of Experts) through a series of operations including AllToAll communication,
    resorting, grouped matrix multiplication (GroupedMM), and its reverse operation.
    """

    def __init__(self, config):
        """
        Initialize the AlltoAll token dispatcher.

        Args:
            config (TransformerConfig): Configuration for the transformer model.
        """
        super().__init__(config)
        self.rank_id = get_rank()

        self.outer_dp = self.config.data_parallel_size // self.ep
        self.inner_dp = self.ep
        self.iep = config.npu_nums_per_device
        self.oep = self.ep // self.iep

        node_expert_num = self.expert_num // self.oep
        ep_idx = self.rank_id % self.ep
        self.a = ep_idx // self.iep * node_expert_num
        self.b = self.a + node_expert_num

        # communication group
        self.oep_group = self._get_oep_group_name()
        self.iep_group = self._get_iep_group_name()

    def _get_oep_group_name(self):
        """
        Generates a unique group name for a set of ranks involved in outer expert partitioning (oep)
        and creates a communication group with this name.
        This method calculates a range of ranks based on the current rank id
        and the expert partition size, hashes this range to create a unique
        identifier, and then establishes a new communication group using this identifier.
        """
        rank_start = self.rank_id // self.ep * self.ep
        rank_start = rank_start + self.rank_id % self.iep
        rand_end = rank_start + self.ep
        rank_list = [i for i in range(rank_start, rand_end, self.iep)]

        rank_list_str = "-".join([str(i) for i in rank_list])
        if rank_list_str in _OEP_GROUP_NAME:
            return _OEP_GROUP_NAME[rank_list_str]

        hashed = hashlib.sha256(rank_list_str.encode()).hexdigest()[:48]
        oep_group_name = str(hashed)
        create_group(oep_group_name, rank_list)
        _OEP_GROUP_NAME[rank_list_str] = oep_group_name
        return oep_group_name

    def _get_iep_group_name(self):
        """
        Generates a unique group name for a set of ranks involved in inner expert partitioning (iep)
        and creates a communication group with this name.
        This method calculates a range of ranks based on the current rank id
        and the expert partition size, hashes this range to create a unique
        identifier, and then establishes a new communication group using this identifier.
        """
        rank_start = self.rank_id // self.iep * self.iep
        rand_end = rank_start + self.iep
        rank_list = [i for i in range(rank_start, rand_end)]

        rank_list_str = "-".join([str(i) for i in rank_list])
        if rank_list_str in _IEP_GROUP_NAME:
            return _IEP_GROUP_NAME[rank_list_str]

        hashed = hashlib.sha256(rank_list_str.encode()).hexdigest()[:48]
        iep_group_name = str(hashed)
        create_group(iep_group_name, rank_list)
        _IEP_GROUP_NAME[rank_list_str] = iep_group_name
        return iep_group_name

    def get_exdispatch_idx(self, x, expert_ids, router_coeff):
        """
        Obtain nddispatch information within nodes.
        """
        chosen_expert_num = expert_ids.shape[1]
        hidden_size = x.shape[1]
        expert_ids = expert_ids.reshape(-1)  # [nK] <-- [n,k]
        sorted_expert_ids, dispatch_idx = ops.sort(expert_ids.astype(ms.float32))
        sorted_expert_ids = sorted_expert_ids.astype(ms.int32)
        router_coeff = router_coeff.reshape(-1)  # [nK] <-- [n,k]
        sorted_router_coeff = IndexSelect()(
            router_coeff, 0, dispatch_idx)
        dispatch_idx = ops.Depend()(dispatch_idx, sorted_router_coeff)
        dispatch_idx_floordiv_k = dispatch_idx // chosen_expert_num
        sorted_expert_ids = ops.Depend()(sorted_expert_ids, dispatch_idx_floordiv_k)
        mask = ops.logical_and(sorted_expert_ids >= self.a, sorted_expert_ids < self.b)
        x = ops.Depend()(x, mask)
        x = ops.AllGather(group=self.oep_group)(x)
        idx = mask.reshape(-1).nonzero()
        idx = idx.reshape(-1)
        dispatch_idx = IndexSelect()(dispatch_idx_floordiv_k, 0, idx)
        sorted_expert_ids = IndexSelect()(
            sorted_expert_ids, 0, idx)
        sorted_router_coeff = IndexSelect()(
            sorted_router_coeff, 0, idx)
        x = x.reshape(-1, hidden_size)
        return x, dispatch_idx, sorted_expert_ids, sorted_router_coeff

    def token_permutation(
            self,
            tokens: Tensor,
            probs: Tensor,
            routing_map: Tensor
        ):
        x_orig_shape = tokens.shape
        x = ops.squeeze(tokens, 0)
        expert_id = ops.squeeze(routing_map, 0).astype(ms.int32)
        router_coeff = ops.squeeze(probs, 0).astype(ms.bfloat16)

        hidden_size = x.shape[1]
        chosen_expert_nums = expert_id.shape[1]
        node_expert_num = self.b - self.a
        if self.use_pad_tokens:
            safe_tokens = ms.ops.zeros((node_expert_num, hidden_size), ms.bfloat16)
            x = ops.concat((safe_tokens, x), 0)
            safe_expert_ids = ms.ops.cumsum(ms.ops.ones((node_expert_num, chosen_expert_nums)), 0)
            safe_expert_ids = (safe_expert_ids - 1 + self.a).astype(ms.int32)
            expert_id = ops.concat((safe_expert_ids, expert_id), 0)
            safe_router_coeff = ms.ops.zeros((node_expert_num, chosen_expert_nums), ms.bfloat16)
            router_coeff = ops.concat((safe_router_coeff, router_coeff), 0)

        # prepare counter
        iepones = [node_expert_num // self.iep for i in range(self.iep)]
        expert_id = ops.AllGather(group=self.oep_group)(expert_id).reshape(-1, chosen_expert_nums)
        excounter = P.OneHot()(
            expert_id.reshape(-1), self.expert_num, Tensor(1, dtype=ms.float32), Tensor(0, dtype=ms.float32)
        )
        excounter = excounter.sum(axis=0)[self.a: self.b]
        local_excounter = ops.AlltoAllV(group=self.iep_group, block_size=1)(excounter.reshape(-1), iepones, iepones)
        exrl = ops.cast(local_excounter.reshape(self.iep, -1).sum(axis=1), ms.int64)  # [outer_ep]
        exgl = ops.cast(
            ops.cumsum(local_excounter.reshape(self.iep, -1).sum(dim=-2, keepdim=False), 0, ms.int32), ms.int64
        )
        exsl = ops.cast(excounter.reshape(self.iep, -1).sum(axis=1), ms.int64)  # [outer_ep]
        exgl = ops.Depend()(exgl, exrl)
        exsl = ops.Depend()(exsl, exgl)

        router_coeff = ops.Depend()(router_coeff, exgl)
        expert_id = ops.Depend()(expert_id, exsl)

        # 1. allgather
        router_coeff = ops.AllGather(group=self.oep_group)(router_coeff).reshape(-1, chosen_expert_nums)

        # 2. exdispatch
        x, exdispatch_idx, expert_id, router_coeff = self.get_exdispatch_idx(x, expert_id, router_coeff)
        exgl = ops.Depend()(exgl, x)
        exsl = ops.Depend()(exsl, exdispatch_idx)
        exsl = self.d2h(exsl, "CPU", True)
        exrl = self.d2h(exrl, "CPU", True)
        excombine_whiteboard = x * Tensor(0.0, dtype=ms.bfloat16)
        x = IndexSelect()(x, 0, exdispatch_idx)

        # 3. inner alltoallv
        x = ops.AlltoAllV(group=self.iep_group, block_size=hidden_size)(x.reshape(-1), exsl, exrl).reshape(
            -1, hidden_size
        )
        expert_id = ops.Depend()(expert_id, x)
        expert_id = ops.AlltoAllV(group=self.iep_group, block_size=1)(expert_id.reshape(-1), exsl, exrl)

        # 4. resort
        _, sort_map = ops.sort(expert_id.astype(ms.float32))
        _, unsort_map = ops.sort(sort_map.astype(ms.float32))
        x = IndexSelect()(x, 0, sort_map)

        ctx = (router_coeff, unsort_map, exrl, exsl, excombine_whiteboard, exdispatch_idx, x_orig_shape)
        return x, exgl, ctx

    def token_unpermutation(
            self,
            tokens,
            ctx
        ):
        """Restores the expert output to its original ordering."""
        probs, unsort_map, exrl, exsl, excombine_whiteboard, exdispatch_idx, x_orig_shape = ctx
        # -4. unresort
        x = IndexSelect()(tokens, 0, unsort_map)

        # -3. allToAllv
        hidden_size = x_orig_shape[-1]
        x = ops.AlltoAllV(group=self.iep_group, block_size=hidden_size)(x.reshape(-1), exrl, exsl).reshape(
            -1, hidden_size
        )

        # -2. excombine
        x = ops.mul(probs.unsqueeze(1), x)
        x = excombine_whiteboard.index_add_(0, exdispatch_idx.reshape(-1), x)

        # -1 reduce scatter
        x = ops.ReduceScatter(group=self.oep_group)(x)
        if self.use_pad_tokens:
            node_expert_num = self.b - self.a
            x = x[node_expert_num:]
        return x.reshape(x_orig_shape)


class MoEAlltoAllTokenDispatcher(MoETokenDispatcher):
    """
    AlltoAll-based token dispatcher.

    The workflow of AlltoAll token dispatcher is as follows:
    (1) preprocess(): calculate necessary metadata for communication and permute
    (2) token_permutation(): permute->A2A(EP)->AG(TP)->sort_chunk(if num_local_experts>1)
    (3) token_unpermutation(): sort_chunk(if num_local_experts>1)->RS(TP)->A2A(EP)->unpermute
    """

    def __init__(self, config):
        """
        Initialize the AlltoAll token dispatcher.

        Args:
            config (TransformerConfig): Configuration for the transformer model.
        """
        super().__init__(config)

        self.moe_permute_fusion = config.moe_permute_fusion
        self.hidden_size = config.hidden_size
        self.moe_router_topk = config.moe_router_topk

        self.dp = config.data_parallel_size * config.tensor_model_parallel_size

        # compute parameters
        self.on_value = Tensor(1.0, dtype=ms.int32)
        self.off_value = Tensor(0.0, dtype=ms.int32)
        self.pad_tokens = Tensor(np.zeros((1, self.expert_num, self.hidden_size)), dtype=ms.bfloat16)
        self.pad_routing_map = Tensor(np.arange(self.expert_num * self.moe_router_topk).reshape(
            1, self.expert_num, self.moe_router_topk) % self.expert_num, ms.int32)
        self.pad_probs = Tensor(np.zeros((1, self.expert_num, self.moe_router_topk)), dtype=ms.bfloat16)

        self.cumsum = CumsumExt()
        self.sort = SortExt()

    def preprocess(self, num_local_tokens_per_expert):
        """
        Preprocess token routing map for AlltoAll communication and token permutation.

        This method computes the number of tokens assigned to each expert based on the routing_map.
        It also initializes the necessary data structures for AlltoAll communication, such as input
        and output splits, and the mapping between global tokens and local experts.
        """
        num_global_tokens_per_expert = ops.AlltoAll(
            split_count=self.ep,
            split_dim=-1,
            concat_dim=-2,
            group=self.ep_group)(num_local_tokens_per_expert)

        # [ep, E/ep]  -->  [ep]
        input_splits = ops.cast(
            num_local_tokens_per_expert.reshape(self.ep, -1).sum(dim=-1, keepdim=False), ms.int64)
        # [ep, E/ep]  --> [ep]
        output_splits = ops.cast(
            num_global_tokens_per_expert.reshape(self.ep, -1).sum(dim=-1, keepdim=False), ms.int64)
        # [ep, E/ep]  --> (E/ep) int64
        tokens_per_expert = ops.cast(self.cumsum(
            num_global_tokens_per_expert.reshape(self.ep, -1).sum(dim=-2, keepdim=False), 0), ms.int64)

        input_splits = ops.Depend()(input_splits, output_splits)
        input_splits = ops.Depend()(input_splits, tokens_per_expert)
        input_splits = self.d2h(input_splits, "CPU", True)
        output_splits = self.d2h(output_splits, "CPU", True)

        return tokens_per_expert, input_splits, output_splits

    def _process_pad_tokens(self, tokens, routing_map, probs):
        """
        Prepares the input tensors by padding them with special tokens.
        These tokens are designed to ensure stable training computation,
        especially in distributed or parallel computing environments
        where data might need to be padded to fit certain dimensions.
        """
        # Ensure each expert has token.
        tokens = ops.concat((self.pad_tokens.astype(tokens.dtype), tokens), axis=1)
        routing_map = ops.concat(
            (self.pad_routing_map.astype(routing_map.dtype), routing_map), axis=1)
        probs = ops.concat((self.pad_probs.astype(probs.dtype), probs), axis=1)

        return tokens, routing_map, probs

    def token_permutation(
            self,
            tokens: Tensor,
            probs: Tensor,
            routing_map: Tensor
    ):
        """Dispatch tokens to experts.

        Args:
            tokens (torch.Tensor): Input tokens.
            probs (torch.Tensor): The routing probability tensor [num_tokens, num_experts].
            routing_map (torch.Tensor): Token to expert mapping tensor.
        """
        # Whether to pad tokens for each expert.
        if self.use_pad_tokens:
            tokens, routing_map, probs = self._process_pad_tokens(tokens, routing_map, probs)

        # Permutation 1: input to AlltoAll input
        permuted_input, routing_map, reshaped_map, outer_unsort_map = permute(
            tokens,
            routing_map,
            self.moe_permute_fusion,
        )

        num_tokens_per_expert = ops.sum(OneHotExt()(
            reshaped_map.astype(ms.int32), self.expert_num, self.on_value, self.off_value), 1)
        num_tokens_per_expert = Cast()(num_tokens_per_expert, ms.float32)

        tokens_per_expert, input_splits, output_splits = self.preprocess(num_tokens_per_expert)

        # Perform expert parallel AlltoAll communication
        # The shape change is: global_input_tokens <- [B, S, h]
        original_shape = permuted_input.shape
        global_input_tokens = ops.AlltoAllV(group=self.ep_group, block_size=self.hidden_size)(
            permuted_input.reshape(-1), input_splits, output_splits).reshape(1, -1, self.hidden_size)
        # The shape change is: routing_map <- [B, S]
        routing_map = ops.AlltoAllV(group=self.ep_group, block_size=1)(
            routing_map.astype(ms.float32).reshape(-1), input_splits, output_splits).reshape(1, -1)

        # Permutation 2: Sort tokens by local expert.
        if self.moe_permute_fusion:
            tokens_shape = global_input_tokens.shape
            tokens_dtype = global_input_tokens.dtype
            # permute only support bfloat16 for now
            global_input_tokens = ops.reshape(ops.cast(
                global_input_tokens, ms.bfloat16), (-1, tokens_shape[-1]))
            routing_map = ops.reshape(routing_map, (-1,))
            global_input_tokens, unsort_map = ops.moe_token_permute(
                global_input_tokens, routing_map.astype(ms.int32))
            global_input_tokens = ops.reshape(ops.cast(
                global_input_tokens, tokens_dtype), tokens_shape)
        else:
            _, sort_map = self.sort(routing_map)
            _, unsort_map = self.sort(sort_map.astype(ms.float32))
            index = mint.reshape(sort_map, (sort_map.shape[0]*sort_map.shape[1],))
            global_input_tokens = IndexSelect()(global_input_tokens, 1, index)

        ctx = (
            probs, unsort_map, outer_unsort_map, input_splits,
            output_splits, original_shape
        )
        return global_input_tokens, tokens_per_expert, ctx

    def token_unpermutation(
            self,
            tokens,
            ctx
    ):
        """
        Reverse the token permutation to restore the original order.

        This method performs the following steps:
        1. Unsort tokens by local expert (if multiple local experts exist).
        2. Perform expert parallel AlltoAll communication to restore the original order.
        3. Unpermute tokens to restore the original order.
        """
        probs, unsort_map, outer_unsort_map, input_splits, \
            output_splits, original_shape = ctx
        tokens = tokens.reshape((1, -1, self.hidden_size))

        # Unpermutation 2: Unsort tokens by local expert.
        if self.moe_permute_fusion:
            tokens_shape = tokens.shape
            tokens_dtype = tokens.dtype
            # permute only support bfloat16 for now
            tokens = ops.reshape(ops.cast(tokens, ms.bfloat16), (-1, tokens_shape[-1]))
            tokens = ops.moe_token_unpermute(tokens, unsort_map)
            tokens = ops.reshape(ops.cast(tokens, tokens_dtype), tokens_shape)
        else:
            index = mint.reshape(unsort_map, (unsort_map.shape[0]*unsort_map.shape[1],))
            tokens = IndexSelect()(tokens, 1, index)

        # Perform expert parallel AlltoAll communication
        permutated_local_input_tokens = ops.AlltoAllV(group=self.ep_group, block_size=self.hidden_size)(
            tokens.reshape(-1), output_splits, input_splits).reshape(1, -1, self.hidden_size)
        permutated_local_input_tokens = permutated_local_input_tokens.reshape(original_shape)

        # Unpermutation 1: AlltoAll output to output
        output = unpermute(
            permutated_local_input_tokens,
            outer_unsort_map,
            probs,
            self.moe_permute_fusion
        )

        # remove pad tokens
        if self.use_pad_tokens:
            output = ops.strided_slice(
                output,
                (0, self.pad_routing_map.shape[1], 0),
                (output.shape[0], output.shape[1], output.shape[2]),
                (1, 1, 1)
            )

        return output


def permute(
        tokens,
        routing_map,
        moe_permute_fusion: bool
):
    """Permute the tokens and probs based on the mask.
    Tokens with the same designated expert will be grouped together.
    The shape of mask is [tokens, num_experts], it indicates which experts were selected
    by each token.

    Args:
        tokens (Tensor): The input token tensor, [num_tokens, hidden].
        routing_map (Tensor): The sparse token to expert mapping, [num_tokens, num_experts].
    """
    # preprocess routing_shape
    routing_shape = routing_map.shape

    if moe_permute_fusion:
        # preprocess routing_shape
        routing_map_kn = ops.transpose(routing_map, (0, 2, 1))
        routing_map_kn = ops.reshape(routing_map_kn, (1, -1))
        sorted_routing_map, _ = SortExt()(ops.cast(routing_map_kn, ms.float32), dim=1)

        tokens = ops.reshape(tokens, (-1, tokens.shape[-1]))
        routing_map = ops.reshape(routing_map, (-1, routing_map.shape[-1]))
        permuted_input, unsort_map = ops.moe_token_permute(tokens, routing_map.astype(ms.int32))
        permuted_input = permuted_input.reshape(1, permuted_input.shape[0], permuted_input.shape[1])
        unsort_map = unsort_map.reshape(routing_shape[0], routing_shape[1], routing_shape[2])

        return permuted_input, sorted_routing_map, routing_map_kn, unsort_map

    # (dp, k, N)int32  <-- (dp, N, k)int32
    routing_map = ops.transpose(routing_map, (0, 2, 1))
    # (dp, kN)int32    <-- (dp, k, N)int32
    routing_map = ops.reshape(routing_map, (1, -1))

    # (dp, kN)fp32, (dp, kN)int32 <-- (dp, kN)fp32 <-- (dp, kN)int32
    sorted_routing_map, sort_map = SortExt()(ops.cast(routing_map, ms.float32), dim=1)

    # compute unsort_map for unpermutation
    # _, (dp, kN)int32 <--  (dp, kN)fp32 <-- (dp, kN)int32
    _, unsort_map = SortExt()(ops.cast(sort_map, ms.float32), dim=1)
    # (dp, k, N)int32  <--  (dp, kN)int32
    unsort_map = ops.reshape(unsort_map, (routing_shape[0], routing_shape[2], routing_shape[1]))
    # (dp, N, k)int32  <--  (dp, k, N)int32
    unsort_map = ops.transpose(unsort_map, (0, 2, 1))

    # use the mapping to permute the tokens
    # (dp, kN)int32    <--  (dp, kN)int32, N int32
    inter_map = FmodScalar()(sort_map, routing_shape[1])
    # (dp, kN, h)bf16  <--  (dp, N, h)bf16, (dp, kN)int32
    index = mint.reshape(inter_map, (inter_map.shape[0]*inter_map.shape[1],))
    permuted_input = IndexSelect()(tokens, 1, index)

    return permuted_input, sorted_routing_map, routing_map, unsort_map


def unpermute(
        permuted_tokens: Tensor,
        unsort_map: Tensor,
        probs: Tensor,
        moe_permute_fusion: bool
):
    """
    Restore the original order of tokens after permutation. If probs are provided, it
    will also apply them to the tokens before restoring the order.

    Args:
        permuted_tokens (Tensor): The permuted token tensor.
        unsort_map (Tensor): The unsort map for permuted tokens.
        probs (Tensor): The unpermuted probs tensor.

    Returns:
        Tensor: The tokens restored to their original order.
    """
    if moe_permute_fusion:
        unsort_map_shape = unsort_map.shape
        permuted_tokens = permuted_tokens.reshape(-1, permuted_tokens.shape[-1])
        unsort_map = unsort_map.reshape(-1,)
        output_tokens = ops.moe_token_unpermute(permuted_tokens, unsort_map.astype(ms.int32))
        output_tokens = output_tokens.reshape(unsort_map_shape[0], unsort_map_shape[1], unsort_map_shape[2], -1)
    else:
        index = mint.reshape(unsort_map, (unsort_map.shape[0]*unsort_map.shape[1]*unsort_map.shape[2],))
        output_tokens = IndexSelect()(permuted_tokens, 1, index)
        output_tokens = mint.reshape(output_tokens, (unsort_map.shape[0], unsort_map.shape[1], unsort_map.shape[2], -1))
    # (dp, N, k, 1)fp32 <-- (dp, N, k)fp32
    probs = ops.reshape(probs, (probs.shape[0], probs.shape[1], probs.shape[2], 1))
    # (dp, N, k, h)bf16 <-- (dp, N, k, h)bf16, (dp, N, k, 1)bf16
    output_tokens = ops.mul(output_tokens, ops.cast(probs, output_tokens.dtype))
    # (dp, N, h)bf16    <-- (dp, N, k, h)bf16
    output_tokens = ops.ReduceSum(keep_dims=False)(output_tokens, 2)
    return output_tokens
