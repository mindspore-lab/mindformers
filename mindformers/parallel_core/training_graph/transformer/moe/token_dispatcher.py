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
import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore.common.tensor import Tensor
from mindspore.communication import create_group, get_rank
from mindspore.ops.auto_generate import AddExt, CumsumExt, FmodScalar, SortExt
from mindformers.parallel_core.transformer_config import TransformerConfig


class MoETokenDispatcher:
    """
    MoE Token Dispatcher
    """

    def __init__(self, config: TransformerConfig) -> None:
        """
        Initialize the MoE Token Dispatcher.
        """
        self.config = config
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
        hashed = hashlib.md5(rank_list_str.encode()).hexdigest()[:48]
        ep_group_name = str(hashed)
        create_group(ep_group_name, rank_list)
        return ep_group_name

    @property
    def tp_group(self):
        """Get expert tensor parallel group."""
        return 'tp_group'

    @property
    def tp_rank(self):
        """Get expert tensor parallel rank."""
        return 'tp_rank'

    @property
    def tp_ep_group(self):
        """Get expert tensor and model parallel group."""
        return 'tp_ep_group'

    @abstractmethod
    def token_permutation(
            self, tokens: Tensor, probs: Tensor, routing_map: Tensor
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
    def token_unpermutation(self, tokens, probs):
        """Restores the expert output to its original ordering."""
        raise NotImplementedError("Restore function not implemented.")


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

        self.expert_num = config.num_moe_experts
        self.use_pad_tokens = config.use_pad_tokens
        self.moe_permute_fusion = config.moe_permute_fusion
        self.hidden_size = config.hidden_size

        self.ep = config.expert_model_parallel_size
        self.dp = config.data_parallel_size * config.tensor_model_parallel_size

        # compute parameters
        self.on_value = Tensor(1.0, dtype=ms.float32)
        self.off_value = Tensor(0.0, dtype=ms.float32)
        self.pad_tokens = Tensor(np.zeros((1, self.expert_num, self.hidden_size)), dtype=ms.bfloat16)
        self.pad_routing_map = Tensor(np.tile(np.arange(self.expert_num), (1, 1)), dtype=ms.int32)

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

    def _process_pad_tokens(self, tokens, routing_map, num_tokens_per_expert):
        """
        Prepares the input tensors by padding them with special tokens.
        These tokens are designed to ensure stable training computation,
        especially in distributed or parallel computing environments
        where data might need to be padded to fit certain dimensions.
        """
        tokens = ops.concat((self.pad_tokens.astype(tokens.dtype), tokens), axis=1)
        routing_map = ops.concat(
            (self.pad_routing_map.astype(routing_map.dtype), routing_map), axis=1)

        num_tokens_per_expert = AddExt()(num_tokens_per_expert, 1)
        # sort [safe_tokens, x] together
        # (dp, E+kN)fp32 <-- (dp, E+kN)fp32
        routing_map, sort_map = self.sort(routing_map.astype(ms.float32), dim=1)
        _, unsort_pad_map = self.sort(sort_map.astype(ms.float32), dim=1)
        tokens = ops.gather(tokens, sort_map, axis=1, batch_dims=1)
        return tokens, routing_map, num_tokens_per_expert, unsort_pad_map

    def token_permutation(
            self, tokens: Tensor, probs: Tensor, routing_map: Tensor
    ):
        """Dispatch tokens to experts.

        Args:
            tokens (torch.Tensor): Input tokens.
            probs (torch.Tensor): The routing probability tensor [num_tokens, num_experts].
            routing_map (torch.Tensor): Token to expert mapping tensor.
        """
        # Permutation 1: input to AlltoAll input
        permuted_input, routing_map, reshaped_map, outer_unsort_map = permute(
            tokens,
            routing_map,
        )

        num_tokens_per_expert = ops.sum(ops.OneHot()(
            reshaped_map.astype(ms.int32), self.expert_num, self.on_value, self.off_value), 1)
        unsort_pad_map = 0
        if self.use_pad_tokens:
            permuted_input, routing_map, num_tokens_per_expert, unsort_pad_map = self._process_pad_tokens(
                permuted_input, routing_map, num_tokens_per_expert)

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
            global_input_tokens = ops.gather(global_input_tokens, sort_map, axis=1, batch_dims=1)

        return (global_input_tokens, tokens_per_expert, unsort_map, outer_unsort_map,
                input_splits, output_splits, original_shape, unsort_pad_map)

    def token_unpermutation(
            self, tokens, probs, unsort_map=None, outer_unsort_map=None,
            input_splits=None, output_splits=None, original_shape=None, unsort_pad_map=None
    ):
        """
        Reverse the token permutation to restore the original order.

        This method performs the following steps:
        1. Unsort tokens by local expert (if multiple local experts exist).
        2. Perform expert parallel AlltoAll communication to restore the original order.
        3. Unpermute tokens to restore the original order.
        """

        # Unpermutation 2: Unsort tokens by local expert.
        if self.moe_permute_fusion:
            tokens_shape = tokens.shape
            tokens_dtype = tokens.dtype
            # permute only support bfloat16 for now
            tokens = ops.reshape(ops.cast(tokens, ms.bfloat16), (-1, tokens_shape[-1]))
            tokens = ops.moe_token_unpermute(tokens, unsort_map)
            tokens = ops.reshape(ops.cast(tokens, tokens_dtype), tokens_shape)
        else:
            tokens = ops.gather(tokens, unsort_map, axis=1, batch_dims=1)

        # Perform expert parallel AlltoAll communication
        permutated_local_input_tokens = ops.AlltoAllV(group=self.ep_group, block_size=self.hidden_size)(
            tokens.reshape(-1), output_splits, input_splits).reshape(1, -1, self.hidden_size)
        permutated_local_input_tokens = permutated_local_input_tokens.reshape(original_shape)

        if self.use_pad_tokens:
            tokens_shape = permutated_local_input_tokens.shape
            permutated_local_input_tokens = ops.gather(
                permutated_local_input_tokens, unsort_pad_map, axis=1, batch_dims=1)
            permutated_local_input_tokens = ops.strided_slice(
                permutated_local_input_tokens,
                (0, self.pad_routing_map.shape[1], 0),
                (tokens_shape[0], tokens_shape[1], tokens_shape[2]),
                (1, 1, 1)
            )

        # Unpermutation 1: AlltoAll output to output
        output = unpermute(
            permutated_local_input_tokens,
            outer_unsort_map,
            probs,
        )
        return output


def permute(
        tokens,
        routing_map
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
    permuted_input = ops.gather(tokens, inter_map, axis=1, batch_dims=1)

    return permuted_input, sorted_routing_map, routing_map, unsort_map


def unpermute(
        permuted_tokens: Tensor,
        unsort_map: Tensor,
        probs: Tensor
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
    output_tokens = ops.gather(permuted_tokens, unsort_map, axis=1, batch_dims=1)
    # (dp, N, k, 1)fp32 <-- (dp, N, k)fp32
    probs = ops.reshape(probs, (probs.shape[0], probs.shape[1], probs.shape[2], 1))
    # (dp, N, k, h)bf16 <-- (dp, N, k, h)bf16, (dp, N, k, 1)bf16
    output_tokens = ops.mul(output_tokens, ops.cast(probs, output_tokens.dtype))
    # (dp, N, h)bf16    <-- (dp, N, k, h)bf16
    output_tokens = ops.ReduceSum(keep_dims=False)(output_tokens, 2)
    return output_tokens
