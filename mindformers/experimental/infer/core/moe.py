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
"""
Note: Mixture of Expert (MoE) structure. This is an experimental interface that is subject to change or deletion.
"""
import mindspore.common.dtype as mstype
from mindspore import Tensor, nn, Parameter, mint
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer
try:
    from mindspore.ops.auto_generate import (MoeFinalizeRouting,
                                             MoeGatingTopKSoftmax,
                                             MoeInitRouting,
                                             MoeComputeExpertTokens)
    MOE_FUSED_OP_VALID = True
except ImportError:
    MOE_FUSED_OP_VALID = False


class TopkRouter(nn.Cell):
    r"""
        A router implementation which maps each tokens to the topk expert.
    """
    def __init__(self, expert_num):
        super(TopkRouter, self).__init__()
        self.topk_bias = Parameter(initializer('zeros', (expert_num), mstype.float32),
                                   requires_grad=False, parallel_optimizer=False)


class Router(nn.Cell):
    r"""
        A router backbone used to calculate logits of each token, which should be cascaded by router implementations
        mapping tokens to experts.
    """
    def __init__(self,
                 hidden_size,
                 moe_config):
        super(Router, self).__init__()
        self.expert_num = moe_config.expert_num
        self.dense = nn.Dense(in_channels=hidden_size, out_channels=self.expert_num,
                              has_bias=False, dtype=moe_config.router_dense_type)
        self.router = TopkRouter(self.expert_num)


class ParallelMoE(nn.Cell):
    r"""
        ParallelMoE. Routing each tokens to the topk expert and calculating the final output.

        Args:
            ffn (Cell): The FeedForward Module.
            hidden_size (int): The hidden size of each token.
            moe_config (MoEConfig): The configuration of MoE (Mixture of Expert).
            use_fused_op (Bool): Whether use fused kernels.
        Inputs:
            - **input_tensor** (Tensor) - should be `[batch, seq_length, hidden_size].

        Outputs:
            - **output_tensor** (Tensor) - should be `[batch, seq_length, hidden_size].
    """

    def __init__(self,
                 ffn,
                 hidden_size,
                 moe_config,
                 use_fused_op=True):
        super(ParallelMoE, self).__init__()
        self.hidden_size = hidden_size
        self.moe_config = moe_config
        self.expert_dim = moe_config.expert_num
        self.topk_norm_prob = moe_config.norm_topk_prob
        self.num_experts_chosen = moe_config.num_experts_chosen
        self.router_dense_type = moe_config.router_dense_type
        self.use_fused_op = use_fused_op and MOE_FUSED_OP_VALID

        self.ffn = ffn
        self.router = Router(hidden_size=self.hidden_size, moe_config=moe_config)
        self.gating = self.router.dense

        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.cast = P.Cast()
        self.softmax = P.Softmax()
        self.add = P.Add()
        self.div = P.Div()
        self.mul = P.Mul()

        self.transpose_2d = P.Transpose()
        self.gather = P.Gather()
        self.onehot = P.OneHot()

        self.on_value = Tensor(1.0, dtype=mstype.float32)
        self.off_value = Tensor(0.0, dtype=mstype.float32)

        self.moe_finalize_routing = MoeFinalizeRouting()
        if self.use_fused_op:
            self.moe_init_routing = MoeInitRouting()
            self.moe_compute_expert_tokens = MoeComputeExpertTokens()
            self.moe_gating_topk_softmax = MoeGatingTopKSoftmax()
            self.moe_finalize_routing = MoeFinalizeRouting()

    def tensor_sort(self, input_tensor, expert_ids):
        '''dispatch and get unsort map for routing'''
        expert_shape = expert_ids.shape
        transposed_index = self.transpose_2d(expert_ids, (1, 0)) # (N, k) -> (k, N)
        reshaped_index = self.reshape(transposed_index, (-1,)) # (k, N) -> (kN)
        _, sort_map = mint.sort(self.cast(reshaped_index, mstype.float32))

        inter_map = mint.remainder(sort_map, expert_shape[0])
        output_tensor = self.gather(input_tensor, inter_map, 0)
        expert_mask = self.onehot(reshaped_index, self.expert_dim, self.on_value, self.off_value)
        expert_cnt = mint.sum(expert_mask, 0)
        group_list = self.cast(mint.cumsum(expert_cnt, 0), mstype.int64)

        _, unsort_map = mint.sort(self.cast(sort_map, mstype.float32))
        unsort_map = self.cast(unsort_map, mstype.int32)
        return output_tensor, group_list, unsort_map

    def tensor_sort_by_fused_op(self, input_tensor, expert_index, row_index):
        """dispatch and get unsort map for routing"""
        expanded_x, expanded_row_idx, expanded_expert_idx = \
            self.moe_init_routing(input_tensor, row_index, expert_index, self.shape(input_tensor)[0])

        expert_tokens = self.moe_compute_expert_tokens(expanded_expert_idx, self.expert_dim)
        expert_tokens = self.cast(expert_tokens, mstype.int64)
        return expanded_x, expert_tokens, expanded_row_idx

    def gating_topk_softmax(self, input_tensor):
        """calculate the expert value and expert index in MoeGatingTopKSoftmax"""
        # (N, E)
        gating_logits = self.gating(self.cast(input_tensor, self.router_dense_type))
        # (N, num_experts_chosen), (N, num_experts_chosen), (N, num_experts_chosen)
        expert_val, expert_index, row_index = \
            self.moe_gating_topk_softmax(gating_logits, finished=None, k=self.num_experts_chosen)
        return expert_val, expert_index, row_index

    def tensor_moe_finalize_routing(self, input_tensor, expert_weight, expert_index, unsort_map, bias=None):
        '''calculate the final output by multiplying FeedForward's output and experts' weight in MoeFinalizeRouting'''
        input_shape = input_tensor.shape  # (kN, h)
        x1 = mint.zeros((input_shape[0] // self.num_experts_chosen, input_shape[-1]), dtype=input_tensor.dtype)  # (N, h)
        x2 = None
        if bias is None:
            bias = mint.zeros((self.expert_dim, input_shape[-1]), dtype=input_tensor.dtype)  # (E, h)
        else:
            bias = self.reshape(bias, (self.expert_dim, input_shape[-1]))
        output_tensor = self.moe_finalize_routing(input_tensor, x1, x2, bias, expert_weight, unsort_map, expert_index)
        return output_tensor

    def construct(self, input_tensor):
        """forward process"""
        input_tensor_shape = self.shape(input_tensor)  # (B, S, H)
        input_dtype = input_tensor.dtype
        input_tensor = self.reshape(input_tensor, (-1, self.hidden_size))  # (bs, seq/1, h) -> (bs*seq, h) : use N replace bs*seq

        if self.use_fused_op:
            expert_val, expert_index, row_index = self.gating_topk_softmax(input_tensor)
            sorted_input_tensor, group_list, unsort_map = \
                self.tensor_sort_by_fused_op(input_tensor, expert_index, row_index)
        else:
            gating_logits = self.gating(self.cast(input_tensor, self.router_dense_type)) # (N, h) * (h, E) -> (bs*seq, E)
            routing_weights = self.softmax(self.cast(gating_logits, mstype.float32)) # (N, E) -> (N, E)
            expert_val, expert_index = mint.topk(routing_weights, self.num_experts_chosen)
            sorted_input_tensor, group_list, unsort_map = self.tensor_sort(input_tensor, expert_index)

        if self.moe_config.norm_topk_prob and self.num_experts_chosen > 1:
            expert_val = self.cast(expert_val, mstype.float32)
            expert_weight = self.div(expert_val, self.add(mint.sum(expert_val, -1, True), 1e-9))
        else:
            expert_weight = self.mul(self.moe_config.routed_scaling_factor, expert_val)
        expert_weight = self.cast(expert_weight, input_dtype)

        # moeffn
        expert_output = self.ffn(sorted_input_tensor, group_list)  # (N, h) (N, k) -> (N, k, h)

        expert_index = self.cast(expert_index, mstype.int32)
        moe_output = self.tensor_moe_finalize_routing(expert_output, expert_weight, expert_index, unsort_map)  # -> (N, h)

        output_tensor = self.reshape(moe_output, input_tensor_shape)  # (N, h) -> (bs, seq, h)
        return output_tensor
