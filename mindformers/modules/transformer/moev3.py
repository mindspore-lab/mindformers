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
"""
Note: Mixture of Expert (MoE) structure. This is an experimental interface that is subject to change or deletion.
"""
from __future__ import absolute_import
from __future__ import division

import hashlib
import numpy as np
import mindspore.ops as ops
import mindspore as ms
from mindspore.communication import get_rank, create_group
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer, Normal
import mindspore.common.dtype as mstype

from mindspore.ops import operations as P
from mindspore.ops.operations import Shape, Sort, Mod, Gather, CumSum, ReduceSum, ReduceMean, AssignAdd, StridedSlice, OneHot
import mindspore.nn as nn
from mindspore.nn.cell import Cell
from mindspore.nn.layer import Dense
from mindspore.ops.auto_generate import GroupedMatmul, Reshape, Cast, Softmax, TopkExt, Mul, Transpose, AddExt, Concat, Div
from mindspore.parallel.shard import Layout
from mindformers.tools.logger import logger

from mindformers.modules.transformer.op_parallel_config import default_moeparallel_config
from mindformers.modules.transformer.moe import default_moe_config


class MoEV3(Cell):
    """
    The mixture of experts (MoE) implementation (using GMM op instead of BMM op used by MOE module).
    The implementation includes a router and a FeedForward layer.
    The router dispatches tokens to experts in FeedForward, then FeedForward does computation, and the final output is
    obtained by multiplying FeedForward's output and router's combine weight.

    Args:
        dim (int): The dimension of the inputs.
        intermediate_size (int): The intermediate hidden size.
        compute_dtype (dtype): The data type of the computation.
        param_init_type (dtype.Number): The parameter initialization type. Can be dtype.float32 or dtype.float16.
        return_extra_loss (bool): whether to return extra regularization loss. Default False.
        moe_config(MoEConfig): The configuration of MoE (Mixture of Expert). Default is an instance of MoEConfig with
            default values. Please see `MoEConfig`.
        parallel_config(MoEParallelConfig): The parallel config for MoE, see `MoEParallelConfig`.
            Default `default_moeparallel_config`, an instance of `MoEParallelConfig` with default args.
        init_method_std (float): Standard deviation of the zero mean normal for the default initialization method,
                not used if init_method and output_layer_init_method are provided. Default: 0.01.
    Inputs:
        - **x** (Tensor) - should be `[batch, seq_length, hidden_size]`. Float tensor.
        - **extra_loss**  (float) , control expert load balance.

    Outputs:
        Tensor, the output of this layer after mapping. The shape is `[batch, seq_length, hidden_size]`.
    """

    def __init__(self,
                 dim,
                 intermediate_size,
                 compute_dtype,
                 param_init_type,
                 return_extra_loss=False,
                 moe_config=default_moe_config,
                 parallel_config=default_moeparallel_config,
                 init_method_std=0.01):
        super(MoEV3, self).__init__()
        print("use MoEV3 computing via GroupedMatMul. Capacity factor is ignored.")
        self.hidden_size = dim
        self.intermediate_size = intermediate_size
        self.compute_dtype = compute_dtype
        self.param_init_type = param_init_type
        self.return_extra_loss = return_extra_loss
        self.moe_config = moe_config
        self.parallel_config = parallel_config

        self.dp = parallel_config.data_parallel * parallel_config.model_parallel
        self.expert_dim = moe_config.expert_num
        self.num_experts_chosen = moe_config.num_experts_chosen
        self.aux_loss_config = dict(zip(moe_config.aux_loss_types, moe_config.aux_loss_factors))
        self.aux_loss_factor = self.aux_loss_config.get("expert", 0.0)
        self.init_method_std = init_method_std

        self.shape = Shape()
        self.reshape = Reshape()
        self.cast = Cast()
        self.gating_activation = Softmax(axis=-1).shard(
            ((self.dp, 1, 1,),)) if not moe_config.use_gating_sigmoid else P.Sigmoid().shard(((self.dp, 1, 1,),))
        self.topk = TopkExt().shard(((self.dp, 1, 1),))
        self.topk.recompute(False)
        self.mul = Mul().shard(((), (self.dp, 1, 1)))

        # _tensor_sort
        self.transpose_3d = Transpose().shard(((self.dp, 1, 1),))
        self.sort = Sort(1).shard(((self.dp, 1),))
        self.mod = Mod().shard(((self.dp, 1), ()))
        self.gather_sort = Gather(batch_dims=1).shard(((self.dp, 1, 1), (self.dp, 1)))
        self.onehot = OneHot().shard(((self.dp, 1, 1), (), ()))
        self.cumsum = CumSum(exclusive=False).shard(((self.dp, 1),))
        self.on_value = Tensor(1.0, dtype=mstype.float32)
        self.off_value = Tensor(0.0, dtype=mstype.float32)
        self.transpose_unsort_map = Transpose().shard(((self.dp, 1, 1),))

        # _tensor_unsort
        self.gather_unsort = Gather(batch_dims=1).shard(((self.dp, 1, 1), (self.dp, 1, 1)))
        self.mul_router_coeff = Mul().shard(((self.dp, 1, 1, 1), (self.dp, 1, 1, 1)))
        self.sum_router_coeff = ReduceSum(keep_dims=False).shard(((self.dp, 1, 1, 1),))

        # _normalize
        self.reduce_sum_keep = ReduceSum(keep_dims=True).shard(((self.dp, 1, 1),))
        self.add_eps = AddExt().shard(((self.dp, 1, 1), ()))
        self.div_3d = Div().shard(((self.dp, 1, 1), (self.dp, 1, 1)))

        # _aux
        self.reduce_mean_aux_3d = ReduceMean(keep_dims=False).shard(((self.dp, 1, 1),))
        self.reduce_mean_aux_2d = ReduceMean(keep_dims=False).shard(((self.dp, 1),))
        self.mul_aux_2d = Mul().shard(((self.dp, 1), (self.dp, 1)))
        self.onehot_aux = OneHot().shard(((self.dp, 1, 1), (), ()))
        self.mul_noshard = Mul().shard(((), ()))
        self.add_loss = AddExt().shard(((1,), ()))

        # _topk
        # aux loss free
        if self.moe_config.balance_via_topk_bias:
            self.topk_bias = Parameter(initializer('zeros', (self.expert_dim), mstype.float32),
                                       requires_grad=False, parallel_optimizer=False)
            self.gate_gather = Gather(batch_dims=2).shard(((self.dp, 1, 1), (self.dp, 1, 1)))
            self.expert_load = Parameter(initializer('zeros', (self.expert_dim), mstype.float32),
                                         requires_grad=False, parallel_optimizer=False)
            self.assign_add = AssignAdd().shard(((1,), (1,)))
            self.assign_add.recompute(False)
            self.onehot_2d = OneHot().shard(((self.dp, 1, 1), (), ()))
            self.reduce_mean = ReduceMean(keep_dims=False).shard(((self.dp, 1, 1),))
            self.afb_reduce_mean = ReduceMean(keep_dims=False).shard(((1, 1),))
            self.afb_topk = TopkExt().shard(((self.dp, 1, 1),))
            self.afb_topk.recompute(False)
            self.afb_add_topk_bias = AddExt().shard(((self.dp, 1, 1), (1,)))
            self.afb_add_topk_bias.recompute(False)

        # ffn
        self.ffn = FFN(self.hidden_size, self.intermediate_size, self.compute_dtype, self.param_init_type,
                       self.moe_config, self.parallel_config, self.init_method_std)

        # dense
        self.router_dense_type = moe_config.router_dense_type
        self.router_dense = Dense(in_channels=self.hidden_size, out_channels=self.expert_dim,
                                  weight_init=initializer(Normal(sigma=self.init_method_std, mean=0.0),
                                                          [self.expert_dim, self.hidden_size], self.router_dense_type),
                                  has_bias=False, dtype=self.router_dense_type)
        self.router_dense.matmul.shard(((self.dp, 1), (1, 1)))

    def construct(self, input_tensor, extra_loss=0.):
        """forward process"""
        input_tensor_shape = self.shape(input_tensor)
        # (dp, N, h) <-- (B*S, h)
        input_tensor = self.reshape(input_tensor, (self.dp, -1, self.hidden_size))

        # 1.gating
        # (dp, N, E) fp32 <-- (dp, N, h)
        router_logits = self.router_dense(input_tensor.astype(self.router_dense_type))
        # (dp, N, E) fp32 <-- (dp, N, E) fp32
        router_prob = self.gating_activation(router_logits)
        # (dp, N, k) fp32,  (dp, N, k) int32 <-- (dp, N, E) fp32
        expert_gate, expert_index = self._topk(router_prob)
        if self.num_experts_chosen > 1 and self.moe_config.norm_topk_prob:
            # (dp, N, k) fp32 <-- (dp, N, k) fp32
            router_coeff = self._normalize(expert_gate)
        else:
            router_coeff = self.mul(self.moe_config.routed_scaling_factor, expert_gate)
        # float32 <-- (dp, N, E) fp32, (dp, N, k) int32, float32
        router_aux_loss = self._expert_load_balancing(router_prob, expert_index, self.aux_loss_factor)

        # 2.dispatch sort
        # (dp, kN, h) bf16, (dp, kN) fp32,  (dp, E) int32, (dp, N, k)int32 <-- (dp, N, h) bf16, (dp, N, k) int32
        expert_input, expert_index_sorted, expert_cnt, unsort_map = self._tensor_sort(input_tensor, expert_index)

        # 3.ffn
        expert_output = self.ffn(expert_input, expert_index_sorted, expert_cnt)

        # 4.combine unsort
        # (dp, N, h)bf16 <-- (dp, kN, h)bf16, (dp, N, k) fp32, (dp, N, k)int32
        output_tensor = self._tensor_unsort(expert_output, router_coeff, unsort_map)
        output_tensor = self.reshape(output_tensor, input_tensor_shape)
        if self.return_extra_loss:
            final_extra_loss = self.add_loss(extra_loss, router_aux_loss)
            return output_tensor, final_extra_loss
        return output_tensor

    def _tensor_unsort(self, expert_output, router_coeff, unsort_map):
        """calculate the final output by multiplying FeedForward's output and experts' weight in MoeFinalizeRouting"""
        # (dp, kN, h)bf16, (dp, N, k) fp32, (dp, N, k)int32
        # unsort output_tensor
        # (dp, N, k, h)bf16 <-- (dp, kN, h)bf16, (dp, N, k)int32
        output_tensor = self.gather_unsort(expert_output, unsort_map, 1)
        # (dp, N, k, 1) fp32<-- (dp, N, k) fp32
        router_coeff = self.reshape(router_coeff, (router_coeff.shape[0], router_coeff.shape[1],
                                                   router_coeff.shape[2], 1))
        # (dp, N, k, h)bf16  <-- (dp, N, k, h)bf16, (dp, N, k, 1)bf16
        output_tensor = self.mul_router_coeff(output_tensor, self.cast(router_coeff, output_tensor.dtype))
        # reduce sum # (dp, N, h)bf16 <-- (dp, N, k, h)bf16
        output_tensor = self.sum_router_coeff(output_tensor, 2)
        # (dp, N, h)bf16
        return output_tensor

    def _tensor_sort(self, input_tensor, expert_ids):
        """dispatch and get unsort map for routing"""

        # sort input_tensor
        expert_shape = expert_ids.shape
        # (dp, k, N) int32  <-- (dp, N, k) int32
        transposed_index = self.transpose_3d(expert_ids, (0, 2, 1))
        # (dp, kN ) int32 <-- (dp, k, N) int32
        reshaped_index = self.reshape(transposed_index, (self.dp, -1))
        # (dp, kN) fp32, (dp, kN) int32 <-- (dp, kN) fp32 <-- (dp, kN) int32
        sorted_index, sort_map = self.sort(self.cast(reshaped_index, mstype.float32))
        # (dp, kN) int32 <-- (dp, kN) int32, N int32
        inter_map = self.mod(sort_map, expert_shape[1])
        # (dp, kN, h) bf16  <-- (dp, N, h) bf16, (dp, kN) int32
        expert_input = self.gather_sort(input_tensor, inter_map, 1)

        # compute group list by cumsuming expert counts
        # (dp, kN, E) fp32 <-- (dp, kN ) int32
        expert_mask = self.onehot(reshaped_index, self.expert_dim, self.on_value, self.off_value)
        # (dp, E) int32 <-- (dp, kN, E) fp32
        expert_cnt = ops.sum(expert_mask, 1)

        # get unsort_map
        # _, (dp, kN) int32 <-- (dp, kN) fp32 <-- (dp, kN) int32
        _, unsort_map = self.sort(self.cast(sort_map, mstype.float32))
        # (dp, k, N)int32 <-- (dp, kN) int32
        unsort_map = self.reshape(unsort_map, (expert_shape[0], expert_shape[2], expert_shape[1]))
        # (dp, N, k)int32  <-- (dp, k, N)int32
        unsort_map = self.transpose_unsort_map(unsort_map, (0, 2, 1))
        # (dp, kN, h) bf16,  (dp, kN) fp32, (dp, E) int32, (dp, N, k)int32
        return expert_input, sorted_index, expert_cnt, unsort_map

    def _normalize(self, router_coeff_raw):
        # (dp, N, 1) <-- (dp, N, k)
        router_coeff_sum = self.reduce_sum_keep(router_coeff_raw, 2)
        # (dp, N, k) <-- (dp, N, k) (dp, N, 1)
        router_coeff = self.div_3d(router_coeff_raw, self.add_eps(router_coeff_sum, 1e-9))
        # (dp, N, k)
        return router_coeff

    def _expert_load_balancing(self, scores, top_indices, alpha):
        """Expert level load balance loss, which regularizes the load from local batch data on each
        expert to be balanced.
        float32 <-- (dp, N, E) fp32, (dp, N, k) int32, float32
        Please refer to DeepSeek-V2:
        A Strong, Economical, and Efficient Mixture-of-Experts Language Model, https://arxiv.org/abs/2405.04434
        """
        # p  (dp, E) <- (dp, N, E) fp32
        pi = self.reduce_mean_aux_3d(scores, 1)

        # f  (dp, Nk)int32, (dp, N, k)int32
        top_indices = self.reshape(top_indices, (top_indices.shape[0], -1))
        # (dp, kN, E)fp32 <-- (dp, kN)int32
        mask = self.onehot_aux(top_indices, self.expert_dim, self.on_value, self.off_value)
        # (dp, E) <- (dp, kN, E)
        fi = self.reduce_mean_aux_3d(mask, 1)

        # p*f  (dp) <- (dp, E)
        expert_load_loss = self.reduce_mean_aux_2d(self.mul_aux_2d(pi, fi))
        # alpha*E \sum_i^E (f_i * P_i)
        expert_load_loss = self.mul_noshard(expert_load_loss, alpha * self.expert_dim ** 2)
        return expert_load_loss

    def _topk(self, router_prob):
        # in default, normal topk will be used
        if self.moe_config.balance_via_topk_bias:
            _, expert_index = self.afb_topk(self.afb_add_topk_bias(router_prob, self.topk_bias),
                                            self.num_experts_chosen)
            expert_gate = self.gate_gather(router_prob, expert_index, 2)
            self._update_expert_load(expert_index)
        else:
            expert_gate, expert_index = self.topk(router_prob, self.num_experts_chosen)
        return expert_gate, expert_index

    def _update_expert_load(self, expert_index):
        expert_index = self.reshape(expert_index, (expert_index.shape[0], -1))
        expert_mask = self.onehot_2d(expert_index, self.expert_dim, self.on_value, self.off_value)
        expert_load_data = self.reduce_mean(expert_mask, 1)
        expert_load_data = self.afb_reduce_mean(expert_load_data, 0)
        self.assign_add(self.expert_load, expert_load_data)


def func_infer_dtype(*args):
    return args[0]


def func_infer_shape(*args):
    return args[0]


def ffn_forward_func(x, expert_id, counter, w1, w2, w3, ep_group, hidden_size, ep, use_fused_ops_permute=False):
    """
    Implements a forward pass functionality mainly used in processing input data x within an expert network
    (such as MoE, Mixture of Experts) through a series of operations including AllToAll communication, resorting,
    grouped matrix multiplication (GroupedMM), and its reverse operation.

    Parameters:
    - x (Tensor): Input tensor, typically with shape [B, S, h], where B is the batch size, S is the sequence length,
      and h is the hidden size.
    - expert_id (Tensor): Identifiers indicating which expert each input belongs to,
      should be compatible with the shape of x.
    - counter (Tensor): Counter tensor used to dynamically calculate parameters such as send_list,
      receive_list, group_list.
    - w1, w2, w3 (List[Tensor]): Lists of weights used respectively for the first, second,
      and third GroupedMatmul operations.
    - ep_group (Group): Communication group object defining the communication group used in AlltoAll
      and AlltoAllV operations.
    - hidden_size (int): The size of the hidden layer, used to determine target dimensions
      in certain reshape operations.
    - ep (int): Number of experts or partitions, affecting the reshape operation of the counter.

    Returns:
    - y (Tensor): Transformed output tensor after a series of operations, with the same shape as the input tensor x.
    """
    # prepare sl, rl, gl.
    # they should be calculated dynamically from the real expert_id and counters
    x_shape_origin = x.shape
    local_counter = ops.AlltoAll(split_count=ep, split_dim=-1, concat_dim=-2, group=ep_group)(counter)
    # [ep, E/ep] -->  [ep]
    send_list = ops.cast(counter.reshape(ep, -1).sum(dim=-1, keepdim=False), ms.int64)
    # [ep, E/ep]  --> [ep]
    receive_list = ops.cast(local_counter.reshape(ep, -1).sum(dim=-1, keepdim=False), ms.int64)
    # [ep, E/ep]  --> (E/ep) int64
    group_list = ops.cast(ops.cumsum(local_counter.reshape(ep, -1).sum(dim=-2, keepdim=False), 0), ms.int64)

    # 1.AllToAllv
    # x [B, S, h]
    x = ops.AlltoAllV(group=ep_group)(x.reshape(-1), send_list * hidden_size,
                                      receive_list * hidden_size).reshape(1, -1, hidden_size)
    # x [B, S]
    expert_id = ops.AlltoAllV(group=ep_group)(expert_id.astype(ms.float32).reshape(-1),
                                              send_list, receive_list).reshape(1, -1)

    # 2.Resort
    x, unresort_map = _ffn_resort(x, expert_id, use_fused_ops_permute)

    # 3.GroupedMM
    # squeeze x [B, S, h] -- > [B*S, h] where B=1
    x = x.reshape((-1, hidden_size))
    gate = GroupedMatmul(split_item=3, group_type=0)([x], [w1], None, None, None, None, None, group_list)[0]
    hidden = GroupedMatmul(split_item=3, group_type=0)([x], [w3], None, None, None, None, None, group_list)[0]
    # pylint: disable=W0212
    h = hidden * P._inner_ops.SiLU()(gate)
    y = GroupedMatmul(split_item=3, group_type=0)([h], [w2], None, None, None, None, None, group_list)[0]
    # unsqueeze y [B*S, h] -- > [B, S, h] where B=1
    y = y.reshape((1, -1, hidden_size))

    # 4.Unresort
    y = _ffn_unresort(y, unresort_map, use_fused_ops_permute)

    # 5.AllToAllv
    # x [B, S, h]
    y = ops.AlltoAllV(group=ep_group)(y.reshape(-1), receive_list * hidden_size,
                                      send_list * hidden_size).reshape(1, -1, hidden_size)
    y = y.reshape(x_shape_origin)
    return y


def _ffn_resort(x, expert_id, use_fused_ops_permute):
    """resort tensor x according to expert_id"""
    if use_fused_ops_permute:
        x_dtype_org = x.dtype
        x_shape_org = x.shape
        # permute only support bfloat16 for now
        x = ops.cast(x, ms.bfloat16)
        x = ops.reshape(x, (-1, x_shape_org[-1]))
        expert_id = ops.reshape(expert_id, (-1,))
        x, unsort_map = ops.moe_token_permute(x, expert_id.astype(ms.int32))
        x = ops.cast(x, x_dtype_org)
        x = ops.reshape(x, x_shape_org)
    else:
        _, sort_map = ops.sort(expert_id)
        _, unsort_map = ops.sort(sort_map.astype(ms.float32))
        x = ops.gather(x, sort_map, axis=1, batch_dims=1)

    return x, unsort_map


def _ffn_unresort(x, unsort_map, use_fused_ops_permute):
    """unresort tensor x according to unsort_map"""
    if use_fused_ops_permute:
        x_dtype_org = x.dtype
        x_shape_org = x.shape
        # permute only support bfloat16 for now
        x = ops.cast(x, ms.bfloat16)
        x = ops.reshape(x, (-1, x_shape_org[-1]))
        x = ops.moe_token_unpermute(x, unsort_map)
        x = ops.cast(x, x_dtype_org)
        x = ops.reshape(x, x_shape_org)
    else:
        x = ops.gather(x, unsort_map, axis=1, batch_dims=1)

    return x


class FFN(nn.Cell):
    """
    Initializes a Feed-Forward Network (FFN) cell, which is a fundamental building block in many
    neural network architectures, especially within transformer models.
    This initialization configures the FFN's dimensions,
    data types, expert configurations, and parallelism strategies.

    Parameters:
    - hidden_size (int): The number of features in the hidden layer.
    - intermediate_size (int): The size of the intermediate (or feed-forward) layer.
    - compute_dtype (dtype): The data type used for computation within the FFN.
    - moe_config (MoeConfig): Configuration object for Mixture of Experts (MoE).
    - parallel_config (ParallelConfig): Configuration object specifying the parallelism strategy for this FFN,
        such as model parallelism, data parallelism, or their combinations.
    - init_method_std (float, optional): Standard deviation for the initialization method used for the weights.
        Defaults to 0.01.
    """
    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 compute_dtype,
                 param_init_type,
                 moe_config,
                 parallel_config,
                 init_method_std=0.01):
        super(FFN, self).__init__()
        self.rank_id = get_rank()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.compute_dtype = compute_dtype
        self.param_init_type = param_init_type
        self.expert_num = moe_config.expert_num
        self.ep = parallel_config.expert_parallel
        self.dp = parallel_config.data_parallel * parallel_config.model_parallel
        self.outer_dp = self.dp // self.ep
        self.inner_dp = self.ep
        self.ep_group = self._get_ep_group_name()
        self.init_method_std = init_method_std
        self.use_fused_ops_permute = moe_config.use_fused_ops_permute

        # parameters
        self.w1 = Parameter(initializer(Normal(sigma=self.init_method_std, mean=0.0),
                                        [self.expert_num, self.hidden_size, self.intermediate_size],
                                        self.param_init_type), name='w1')
        self.w2 = Parameter(initializer(Normal(sigma=self.init_method_std, mean=0.0),
                                        [self.expert_num, self.intermediate_size, self.hidden_size],
                                        self.param_init_type), name='w2')
        self.w3 = Parameter(initializer(Normal(sigma=self.init_method_std, mean=0.0),
                                        [self.expert_num, self.hidden_size, self.intermediate_size],
                                        self.param_init_type), name='w3')

        # ops
        self.cast = P.Cast()
        self.enable_gmm_safe_tokens = moe_config.enable_gmm_safe_tokens
        self.safe_tokens = Tensor(np.zeros((self.dp, self.expert_num, self.hidden_size)), mstype.bfloat16)
        self.safe_tokens_expert_ids = Tensor(
            np.arange(self.dp * self.expert_num).reshape(self.dp, self.expert_num) % self.expert_num, ms.int32)

        self.op_sort_safe_tokens = Sort(1).shard(((self.dp, 1),))
        self.op_gather_safe_tokens = Gather(batch_dims=1).shard(((self.dp, 1, 1), (self.dp, 1)))
        self.op_add_safe_tokens = AddExt().shard(((self.dp, 1), ()))
        self.op_concat_safe_tokens = Concat(1).shard(((self.dp, 1, 1), (self.dp, 1, 1)))
        self.op_concat_safe_tokens_expert_ids = Concat(1).shard(((self.dp, 1), (self.dp, 1)))
        self.op_stridedslice_safe_tokens = StridedSlice().shard(((self.dp, 1, 1),))

        # hook_ffn_forward
        self.hook_ffn_forward = P.Morph(ffn_forward_func, func_infer_shape, func_infer_dtype).add_prim_attr(
            "self_define_shard", True)

        self.layout = Layout((self.outer_dp, self.inner_dp, 1, 1, 1), ("outer_dp", "inner_dp", "sp", "mp0", "mp1"))
        self.hook_ffn_forward.shard(
            in_strategy=(
                self.layout(("outer_dp", "inner_dp"), "sp", "mp0"),  # x [B, S, h]
                self.layout(("outer_dp", "inner_dp"), "sp"),  # expert_id [B, S]
                self.layout(("outer_dp", "inner_dp"), "sp"),  # conter [B, E]
                self.layout("inner_dp", "mp0", "mp1"),  # w1 [E, h, H]
                self.layout("inner_dp", "mp1", "mp0"),  # w2 [E, H, h]
                self.layout("inner_dp", "mp0", "mp1"),  # w3 [E, h, H]
            ),
            out_strategy=(
                self.layout(("outer_dp", "inner_dp"), "sp", "mp0"),  # output [B, S, h]
            )
        )

    def construct(self, x, expert_id, counter):
        """
        Defines the forward pass of the Feed-Forward Network (FFN) cell. This method processes the
        input tensor x through the FFN, handling padding for safety tokens,
        performing the core FFN operations, and then removing the safety tokens before returning
        the processed output.

        Parameters:
        - x (Tensor): Input tensor with shape [B, S, h], where B is the batch size,
          S is the sequence length, and h is the hidden size.
        - expert_id (Tensor): Tensor indicating which expert each input belongs to.
        - counter (Tensor): A tensor used for managing counts or indices related to the experts and data distribution.

        Returns:
        - x (Tensor): The output tensor after processing through the FFN, having the same shape as the input tensor x.
        """
        dtype = x.dtype
        w1 = self.cast(self.w1, dtype)
        w2 = self.cast(self.w2, dtype)
        w3 = self.cast(self.w3, dtype)
        x, expert_id, counter, unsort_map_safe_tokens = self._pad_safe_tokens(x, expert_id, counter)
        x = self.hook_ffn_forward(
            x, expert_id, counter, w1, w2, w3,
            self.ep_group, self.hidden_size, self.ep, self.use_fused_ops_permute
        )
        x = self._remove_safe_tokens(x, unsort_map_safe_tokens)
        return x

    def _pad_safe_tokens(self, x, expert_id, counter):
        """
        Prepares the input tensors by padding them with safety tokens.
        These tokens are designed to ensure safe computation,
        especially in distributed or parallel computing environments
        where data might need to be padded to fit certain dimensions.

        Parameters:
        - x (Tensor): Input tensor of shape [dp, kN, h], where dp is the data parallelism dimension,
          kN is a dimension related to the
          number of experts or similar partitioning, and h is the hidden size.
        - expert_id (Tensor): Tensor indicating the expert each part of the input belongs to,
          typically of shape [dp, kN].
        - counter (Tensor): A tensor used for managing counts or indices related to the experts and data distribution,
          usually of shape [dp, E].

        Returns:
        - x (Tensor): The input tensor x after padding with safety tokens.
        - expert_id (Tensor): The expert_id tensor after padding with safety tokens.
        - counter (Tensor): The counter tensor incremented by the number of added safety tokens.
        - unsort_map_safe_tokens (Tensor or int): If safety tokens are enabled, returns a tensor
          that maps from sorted to unsorted state
          necessary for removing the safety tokens later. If not enabled, returns 0.
        """
        # (dp, kN, h)bf16, (dp, kN)fp32, (dp, E)int32
        if self.enable_gmm_safe_tokens:
            x = self.op_concat_safe_tokens((self.safe_tokens.astype(x.dtype), x))
            expert_id = self.op_concat_safe_tokens_expert_ids(
                (self.safe_tokens_expert_ids.astype(expert_id.dtype), expert_id))
            counter = self.op_add_safe_tokens(counter, 1)
            # sort [safe_tokens, x] together
            # (dp, E+kN)fp32 <-- (dp, E+kN)fp32
            expert_id, sort_map_safe_tokens = self.op_sort_safe_tokens(expert_id.astype(ms.float32))
            _, unsort_map_safe_tokens = self.op_sort_safe_tokens(sort_map_safe_tokens.astype(ms.float32))
            x = self.op_gather_safe_tokens(x, sort_map_safe_tokens, 1)
            return x, expert_id, counter, unsort_map_safe_tokens
        return x, expert_id, counter, 0

    def _remove_safe_tokens(self, x, unsort_map_safe_tokens):
        """
        Removes the safety tokens that were previously added to the input tensor x.
        This process involves unsorting the tensor if it was sorted during the padding step
        and then slicing out the safety tokens to restore the original data structure.

        Parameters:
        - x (Tensor): Input tensor that may contain safety tokens at the beginning.
        - unsort_map_safe_tokens (Tensor or int): A tensor used to map from the sorted state back to the unsorted state.
          If safety tokens are not enabled, this will be 0.

        Returns:
        - x (Tensor): The output tensor after removing the safety tokens and restoring
          the original order if applicable.
        """
        if self.enable_gmm_safe_tokens:
            # unsort [safe_tokens, x] together
            x = self.op_gather_safe_tokens(x, unsort_map_safe_tokens, 1)
            # slice x
            x = self.op_stridedslice_safe_tokens(x,
                                                 (0, self.safe_tokens_expert_ids.shape[1], 0),
                                                 (x.shape[0], x.shape[1], x.shape[2]),
                                                 (1, 1, 1))
        return x

    def _get_ep_group_name(self):
        """
        Generates a unique group name for a set of ranks involved in expert partitioning (ep)
        and creates a communication group with this name.
        This method calculates a range of ranks based on the current rank id
        and the expert partition size, hashes this range to create a unique
        identifier, and then establishes a new communication group using this identifier.
        """
        rank_start = self.rank_id // self.ep * self.ep
        rand_end = self.rank_id // self.ep * self.ep + self.ep
        rank_list = [i for i in range(rank_start, rand_end)]

        rank_list_str = "-".join([str(i) for i in range(rank_start, rand_end)])
        hashed = hashlib.md5(rank_list_str.encode()).hexdigest()[:48]
        ep_group_name = str(hashed)
        create_group(ep_group_name, rank_list)
        return ep_group_name

    def shard(self, parallel_config):
        """
        Handles the sharding configuration for the model's parallelism settings.
        """
        logger.info(f"Using MoE v3, dp is {parallel_config.data_parallel}, mp is {parallel_config.model_parallel}")
        print("MoE v3 use customized parallel, not sharding.")
