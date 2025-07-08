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
import numpy as np

import mindspore.common.dtype as mstype
from mindspore import Tensor, nn, Parameter, mint, ops
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer
from mindspore.communication import get_rank, get_group_size

from mindformers.modules.layers import Linear
from mindformers.experimental.infer.core.activation import get_act_func
from mindformers.experimental.infer.core.layers import ColumnParallelLinear, RowParallelLinear
from mindformers.parallel_core.inference.tensor_parallel.mappings import (ReduceFromModelParallelRegion,
                                                                          GatherFromMoeTensorParallelRegionV2,
                                                                          GatherFromMoeTensorParallelRegion,
                                                                          GatherFromWorldParallelRegionV1,
                                                                          ReduceFromMoeTensorParallelRegion,
                                                                          ReduceScatterToMoeTensorParallelRegion,
                                                                          ReduceScatterToWorldParallelRegion,
                                                                          ReduceFromWorldParallelRegion,
                                                                          ScatterToMoeTensorParallelRegion,
                                                                          ScatterToWorldParallelRegion,
                                                                          GatherFromWorldParallelRegionV2)

# pylint: disable=C0412
from mindformers.parallel_core.inference.utils import get_tp_world_size, get_moe_ep_world_size, get_moe_tp_world_size

from mindformers.parallel_core.inference.parallel_state import get_moe_expert_parallel_group
from mindformers.version_control import is_910b
from mindformers.tools.utils import divide

try:
    from mindspore.ops.auto_generate import (MoeComputeExpertTokens,
                                             MoeFinalizeRouting,
                                             MoeGatingTopKSoftmax,
                                             MoeInitRouting,
                                             MoeInitRoutingV2,
                                             MoeTokenUnpermute,
                                             FusedAddTopKDiv,
                                             MoeDistributeDispatch,
                                             MoeDistributeCombine)
    MOE_FUSED_OP_VALID = True
except ImportError:
    MOE_FUSED_OP_VALID = False


dtype_map = {
    'float16': mstype.float32,
    'float32': mstype.float32,
    'bfloat16': mstype.bfloat16
}


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
                              has_bias=False, dtype=dtype_map.get(moe_config.router_dense_type))
        self.router = TopkRouter(self.expert_num)
        self.e_score_correction_bias = Parameter(initializer('zeros', (self.expert_num), mstype.float32),
                                                 requires_grad=False, parallel_optimizer=False)


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
        self.router_dense_type = dtype_map.get(moe_config.router_dense_type)
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
        x1 = mint.zeros((input_shape[0] // self.num_experts_chosen, input_shape[-1]),
                        dtype=input_tensor.dtype)  # (N, h)
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
        input_tensor = self.reshape(input_tensor,
                                    (-1, self.hidden_size))  # (bs, seq/1, h) -> (bs*seq, h) : use N replace bs*seq

        if self.use_fused_op:
            expert_val, expert_index, row_index = self.gating_topk_softmax(input_tensor)
            sorted_input_tensor, group_list, unsort_map = \
                self.tensor_sort_by_fused_op(input_tensor, expert_index, row_index)
        else:
            gating_logits = self.gating(self.cast(input_tensor,
                                                  self.router_dense_type)) # (N, h) * (h, E) -> (bs*seq, E)
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
        moe_output = self.tensor_moe_finalize_routing(expert_output,
                                                      expert_weight,
                                                      expert_index,
                                                      unsort_map)  # -> (N, h)

        output_tensor = self.reshape(moe_output, input_tensor_shape)  # (N, h) -> (bs, seq, h)
        return output_tensor


class SharedMLP(nn.Cell):
    r"""
        SharedMLP. Shared Expert for MoE .

        Args:
            config (Config): The configuration of Model.
        Inputs:
            - **input_tensor** (Tensor) - should be `[batch, seq_length, hidden_size].

        Outputs:
            - **output_tensor** (Tensor) - should be `[batch, seq_length, hidden_size].
    """

    def __init__(self, config, intermediate_size):
        super().__init__(config)
        self.config = config
        self.has_bias = self.config.mlp_has_bias
        self.hidden_size = self.config.hidden_size
        self.ffn_hidden_size = intermediate_size
        self.ffn_concat = self.config.ffn_concat
        if self.ffn_concat:
            self.w_gate_hidden = Linear(
                self.hidden_size,
                self.ffn_hidden_size * 2,
                has_bias=self.has_bias,
                transpose_b=True,
                param_init_type=self.config.param_init_dtype,
                compute_dtype=self.config.compute_dtype,
            )
        else:
            self.w1 = Linear(
                self.hidden_size,
                self.ffn_hidden_size,
                has_bias=self.has_bias,
                transpose_b=True,
                param_init_type=self.config.param_init_dtype,
                compute_dtype=self.config.compute_dtype,
            )
            self.w3 = Linear(
                self.hidden_size,
                self.ffn_hidden_size,
                has_bias=self.has_bias,
                transpose_b=True,
                param_init_type=self.config.param_init_dtype,
                compute_dtype=self.config.compute_dtype,
            )

        self.act_type = self.config.hidden_act
        self.act_func = get_act_func(self.act_type)

        self.w2 = Linear(
            self.ffn_hidden_size,
            self.hidden_size,
            has_bias=self.has_bias,
            transpose_b=True,
            param_init_type=self.config.param_init_dtype,
            compute_dtype=self.config.compute_dtype,
        )

        self.mul = ops.Mul()
        self.reshape = ops.Reshape()

    def construct(self, x):
        """ Construct function of mlp block. """
        if self.ffn_concat:
            gate_hidden_out = self.w_gate_hidden(x)  # dp,1 -> dp, mp  # dp,1 -> dp, mp
            gate, hidden = mint.split(gate_hidden_out,
                                      (self.ffn_hidden_size, self.ffn_hidden_size), -1)
        else:
            gate = self.w1(x)
            hidden = self.w3(x)
        gate = self.act_func(gate)
        hidden = mint.mul(hidden, gate)
        output = self.w2(hidden)
        return output


class SharedParallelMLP(nn.Cell):
    r"""
        SharedParallelMLP. Parallel Shared Expert for MoE .

        Args:
            config (Config): The configuration of Model.
        Inputs:
            - **input_tensor** (Tensor) - should be `[batch, seq_length, hidden_size].

        Outputs:
            - **output_tensor** (Tensor) - should be `[batch, seq_length, hidden_size].
    """

    def __init__(self, config, intermediate_size):
        super().__init__(config)
        self.config = config
        self.has_bias = self.config.mlp_has_bias
        self.hidden_size = self.config.hidden_size
        self.ffn_hidden_size = intermediate_size
        self.ffn_concat = self.config.ffn_concat
        tp_group_size = get_tp_world_size()
        self.ffn_hidden_size_per_partition = divide(self.ffn_hidden_size, tp_group_size)
        if self.ffn_concat:
            self.w_gate_hidden = ColumnParallelLinear(
                self.hidden_size,
                self.ffn_hidden_size * 2,
                config=self.config.parallel_config,
                bias=self.has_bias,
                transpose_b=True,
                gather_output=False,
                is_expert=False,
                param_init_type=self.config.param_init_dtype,
                compute_dtype=self.config.compute_dtype,
            )
        else:
            self.w1 = ColumnParallelLinear(
                self.hidden_size,
                self.ffn_hidden_size,
                config=self.config.parallel_config,
                bias=self.has_bias,
                transpose_b=True,
                gather_output=False,
                is_expert=False,
                param_init_type=self.config.param_init_dtype,
                compute_dtype=self.config.compute_dtype,
            )
            self.w3 = ColumnParallelLinear(
                self.hidden_size,
                self.ffn_hidden_size,
                config=self.config.parallel_config,
                bias=self.has_bias,
                transpose_b=True,
                gather_output=False,
                is_expert=False,
                param_init_type=self.config.param_init_dtype,
                compute_dtype=self.config.compute_dtype,
            )

        self.act_type = self.config.hidden_act
        self.act_func = get_act_func(self.act_type)

        self.w2 = RowParallelLinear(
            self.ffn_hidden_size,
            self.hidden_size,
            input_is_parallel=True,
            config=self.config.parallel_config,
            bias=self.has_bias,
            transpose_b=True,
            is_expert=True,
            param_init_type=self.config.param_init_dtype,
            compute_dtype=self.config.compute_dtype,
            delay_allreduce=True,
        )

        self.mul = ops.Mul()
        self.reshape = ops.Reshape()

    def construct(self, x):
        """ Construct function of mlp block. """
        if self.ffn_concat:
            gate_hidden_out = self.w_gate_hidden(x)  # dp,1 -> dp, mp  # dp,1 -> dp, mp
            gate, hidden = mint.split(gate_hidden_out,
                                      (self.ffn_hidden_size_per_partition, self.ffn_hidden_size_per_partition), -1)
        else:
            gate = self.w1(x)
            hidden = self.w3(x)
        gate = self.act_func(gate)
        hidden = mint.mul(hidden, gate)
        output = self.w2(hidden)
        return output


class WorldRegionSharedParallelMLP(SharedParallelMLP):
    r"""
        WorldRegionSharedParallelMLP. Parallel MoE with global world region.

        Args:
            config (Config): The configuration of Model.
        Inputs:
            - **input_tensor** (Tensor) - should be `[batch, seq_length, hidden_size].

        Outputs:
            - **output_tensor** (Tensor) - should be `[batch, seq_length, hidden_size].
    """
    def __init__(self, config, intermediate_size):
        super(WorldRegionSharedParallelMLP, self).__init__(config=config, intermediate_size=intermediate_size)
        world_group_size = get_group_size()
        self.ffn_hidden_size_per_partition = divide(self.ffn_hidden_size, world_group_size)
        if self.ffn_concat:
            self.w_gate_hidden = ColumnParallelLinearWorldRegion(
                self.hidden_size,
                self.ffn_hidden_size * 2,
                config=self.config.parallel_config,
                bias=self.has_bias,
                transpose_b=True,
                gather_output=False,
                is_expert=False,
                param_init_type=self.config.param_init_dtype,
                compute_dtype=self.config.compute_dtype,
            )
        else:
            self.w1 = ColumnParallelLinearWorldRegion(
                self.hidden_size,
                self.ffn_hidden_size,
                config=self.config.parallel_config,
                bias=self.has_bias,
                transpose_b=True,
                gather_output=False,
                is_expert=False,
                param_init_type=self.config.param_init_dtype,
                compute_dtype=self.config.compute_dtype,
            )
            self.w3 = ColumnParallelLinearWorldRegion(
                self.hidden_size,
                self.ffn_hidden_size,
                config=self.config.parallel_config,
                bias=self.has_bias,
                transpose_b=True,
                gather_output=False,
                is_expert=False,
                param_init_type=self.config.param_init_dtype,
                compute_dtype=self.config.compute_dtype,
            )

        self.w2 = RowParallelLinearWorldRegion(
            self.ffn_hidden_size,
            self.hidden_size,
            input_is_parallel=True,
            config=self.config.parallel_config,
            bias=self.has_bias,
            transpose_b=True,
            is_expert=True,
            param_init_type=self.config.param_init_dtype,
            compute_dtype=self.config.compute_dtype,
            delay_allreduce=True,
        )


class ColumnParallelLinearWorldRegion(ColumnParallelLinear):
    r"""
        The dense layer with weight sliced on second dimension by global world region parallel size.
        This layer implements the operation as:

        .. math::
            \text{outputs} = \text{inputs} * \text{weight} + \text{bias},

        where :math:`inputs` is the input tensors, :math:`\text{weight}` is a weight matrix created by the layer,
        and :math:`\text{bias}` is a bias vector created by the layer (only if has_bias is True).

        Args:
            input_size (int): The number of channels in the input space.
            output_size (int): The number of channels in the output space.
            config (dict): Parallel configuration.
            weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter.
            The values of str refer to the function `initializer`. Default: 'normal'.
            bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The values
                of str refer to the function `initializer`. Default: 'zeros'.
            bias (bool): Specifies whether the layer uses a bias vector. Default: True.
            gather_output (bool): Specifies whether gather the output on each tensor parallel rank. Default: False.
            skip_weight_param_allocation (bool): Specifies whether skip the initialization of weight parameter.
                When set True, an weight tensor should be passed to construct function. Default: False.
            is_expert (bool): Specifies whether this linear layer is an expert. Default: False.
            transpose_b (bool): Specifies whether the weight parameter will be initialized as a transposed shape.
            param_init_type (dtype.Number): The parameter initialization type. Default: mstype.float32.
            compute_dtype (dtype.Number): The computation type. Default: mstype.float16.
            expert_num (int): The number of expert. Default: 1.

        Inputs:
            - **x** (Tensor) - Tensor of shape :math:`(*, in\_channels)`. The `input_size` in `Args` should be equal
              to :math:`in\_channels` in `Inputs`.

        Outputs:
            Tensor of shape :math:`(*, out\_channels)`.

        Raises:
            ValueError: `skip_weight_param_allocation=True` but weight_tensor is not passed to construct function.

        Supported Platforms:
            ``Ascend``
    """
    def __init__(
            self,
            input_size,
            output_size,
            config,
            bias=False,
            is_expert=False,
            param_init_type=mstype.float32,
            compute_dtype=mstype.float16,
            expert_num=1,
            weight_init="normal",
            bias_init="zeros",
            **kwargs
    ):
        super(ColumnParallelLinearWorldRegion, self).__init__(
            input_size=input_size,
            output_size=output_size,
            config=config,
            bias=bias,
            is_expert=is_expert,
            expert_num=expert_num,
            param_init_type=param_init_type,
            compute_dtype=compute_dtype,
            weight_init=weight_init,
            bias_init=bias_init,
            **kwargs
        )
        self.tensor_parallel_group_size = get_group_size()
        self.output_size_per_partition = divide(output_size, self.tensor_parallel_group_size)

        weight_shape = (self.output_size_per_partition, self.input_size) if self.transpose_b else (
            self.input_size, self.output_size_per_partition)
        self.weight = Parameter(initializer(weight_init, weight_shape, param_init_type), name="weight")

        if bias:
            bias_shape = (self.output_size_per_partition,)
            self.bias = Parameter(initializer(bias_init, bias_shape, param_init_type), name="bias")
            self.bias_add = P.Add()
        self.gather_from_mp_region = GatherFromWorldParallelRegionV1()
        if self.sequence_parallel:
            self.gather_from_sp_region = GatherFromWorldParallelRegionV2()


class RowParallelLinearWorldRegion(RowParallelLinear):
    r"""
        The dense layer with weight sliced on first dimension by global world region parallel size.
        This layer implements the operation as:

        .. math::
            \text{outputs} = \text{inputs} * \text{weight} + \text{bias},

        where :math:`inputs` is the input tensors, :math:`\text{weight}` is a weight matrix created by the layer,
        and :math:`\text{bias}` is a bias vector created by the layer (only if has_bias is True).

        Args:
            input_size (int): The number of channels in the input space.
            output_size (int): The number of channels in the output space.
            config (dict): Parallel configuration.
            input_is_parallel (bool): Specifies whether the input tensor has already been sliced on last dimension.
            weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter.
                The values of str refer to the function `initializer`. Default: 'normal'.
            bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The values
                of str refer to the function `initializer`. Default: 'zeros'.
            bias (bool): Specifies whether the layer uses a bias vector. Default: True.
            skip_bias_add (bool): Specifies whether the layer doesn't need to add bias. Default: False.
            is_expert (bool): Specifies whether this linear layer is an expert. Default: False.
            transpose_b (bool): Specifies whether the weight parameter will be initialized as a transposed shape.
            param_init_type (dtype.Number): The parameter initialization type. Default: mstype.float32.
            compute_dtype (dtype.Number): The computation type. Default: mstype.float16.
            expert_num (int): The number of expert. Default: 1.

        Inputs:
            - **x** (Tensor) - Tensor of shape :math:`(*, in\_channels)`. The `input_size` in `Args` should be equal
              to :math:`in\_channels` in `Inputs`.

        Outputs:
            Tensor of shape :math:`(*, out\_channels)`.

        Supported Platforms:
            ``Ascend``
    """
    def __init__(
            self,
            input_size,
            output_size,
            config,
            input_is_parallel,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            param_init_type=mstype.float32,
            compute_dtype=mstype.float16,
            expert_num=1,
            weight_init="normal",
            bias_init="zeros",
            delay_allreduce=False,
            **kwargs
    ):
        super(RowParallelLinearWorldRegion, self).__init__(
            input_size=input_size,
            output_size=output_size,
            config=config,
            input_is_parallel=input_is_parallel,
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            expert_num=expert_num,
            delay_allreduce=delay_allreduce,
            param_init_type=param_init_type,
            compute_dtype=compute_dtype,
            weight_init=weight_init,
            bias_init=bias_init,
            **kwargs
        )
        self.tensor_parallel_group_size = get_group_size()
        self.input_size_per_partition = divide(input_size, self.tensor_parallel_group_size)

        # weight
        weight_shape = (self.output_size, self.input_size_per_partition) if self.transpose_b else (
            self.input_size_per_partition, self.output_size)
        self.weight = Parameter(initializer(weight_init, weight_shape, param_init_type), name="weight")

        # bias
        if self.has_bias and not self.skip_bias_add:
            bias_shape = (self.output_size,)
            self.bias = Parameter(initializer(super().bias_init, bias_shape, super().param_init_type), name="bias")
            self.bias_add = P.Add()

        self.reduce_from_mp_region = ReduceFromWorldParallelRegion()
        if not self.input_is_parallel:
            self.scatter_to_mp_region = ScatterToWorldParallelRegion()
        if self.sequence_parallel:
            self.reduce_scatter_to_sp_region = ReduceScatterToWorldParallelRegion()


class ColumnParallelGroupLinear(ColumnParallelLinear):
    r"""
        The group linear layer with weight sliced on second dimension by tensor parallel size.
        This layer implements the operation as:

        .. math::
            \text{outputs} = \text{inputs} * \text{weight} + \text{bias},

        where :math:`inputs` is the input tensors, :math:`\text{weight}` is a weight matrix created by the layer,
        and :math:`\text{bias}` is a bias vector created by the layer (only if has_bias is True).

        Args:
            input_size (int): The number of channels in the input space.
            output_size (int): The number of channels in the output space.
            config (dict): Parallel configuration.
            weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter.
                The values of str refer to the function `initializer`. Default: 'normal'.
            bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The values
                of str refer to the function `initializer`. Default: 'zeros'.
            bias (bool): Specifies whether the layer uses a bias vector. Default: True.
            gather_output (bool): Specifies whether gather the output on each tensor parallel rank. Default: False.
            skip_weight_param_allocation (bool): Specifies whether skip the initialization of weight parameter.
                When set True, an weight tensor should be passed to construct function. Default: False.
            is_expert (bool): Specifies whether this linear layer is an expert. Default: False.
            transpose_b (bool): Specifies whether the weight parameter will be initialized as a transposed shape.
            param_init_type (dtype.Number): The parameter initialization type. Default: mstype.float32.
            compute_dtype (dtype.Number): The computation type. Default: mstype.float16.
            expert_num (int): The number of expert. Default: 1.

        Inputs:
            - **x** (Tensor) - Tensor of shape :math:`(*, in\_channels)`. The `input_size` in `Args` should be equal
              to :math:`in\_channels` in `Inputs`.

        Outputs:
            Tensor of shape :math:`(*, out\_channels)`.

        Raises:
            ValueError: `skip_weight_param_allocation=True` but weight_tensor is not passed to construct function.

        Supported Platforms:
            ``Ascend``
    """
    def __init__(
            self,
            input_size,
            output_size,
            config,
            bias=False,
            is_expert=False,
            param_init_type=mstype.float32,
            compute_dtype=mstype.float16,
            expert_num=1,
            weight_init="normal",
            bias_init="zeros",
            **kwargs
    ):
        super(ColumnParallelGroupLinear, self).__init__(
            input_size=input_size,
            output_size=output_size,
            config=config,
            bias=bias,
            is_expert=is_expert,
            expert_num=expert_num,
            param_init_type=param_init_type,
            compute_dtype=compute_dtype,
            weight_init=weight_init,
            bias_init=bias_init,
            **kwargs
        )
        # moe tp ep
        self.moe_tp_size = get_moe_tp_world_size()
        self.output_size_per_partition = divide(output_size, self.moe_tp_size)
        self.moe_ep_size = get_moe_ep_world_size()
        self.ep_size_per_partition = divide(expert_num, self.moe_ep_size)

        weight_shape = (self.ep_size_per_partition, self.input_size, self.output_size_per_partition)
        self.weight = Parameter(initializer(weight_init, weight_shape, param_init_type), name="weight")

        if bias:
            bias_shape = (self.ep_size_per_partition, self.output_size_per_partition)
            self.bias = Parameter(initializer(bias_init, bias_shape, param_init_type), name="bias")
            self.bias_add = P.Add()
        self.gather_from_mp_region = GatherFromMoeTensorParallelRegion()
        if self.sequence_parallel:
            self.gather_from_sp_region = GatherFromMoeTensorParallelRegionV2()


class RowParallelGroupLinear(RowParallelLinear):
    r"""
        The group linear layer with weight sliced on first dimension by tensor parallel size.
        This layer implements the operation as:

        .. math::
            \text{outputs} = \text{inputs} * \text{weight} + \text{bias},

        where :math:`inputs` is the input tensors, :math:`\text{weight}` is a weight matrix created by the layer,
        and :math:`\text{bias}` is a bias vector created by the layer (only if has_bias is True).

        Args:
            input_size (int): The number of channels in the input space.
            output_size (int): The number of channels in the output space.
            config (dict): Parallel configuration.
            input_is_parallel (bool): Specifies whether the input tensor has already been sliced on last dimension.
            weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter.
                The values of str refer to the function `initializer`. Default: 'normal'.
            bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The values
                of str refer to the function `initializer`. Default: 'zeros'.
            bias (bool): Specifies whether the layer uses a bias vector. Default: True.
            skip_bias_add (bool): Specifies whether the layer doesn't need to add bias. Default: False.
            is_expert (bool): Specifies whether this linear layer is an expert. Default: False.
            transpose_b (bool): Specifies whether the weight parameter will be initialized as a transposed shape.
            param_init_type (dtype.Number): The parameter initialization type. Default: mstype.float32.
            compute_dtype (dtype.Number): The computation type. Default: mstype.float16.
            expert_num (int): The number of expert. Default: 1.

        Inputs:
            - **x** (Tensor) - Tensor of shape :math:`(*, in\_channels)`. The `input_size` in `Args` should be equal
              to :math:`in\_channels` in `Inputs`.

        Outputs:
            Tensor of shape :math:`(*, out\_channels)`.

        Supported Platforms:
            ``Ascend``
    """
    def __init__(
            self,
            input_size,
            output_size,
            config,
            input_is_parallel,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            param_init_type=mstype.float32,
            compute_dtype=mstype.float16,
            expert_num=1,
            weight_init="normal",
            bias_init="zeros",
            delay_allreduce=False,
            **kwargs
    ):
        super(RowParallelGroupLinear, self).__init__(
            input_size=input_size,
            output_size=output_size,
            config=config,
            input_is_parallel=input_is_parallel,
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            expert_num=expert_num,
            delay_allreduce=delay_allreduce,
            param_init_type=param_init_type,
            compute_dtype=compute_dtype,
            weight_init=weight_init,
            bias_init=bias_init,
            **kwargs
        )
        # tp ep
        self.moe_tp_size = get_moe_tp_world_size()
        self.input_size_per_partition = divide(input_size, self.moe_tp_size)
        self.moe_ep_size = get_moe_ep_world_size()
        self.ep_size_per_partition = divide(expert_num, self.moe_ep_size)

        # weight
        weight_shape = (self.ep_size_per_partition, self.input_size_per_partition, self.output_size)
        self.weight = Parameter(initializer(weight_init, weight_shape, param_init_type), name="weight")

        # bias
        if self.has_bias and not self.skip_bias_add:
            bias_shape = (self.ep_size_per_partition, self.output_size)
            self.bias = Parameter(initializer(super().bias_init, bias_shape, super().param_init_type), name="bias")
            self.bias_add = P.Add()

        # moe_tp
        self.reduce_from_moe_tp_region = ReduceFromMoeTensorParallelRegion()
        self.reduce_from_mp_region = self.reduce_from_moe_tp_region
        if not self.input_is_parallel:
            self.scatter_to_mp_region = ScatterToMoeTensorParallelRegion()
        if self.sequence_parallel:
            self.reduce_scatter_to_sp_region = ReduceScatterToMoeTensorParallelRegion()


class RoutedParallelMLP(nn.Cell):
    r"""
        RoutedParallelMLP. Routing each tokens to the topk expert and calculating the final output.

        Args:
            config (Config): The configuration of Model.
        Inputs:
            - **input_tensor** (Tensor) - should be `[batch, seq_length, hidden_size].

        Outputs:
            - **output_tensor** (Tensor) - should be `[batch, seq_length, hidden_size].
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.has_bias = self.config.mlp_has_bias
        self.hidden_size = self.config.hidden_size
        self.ffn_hidden_size = self.config.moe_config.moe_intermediate_size
        self.cast = P.Cast()
        self.act_type = self.config.hidden_act
        self.act_func = get_act_func(self.act_type)
        self.ffn_concat = self.config.ffn_concat
        self.moe_tp_size = get_moe_tp_world_size()
        self.ffn_hidden_size_per_partition = divide(self.ffn_hidden_size, self.moe_tp_size)
        if self.ffn_concat:
            self.w_gate_hidden = ColumnParallelGroupLinear(
                self.hidden_size,
                self.ffn_hidden_size * 2,
                config=self.config.parallel_config,
                bias=self.has_bias,
                is_expert=True,
                transpose_b=True,
                expert_num=self.config.moe_config.expert_num,
                param_init_type=self.config.param_init_dtype,
                compute_dtype=self.config.compute_dtype,
            )
        else:
            self.w1 = ColumnParallelGroupLinear(
                self.hidden_size,
                self.ffn_hidden_size,
                config=self.config.parallel_config,
                bias=self.has_bias,
                transpose_b=True,
                gather_output=False,
                param_init_type=self.config.param_init_dtype,
                compute_dtype=self.config.compute_dtype,
                is_expert=True,
                expert_num=self.config.moe_config.expert_num,
            )
            self.w3 = ColumnParallelGroupLinear(
                self.hidden_size,
                self.ffn_hidden_size,
                config=self.config.parallel_config,
                bias=self.has_bias,
                transpose_b=True,
                gather_output=False,
                param_init_type=self.config.param_init_dtype,
                compute_dtype=self.config.compute_dtype,
                is_expert=True,
                expert_num=self.config.moe_config.expert_num,
            )
        self.w2 = RowParallelGroupLinear(
            self.ffn_hidden_size,
            self.hidden_size,
            input_is_parallel=True,
            config=self.config.parallel_config,
            bias=self.has_bias,
            skip_bias_add=True,
            transpose_b=True,
            param_init_type=self.config.param_init_dtype,
            compute_dtype=self.config.compute_dtype,
            is_expert=True,
            expert_num=self.config.moe_config.expert_num,
            delay_allreduce=True,
        )

    def construct(self, x, group_list=None):
        """Forward process of the FeedForward"""
        if self.ffn_concat:
            gate_hidden_out = self.w_gate_hidden(x, group_list=group_list)  # dp,1 -> dp, mp  # dp,1 -> dp, mp
            gate, hidden = mint.split(gate_hidden_out,
                                      (self.ffn_hidden_size_per_partition, self.ffn_hidden_size_per_partition), -1)
        else:
            gate = self.w1(x, group_list=group_list)
            hidden = self.w3(x, group_list=group_list)
        gate = self.act_func(gate)
        hidden = mint.mul(hidden, gate)
        output = self.w2(hidden, group_list)
        return output


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


class ParallelMoEV2(nn.Cell):
    r"""
        ParallelMoEV2. Routing each tokens to the topk expert and calculating the final output.

        Args:
            ffn (Cell): The FeedForward Module.
            hidden_size (int): The hidden size of each token.
            moe_config (MoEConfig): The configuration of MoE (Mixture of Expert).
        Inputs:
            - **input_tensor** (Tensor) - should be `[batch, seq_length, hidden_size].

        Outputs:
            - **output_tensor** (Tensor) - should be `[batch, seq_length, hidden_size].
    """

    def __init__(self,
                 ffn,
                 hidden_size,
                 moe_config,
                 is_reduce_moe_output=True):
        super(ParallelMoEV2, self).__init__()
        self.hidden_size = hidden_size
        self.moe_config = moe_config
        self.is_reduce_moe_output = is_reduce_moe_output
        self.expert_num = moe_config.expert_num
        self.num_experts_chosen = moe_config.num_experts_chosen
        self.router_dense_type = dtype_map.get(moe_config.router_dense_type)
        self.topk_group = moe_config.topk_group
        self.n_group = moe_config.n_group

        self.ffn = ffn
        self.router = Router(hidden_size=self.hidden_size, moe_config=moe_config)
        self.gating = self.router.dense

        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.cast = P.Cast()
        self.add = P.Add()
        self.div = P.Div()
        self.mul = P.Mul()
        self.gather = P.Gather()

        self.idx_arr = Tensor(np.arange(1024, dtype=np.int32))
        self.group_topk_inner = 2
        self.group_topk = GroupTopkCell()

        self.moe_token_unpermute = MoeTokenUnpermute()
        self.moe_init_routing_v2 = MoeInitRoutingV2()
        self.reduce_from_mp_region = ReduceFromModelParallelRegion()
        self.fused_valid = MOE_FUSED_OP_VALID
        if self.fused_valid:
            self.fused_add_topk_div = FusedAddTopKDiv()

    def construct(self, input_tensor):
        """forward process"""
        gating_logits = self.gating(self.cast(input_tensor, self.router_dense_type))
        if self.fused_valid:
            gating_logits = self.cast(gating_logits, mstype.float32)
            expert_weight, expert_index = \
                self.fused_add_topk_div(
                    gating_logits,
                    self.router.e_score_correction_bias,
                    self.num_experts_chosen,
                    self.topk_group,
                    self.group_topk_inner,
                    self.num_experts_chosen,
                    0,
                    True,
                    self.moe_config.routed_scaling_factor)
            expert_weight = expert_weight.astype(input_tensor.dtype)
        else:
            score = mint.sigmoid(gating_logits)
            origin_score = score

            # bias
            score = score + self.router.e_score_correction_bias
            # n_group
            score = self.group_topk(score.astype(mstype.bfloat16), self.idx_arr, self.n_group,
                                    self.topk_group, self.group_topk_inner)
            # topk
            expert_index = mint.topk(score, self.num_experts_chosen, dim=-1)[1]
            expert_index = self.cast(expert_index, mstype.int32)

            weight = origin_score.gather(expert_index, 1, 1)
            expert_weight = self.div(weight, mint.sum(weight, -1, True))
            expert_weight = self.mul(self.moe_config.routed_scaling_factor, expert_weight).astype(input_tensor.dtype)

        sorted_input_tensor, unsort_map, group_list, _ = \
            self.moe_init_routing_v2(
                input_tensor,
                expert_index,
                active_num=0,
                expert_capacity=0,
                expert_num=self.expert_num,
                drop_pad_mode=0,
                expert_tokens_count_or_cumsum_flag=2,
                expert_tokens_before_capacity_flag=True)
        group_list = self.cast(group_list, mstype.int64)

        expert_output = self.ffn(sorted_input_tensor, group_list)

        moe_output = self.moe_token_unpermute(permuted_tokens=expert_output,
                                              sorted_indices=unsort_map,
                                              probs=expert_weight,
                                              padded_mode=False,
                                              restore_shape=None)
        return moe_output


class ExpertParallelMoE(nn.Cell):
    r"""
        ExpertParallelMoE. Routing each tokens to the topk expert and calculating the final output.

        Args:
            ffn (Cell): The FeedForward Module.
            hidden_size (int): The hidden size of each token.
            moe_config (MoEConfig): The configuration of MoE (Mixture of Expert).
            compute_dtype(dtype.Number): The computation type of the layer.
        Inputs:
            - **input_tensor** (Tensor) - should be `[batch, seq_length, hidden_size].

        Outputs:
            - **output_tensor** (Tensor) - should be `[batch, seq_length, hidden_size].
    """

    def __init__(self,
                 ffn,
                 hidden_size,
                 moe_config,
                 use_alltoall,
                 compute_dtype):
        super(ExpertParallelMoE, self).__init__()
        self.compute_dtype = compute_dtype
        self.hidden_size = hidden_size
        self.moe_config = moe_config
        self.expert_num = moe_config.expert_num
        self.num_experts_chosen = moe_config.num_experts_chosen
        self.router_dense_type = dtype_map.get(moe_config.router_dense_type)
        self.topk_group = moe_config.topk_group

        self.ffn = ffn
        self.router = Router(hidden_size=self.hidden_size, moe_config=moe_config)
        self.gating = self.router.dense
        self.cast = P.Cast()
        self.group_topk_inner = 2

        self.moe_token_unpermute = MoeTokenUnpermute()
        self.moe_init_routing_v2 = MoeInitRoutingV2()
        self.fused_add_topk_div = FusedAddTopKDiv()
        self.dummy_token = mint.zeros((1, self.hidden_size), dtype=self.compute_dtype)
        self.fill_value = Tensor(0, self.compute_dtype)

        self.moe_tp_size = get_moe_tp_world_size()
        self.moe_ep_size = get_moe_ep_world_size()
        self.use_alltoall = use_alltoall

        self.moe_ep_group = get_moe_expert_parallel_group()
        self.dispatch = MoeDistributeDispatch() # only support in 910b and 910_A3
        self.combine = MoeDistributeCombine()   # only support in 910b and 910_A3
        self.dispatch_tp_world_size = 0 if is_910b() else 1     # 910b:0, 910_A3:1
        self.dispatch_shared_expert_num = 0 if is_910b() else 1 # 910b:0, 910_A3:1
        self.max_bs = 256 if is_910b() else 512 # max b*s in single npu
        self.dispatch_global_max_bs = min(moe_config.dispatch_global_max_bs, self.max_bs)

        self.local_ep_num = self.expert_num // self.moe_ep_size
        self.ep_rank_index = get_rank() // self.moe_tp_size
        self.in_start_expert_idx = self.ep_rank_index * self.local_ep_num
        self.group_list_index = Tensor([0,], mstype.int32)

        if self.moe_ep_size > 1 and not self.use_alltoall:
            bias_idx = [idx for idx in range(self.expert_num)]
            self.bias_idx = bias_idx[self.in_start_expert_idx:] + bias_idx[:self.in_start_expert_idx]
            self.router.e_score_correction_bias = self.router.e_score_correction_bias[self.bias_idx]

    def moe_with_allgather(self, input_tensor, expert_weight, expert_index):
        """moe feed forward with allgather."""
        local_expert_index = self.cast(expert_index, mstype.int32)
        expert_weight_mask = expert_index >= self.local_ep_num
        expert_weight = ops.masked_fill(expert_weight, expert_weight_mask, self.fill_value)

        sorted_input_tensor, unsort_map, group_list, _ = \
            self.moe_init_routing_v2(
                input_tensor,
                local_expert_index,
                active_num=0,
                expert_capacity=0,
                expert_num=self.expert_num,
                drop_pad_mode=0,
                expert_tokens_count_or_cumsum_flag=2,
                expert_tokens_before_capacity_flag=True)

        #Avoid the problem of poor performance of the split(int32) operator
        group_list = group_list.reshape(self.moe_ep_size, -1)
        group_list = mint.index_select(group_list, 0, self.group_list_index)
        group_list = group_list.reshape(-1)

        group_list = self.cast(group_list, mstype.int64)
        expert_output = self.ffn(sorted_input_tensor, group_list)
        expert_output = mint.nan_to_num(expert_output, 0, 0, 0)
        moe_output = self.moe_token_unpermute(permuted_tokens=expert_output,
                                              sorted_indices=unsort_map,
                                              probs=expert_weight,
                                              padded_mode=False,
                                              restore_shape=None)
        return moe_output

    def moe_with_alltoallv(self, input_tensor, expert_weight, expert_index):
        """small ops, moe feed forward with alltoall and alltoallv."""
        sorted_input_tensor, unsort_map, group_list, _ = \
            self.moe_init_routing_v2(
                input_tensor,
                expert_index,
                active_num=0,
                expert_capacity=0,
                expert_num=self.expert_num,
                drop_pad_mode=0,
                expert_tokens_count_or_cumsum_flag=2,
                expert_tokens_before_capacity_flag=True)

        group_list = group_list.reshape(1, -1).astype(mstype.float32)
        local_counter = ops.AlltoAll(split_count=self.moe_ep_size, split_dim=-1, concat_dim=-2)(group_list)
        send_list = ops.cast(group_list.reshape(self.moe_ep_size, -1).sum(dim=-1, keepdim=False), mstype.int64)
        recv_list = ops.cast(local_counter.reshape(self.moe_ep_size, -1).sum(dim=-1, keepdim=False), mstype.int64)
        local_grouplist = ops.cast(local_counter.reshape(self.moe_ep_size, -1).sum(dim=-2, keepdim=False), mstype.int64)

        recv_num_token = recv_list.sum()

        recv_token_x = ops.AlltoAllV(block_size=self.hidden_size)(sorted_input_tensor.reshape(-1), send_list, recv_list)
        expert_index_1d, _ = ops.sort(expert_index.astype(mstype.float32).reshape(-1))
        expert_id = ops.AlltoAllV()(expert_index_1d, send_list, recv_list)

        y = self.dummy_token
        if recv_num_token != 0:
            x = recv_token_x.reshape(-1, self.hidden_size)

            _, inner_sort_map = ops.sort(expert_id)
            _, inner_unsort_map = ops.sort(inner_sort_map.astype(mstype.float32))
            resort_x = ops.gather(x, inner_sort_map, axis=0)

            ffn_res = self.ffn(resort_x, local_grouplist)
            y = ops.gather(ffn_res, inner_unsort_map, axis=0)

        yout = ops.AlltoAllV(block_size=self.hidden_size)(y.reshape(-1), recv_list, send_list)
        expert_output = yout.reshape((-1, self.hidden_size))

        moe_output = self.moe_token_unpermute(permuted_tokens=expert_output,
                                              sorted_indices=unsort_map,
                                              probs=expert_weight,
                                              padded_mode=False,
                                              restore_shape=None)

        return moe_output

    def moe_with_dispatch_combine(self, input_tensor, expert_weight, expert_index):
        """fused ops, moe feed forward with dispatch and combine."""
        # Dispatch
        expand_x, _, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts, _ = self.dispatch(
            x=input_tensor,
            expert_ids=expert_index,
            ep_world_size=self.moe_ep_size,
            ep_rank_id=self.ep_rank_index,
            moe_expert_num=self.expert_num,
            group_ep=self.moe_ep_group,
            tp_world_size=self.dispatch_tp_world_size,
            shared_expert_num=self.dispatch_shared_expert_num,
            global_bs=self.dispatch_global_max_bs*self.moe_ep_size,
            expert_token_nums_type=1)

        # GroupMamtul
        ffn_res = self.ffn(expand_x, expert_token_nums)

        # Combine
        moe_output = self.combine(
            expand_x=ffn_res,
            expert_ids=expert_index,
            expand_idx=expand_idx,
            ep_send_counts=ep_recv_counts,
            expert_scales=expert_weight,
            ep_world_size=self.moe_ep_size,
            ep_rank_id=self.ep_rank_index,
            moe_expert_num=self.expert_num,
            tp_send_counts=tp_recv_counts,
            group_ep=self.moe_ep_group,
            tp_world_size=self.dispatch_tp_world_size,
            shared_expert_num=self.dispatch_shared_expert_num,
            global_bs=self.dispatch_global_max_bs*self.moe_ep_size)

        return moe_output

    def construct(self, input_tensor):
        """forward process"""
        # Gating
        gating_logits = self.gating(self.cast(input_tensor, self.router_dense_type))
        gating_logits = self.cast(gating_logits, mstype.float32)
        expert_weight, expert_index = \
            self.fused_add_topk_div(
                gating_logits,
                self.router.e_score_correction_bias,
                self.num_experts_chosen,
                self.topk_group,
                self.group_topk_inner,
                self.num_experts_chosen,
                0,
                True,
                self.moe_config.routed_scaling_factor)

        # AllGather
        if not self.use_alltoall:
            expert_weight = expert_weight.astype(input_tensor.dtype)
            return self.moe_with_allgather(input_tensor, expert_weight, expert_index)

        return self.moe_with_dispatch_combine(input_tensor, expert_weight, expert_index)
