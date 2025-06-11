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
"""Linear units for tensor parallelism"""

__all__ = [
    "ColumnParallelLinear",
    "QKVParallelLinear",
    "RowParallelLinear",
    "VocabParallelEmbedding",
    "ReplicatedLinear"
]

from typing import Callable, Optional, List

import mindspore.common.dtype as mstype
import mindspore.ops.operations as P
from mindspore import Parameter, Tensor, mint, nn, ops
from mindspore.common.initializer import initializer

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.inference.tensor_parallel.mappings import (GatherFromModelParallelRegion,
                                                                          ReduceFromModelParallelRegion,
                                                                          ReduceScatterToSequenceParallelRegion,
                                                                          ScatterToModelParallelRegion)
from mindformers.parallel_core.inference.utils import get_tp_world_size, divide
from mindformers.parallel_core.inference.parallel_state import get_tensor_model_parallel_rank
from mindformers.parallel_core.inference.tensor_parallel.random import (TENSOR_PARALLEL_GENERATOR,
                                                                        get_rng_tracer)


class ColumnParallelLinear(nn.Cell):
    r"""
    The dense layer with weight sliced on second dimension by tensor parallel size.
    This layer implements the operation as:

    .. math::
        \text{outputs} = \text{inputs} * \text{weight} + \text{bias},

    where :math:`inputs` is the input tensors, :math:`\text{weight}` is a weight matrix created by the layer,
    and :math:`\text{bias}` is a bias vector created by the layer (only if has_bias is True).

    Args:
        input_size (int): The number of channels in the input space.
        output_size (int): The number of channels in the output space.
        config (dict): Transformer configuration.
        init_method (Union[Tensor, str, Initializer, numbers.Number]): The trainable init_method parameter. The values
            of str refer to the function `initializer`. Default: 'normal'.
        bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        gather_output (bool): Specifies whether gather the output on each tensor parallel rank. Default: False.
        skip_bias_add: This was added to enable performance optimizations where bias can be fused with other
                       element-wise operations. We skip adding bias but instead return it. Default: False.
        skip_weight_param_allocation (bool): Specifies whether skip the initialization of weight parameter.
            When set True, an weight tensor should be passed to construct function. Default: False.
        is_expert (bool): Specifies whether this linear layer is an expert. Default: False.
        transpose_b (bool): Specifies whether the weight parameter will be initialized as a transposed shape.
        compute_dtype (dtype.Number): The computation type. Default: mstype.bfloat16.

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
            input_size: int,
            output_size: int,
            *,
            config: TransformerConfig,
            init_method: Callable = "normal",
            bias: bool = True,
            gather_output: bool = False,
            stride: int = 1,
            keep_master_weight_for_test: bool = False,
            skip_bias_add: bool = False,
            skip_weight_param_allocation: bool = False,
            embedding_activation_buffer: Optional[List[Tensor]] = None,
            is_expert: bool = False,
            tp_comm_buffer_name: str = None,
            transpose_b: bool = True,
            compute_dtype: mstype = mstype.bfloat16
    ):
        super(ColumnParallelLinear, self).__init__()
        if stride > 1:
            raise NotImplementedError("For ColumnParallelLinear, `stride > 1` is not supported for now, "
                                      "but got `stride={}`".format(stride))
        if keep_master_weight_for_test:
            raise NotImplementedError(
                "For ColumnParallelLinear, `keep_master_weight_for_test` is not supported for now")
        if skip_bias_add:
            raise NotImplementedError("For ColumnParallelLinear, `skip_bias_add=True` is not supported for now")
        if embedding_activation_buffer is not None:
            raise NotImplementedError(
                "For ColumnParallelLinear, `embedding_activation_buffer` is not supported for now")
        if is_expert:
            raise NotImplementedError("For ColumnParallelLinear, `is_expert` is not supported for now")
        if tp_comm_buffer_name is not None:
            raise NotImplementedError("For ColumnParallelLinear, `tp_comm_buffer_name` is not supported for now")

        self.input_size = input_size
        self.output_size = output_size
        self.config = config
        self.init_method = init_method
        self.has_bias = bias
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add
        self.skip_weight_param_allocation = skip_weight_param_allocation
        self.transpose_b = transpose_b
        self.compute_dtype = compute_dtype
        self.params_dtype = config.params_dtype
        self.cast = P.Cast()
        self.matmul = P.MatMul(transpose_b=self.transpose_b)

        self.tensor_parallel_group_size = get_tp_world_size()
        self.output_size_per_partition = divide(output_size, self.tensor_parallel_group_size)
        self.gather_from_mp_region = GatherFromModelParallelRegion()

        weight_shape = (self.output_size_per_partition, self.input_size) if self.transpose_b else (
            self.input_size, self.output_size_per_partition)
        if not self.skip_weight_param_allocation:
            with get_rng_tracer().rng_fork(TENSOR_PARALLEL_GENERATOR):
                self.weight = Parameter(initializer(init_method, weight_shape, self.params_dtype), name="weight")
        else:
            self.weight = None

        bias_shape = (self.output_size_per_partition,)
        if self.has_bias:
            self.bias = Parameter(
                mint.empty(bias_shape, dtype=self.params_dtype),
                name="bias"
            )
        else:
            self.bias = None

    def construct(self, input_, weight=None):
        """
        Forward of ColumnParallelLinear.
        Performs a linear transformation considering various parallel modes and data type conversions.
        """

        if weight is None:
            if self.weight is None:
                raise RuntimeError(
                    "For ColumnParallelLinear, weight was not supplied to construct(), "
                    "and `skip_weight_param_allocation` is True."
                    )
            weight = self.weight
        else:
            # Check the weight passed in is the correct shape.
            experted_shape = (self.output_size_per_partition, self.input_size)
            if weight.shape != experted_shape:
                raise RuntimeError(
                    f"supplied weight's shape is {tuple(weight.shape)}, "
                    f"not {experted_shape} as expected."
                )

        origin_dtype = input_.dtype
        output_shape = input_.shape[:-1] + (self.output_size_per_partition,)

        input_ = mint.reshape(input_, (-1, self.input_size))
        input_ = self.cast(input_, self.compute_dtype)
        weight = self.cast(weight, self.compute_dtype)
        output_parallel = self.matmul(input_, weight)

        if self.has_bias and not self.skip_bias_add:
            bias = self.cast(self.bias, self.compute_dtype)
            output_parallel = mint.add(output_parallel, bias)

        output_parallel = mint.reshape(output_parallel, output_shape)
        output_parallel = self.cast(output_parallel, origin_dtype)

        if self.gather_output:
            output = self.gather_from_mp_region(output_parallel)
        else:
            output = output_parallel
        return output

    def sharded_state_dict(self):
        """provide the sharded state dict based on the config"""
        w_shard = (self.tensor_parallel_group_size, 1) if self.transpose_b else (1, self.tensor_parallel_group_size)

        state_dict = {}
        if not self.skip_weight_param_allocation:
            state_dict[self.weight.name] = {'shape': self.weight.shape,
                                            'shard': w_shard}
        if self.bias:
            state_dict[self.bias.name] = {'shape': self.bias.shape,
                                          'shard': (self.tensor_parallel_group_size,)}
        return state_dict


class QKVParallelLinear(ColumnParallelLinear):
    """Linear layers for the attention's QKV transformation.

    Linear layers for the linear transformation of the query, key, and value
    vectors in the attention layer. The weight matrix is concatenated along
    the output dimension. The layer is parallelized along the head dimension.
    When the number of key/value heads is smaller than the number of query
    heads (e.g., multi-query/grouped-query attention), the key/value head may
    be replicated while the query heads are partitioned.

    Args:
        hidden_size: input hidden state size of the transformer.
        head_size: size of each attention head.
        total_num_heads: total number of attention query heads.
        total_num_kv_heads: total number of attention key/value heads. If
                            None, assume total_num_kv_heads = total_num_heads.
        config (dict): Transformer configuration.
        bias: If true, add bias.
        gather_output (bool): Specifies whether gather the output on each tensor parallel rank. Default: False.
        transpose_b (bool): Specifies whether the weight parameter will be initialized as a transposed shape.
        compute_dtype (dtype.Number): The computation type. Default: mstype.bfloat16.
    """

    def __init__(
            self,
            hidden_size: int,
            head_size: int,
            total_num_heads: int,
            total_num_kv_heads: int,
            *,
            config: TransformerConfig,
            bias: bool = True,
            gather_output: bool = False,
            transpose_b: bool = True,
            compute_dtype: mstype = None
    ):
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.params_dtype = config.params_dtype

        # Divide the weight matrix along the last dimension.
        tp_size = get_tp_world_size()
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        output_size = (
            (self.num_heads + 2 * self.num_kv_heads) * tp_size * self.head_size
        )
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,  # q_proj
            self.num_kv_heads * self.head_size * tp_size,  # k_proj
            self.num_kv_heads * self.head_size * tp_size,  # v_proj
        ]

        super().__init__(
            input_size=hidden_size,
            output_size=output_size,
            config=config,
            bias=bias,
            gather_output=gather_output,
            transpose_b=transpose_b,
            compute_dtype=compute_dtype
        )


class RowParallelLinear(nn.Cell):
    r"""
    The dense layer with weight sliced on first dimension by tensor parallel size.
    This layer implements the operation as:

    .. math::
        \text{outputs} = \text{inputs} * \text{weight} + \text{bias},

    where :math:`inputs` is the input tensors, :math:`\text{weight}` is a weight matrix created by the layer,
    and :math:`\text{bias}` is a bias vector created by the layer (only if has_bias is True).

    Args:
        input_size (int): The number of channels in the input space.
        output_size (int): The number of channels in the output space.
        config (dict): Transformer configuration.
        init_method (Union[Tensor, str, Initializer, numbers.Number]): The trainable init_method parameter. The values
            of str refer to the function `initializer`. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The values
            of str refer to the function `initializer`. Default: 'zeros'.
        input_is_parallel (bool): Specifies whether the input tensor has already been sliced on last dimension.
        skip_bias_add: This was added to enable performance optimizations where bias can be fused with other
            element-wise operations. We skip adding bias but instead return it. Default: False.
        is_expert (bool): Specifies whether this linear layer is an expert. Default: False.
        transpose_b (bool): Specifies whether the weight parameter will be initialized as a transposed shape.
        compute_dtype (dtype.Number): The computation type. Default: mstype.bfloat16.

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
            input_size: int,
            output_size: int,
            *,
            config: TransformerConfig,
            init_method: Callable = "normal",
            bias: bool = True,
            input_is_parallel: bool = True,
            skip_bias_add: bool = False,
            stride: int = 1,
            keep_master_weight_for_test: bool = False,
            is_expert: bool = False,
            tp_comm_buffer_name: str = None,
            transpose_b: bool = True,
            compute_dtype: mstype = mstype.bfloat16,
    ):
        super(RowParallelLinear, self).__init__()
        if stride > 1:
            raise NotImplementedError("For RowParallelLinear, `stride > 1` is not supported for now, "
                                      "but got `stride={}`".format(stride))
        if skip_bias_add:
            raise NotImplementedError("For RowParallelLinear, `skip_bias_add=True` is not supported for now")
        if keep_master_weight_for_test:
            raise NotImplementedError("For RowParallelLinear, `keep_master_weight_for_test=True` "
                                      "is not supported for now.")
        if is_expert:
            raise NotImplementedError("For RowParallelLinear, `is_expert` is not supported for now")
        if tp_comm_buffer_name:
            raise NotImplementedError("For RowParallelLinear, `tp_comm_buffer_name` is not supported for now.")

        self.input_size = input_size
        self.output_size = output_size
        self.config = config
        self.input_is_parallel = input_is_parallel
        self.has_bias = bias
        self.skip_bias_add = skip_bias_add

        self.tensor_parallel_group_size = get_tp_world_size()
        self.input_size_per_partition = divide(input_size, self.tensor_parallel_group_size)
        self.compute_dtype = compute_dtype
        self.transpose_b = transpose_b
        self.params_dtype = config.params_dtype
        self.cast = P.Cast()
        self.matmul = P.MatMul(transpose_b=self.transpose_b)

        self.reduce_from_mp_region = ReduceFromModelParallelRegion()
        if not self.input_is_parallel:
            self.scatter_to_mp_region = ScatterToModelParallelRegion()

        weight_shape = (self.output_size, self.input_size_per_partition) if self.transpose_b else (
            self.input_size_per_partition, self.output_size)
        with get_rng_tracer().rng_fork(TENSOR_PARALLEL_GENERATOR):
            self.weight = Parameter(initializer(init_method, weight_shape, self.params_dtype), name="weight")

        bias_shape = (self.output_size,)
        if self.has_bias:
            self.bias = Parameter(
                mint.empty(bias_shape, dtype=self.params_dtype),
                name="bias"
                )
        else:
            self.bias = None

    def construct(self, input_):
        """
        Forward of RowParallelLinear.
        Performs a linear transformation considering various parallel modes and data type conversions.
        """

        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = self.scatter_to_mp_region(input_)

        origin_dtype = input_parallel.dtype
        output_shape = input_parallel.shape[:-1] + (self.output_size,)

        input_parallel = mint.reshape(input_parallel, (-1, self.input_size_per_partition))
        input_parallel = self.cast(input_parallel, self.compute_dtype)
        weight = self.cast(self.weight, self.compute_dtype)
        output_parallel = self.matmul(input_parallel, weight)
        output = self.reduce_from_mp_region(output_parallel)

        if self.has_bias and not self.skip_bias_add:
            bias = self.cast(self.bias, self.compute_dtype)
            output = mint.add(output, bias)

        output = mint.reshape(output, output_shape)
        output = self.cast(output, origin_dtype)
        return output

    def sharded_state_dict(self):
        """provide the sharded state dict based on the config"""
        w_shard = (1, self.tensor_parallel_group_size) if self.transpose_b else (self.tensor_parallel_group_size, 1)

        if self.is_expert and self.expert_num > 1:
            w_shard = (1, 1, self.tensor_parallel_group_size) if self.transpose_b \
                else (1, self.tensor_parallel_group_size, 1)

        state_dict = {}
        state_dict[self.weight.name] = {'shape': self.weight.shape,
                                        'shard': w_shard}
        if self.bias:
            state_dict[self.bias.name] = {'shape': self.bias.shape,
                                          'shard': (1,)}
        return state_dict


class ReplicatedLinear(nn.Cell):
    """Replicated linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        config (dict): Transformer configuration.
        init_method (Union[Tensor, str, Initializer, numbers.Number]): The trainable init_method parameter. The values
            of str refer to the function `initializer`. Default: 'normal'.
        bias: If true, add bias.
        gather_output (bool): Specifies whether gather the output on each tensor parallel rank. Default: False.
        skip_bias_add: This was added to enable performance optimizations where bias can be fused with other
            element-wise operations. We skip adding bias but instead return it.
        skip_weight_param_allocation (bool): Specifies whether skip the initialization of weight parameter.
            When set True, an weight tensor should be passed to construct function. Default: False.
        is_expert (bool): Specifies whether this linear layer is an expert. Default: False.
        transpose_b (bool): Specifies whether the weight parameter will be initialized as a transposed shape.
        compute_dtype (dtype.Number): The computation type. Default: mstype.bfloat16.
    """

    def __init__(
            self,
            input_size: int,
            output_size: int,
            *,
            config: TransformerConfig,
            init_method: Callable = "normal",
            bias: bool = True,
            stride: int = 1,
            keep_master_weight_for_test: bool = False,
            skip_bias_add: bool = False,
            skip_weight_param_allocation: bool = False,
            embedding_activation_buffer: Optional[List[Tensor]] = None,
            is_expert: bool = False,
            tp_comm_buffer_name: str = None,
            transpose_b: bool = True,
            compute_dtype: mstype = None
    ):
        super(ReplicatedLinear, self).__init__()
        if stride > 1:
            raise NotImplementedError("For ReplicatedLinear, `stride > 1` is not supported for now, "
                                      "but got `stride={}`".format(stride))
        if skip_bias_add:
            raise NotImplementedError("For ReplicatedLinear, `skip_bias_add=True` is not supported for now")
        if keep_master_weight_for_test:
            raise NotImplementedError(
                "For ReplicatedLinear, `keep_master_weight_for_test` is not supported for now")
        if embedding_activation_buffer is not None:
            raise NotImplementedError(
                "For ReplicatedLinear, `embedding_activation_buffer` is not supported for now")
        if is_expert:
            raise NotImplementedError("For ReplicatedLinear, `is_expert` is not supported for now")
        if tp_comm_buffer_name is not None:
            raise NotImplementedError("For ReplicatedLinear, `tp_comm_buffer_name` is not supported for now")

        self.input_size = input_size
        self.output_size = output_size
        self.config = config
        self.init_method = init_method
        self.has_bias = bias
        self.skip_bias_add = skip_bias_add
        self.skip_weight_param_allocation = skip_weight_param_allocation
        self.transpose_b = transpose_b
        self.compute_dtype = compute_dtype
        self.params_dtype = config.params_dtype
        self.cast = P.Cast()
        self.matmul = P.MatMul(transpose_b=self.transpose_b)

        self.tensor_parallel_group_size = 1

        weight_shape = (self.output_size, self.input_size) if self.transpose_b else (
            self.input_size, self.output_size)
        if not self.skip_weight_param_allocation:
            self.weight = Parameter(initializer(init_method, weight_shape, self.params_dtype), name="weight")
        else:
            self.weight = None

        bias_shape = (self.output_size,)
        if self.has_bias:
            self.bias = Parameter(
                mint.empty(bias_shape, dtype=self.params_dtype),
                name="bias"
            )
        else:
            self.bias = None

    def construct(self, input_, weight=None):
        """
        Forward of ReplicatedLinear.
        Performs a linear transformation considering various parallel modes and data type conversions.
        """
        if weight is None:
            if self.weight is None:
                raise RuntimeError(
                    "For ColumnParallelLinear, weight was not supplied to construct(), "
                    "and `skip_weight_param_allocation` is True."
                    )
            weight = self.weight
        else:
            # Check the weight passed in is the correct shape.
            experted_shape = (self.output_size_per_partition, self.input_size)
            if weight.shape != experted_shape:
                raise RuntimeError(
                    f"supplied weight's shape is {tuple(weight.shape)}, "
                    f"not {experted_shape} as expected."
                )

        origin_dtype = input_.dtype
        output_shape = input_.shape[:-1] + (self.output_size,)

        input_ = mint.reshape(input_, (-1, self.input_size))
        input_ = self.cast(input_, self.compute_dtype)
        weight = self.cast(weight, self.compute_dtype)
        output = self.matmul(input_, weight)

        if self.has_bias and not self.skip_bias_add:
            bias = self.cast(self.bias, self.compute_dtype)
            output = mint.add(output, bias)

        output = mint.reshape(output, output_shape)
        output = self.cast(output, origin_dtype)
        return output


class VocabParallelEmbedding(nn.Cell):
    """
    Embedding parallelized in the vocabulary dimension.

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        config (TransformerConfig): Configuration for the transformer model.
        init_method (Union[Tensor, str, Initializer, numbers.Number]): The trainable init_method parameter. The values
            of str refer to the function `initializer`. Default: 'normal'.
        reduce_scatter_embeddings (bool): Decides whether to perform ReduceScatter after embedding lookup.
    """

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            *,
            init_method: Callable,
            config: TransformerConfig,
            reduce_scatter_embeddings: bool = False,
    ):
        super().__init__()
        if reduce_scatter_embeddings:
            raise NotImplementedError("For VocabParallelEmbedding, reduce_scatter_embeddings is not supported for now")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.sequence_parallel = config.sequence_parallel

        self.tensor_parallel_group_size = get_tp_world_size()
        rank_id = get_tensor_model_parallel_rank() if self.tensor_parallel_group_size > 1 else 0

        (
            self.vocab_start_index,
            self.vocab_end_index,
        ) = self._vocab_range_from_global_vocab_size(
            self.num_embeddings, rank_id, self.tensor_parallel_group_size
        )
        self.num_embeddings_per_partition = (
            self.vocab_end_index - self.vocab_start_index
        )

        with get_rng_tracer().rng_fork():
            self.weight = Parameter(
                init_method([self.num_embeddings_per_partition, self.embedding_dim]).astype(config.params_dtype),
                name="weight",
            )
        self.reduce_from_mp_region = ReduceFromModelParallelRegion()
        self.reduce_scatter_to_sp_region = ReduceScatterToSequenceParallelRegion()
        self.max_index_per_partition = Tensor(self.num_embeddings_per_partition - 1, dtype=mstype.int32)
        self.expand_dims = ops.ExpandDims()
        self.gather = ops.Gather()

    def construct(self, x):
        """
        Forward of VocabParallelEmbedding.
        Computes embeddings with optional masking and parallel reduction based on the model parallel size.
        """

        if self.tensor_parallel_group_size > 1:
            displaced_x = mint.sub(x, self.vocab_start_index)
            down_truncated_x = mint.nn.functional.relu(displaced_x)
            truncated_x = mint.minimum(down_truncated_x, self.max_index_per_partition)
            input_mask = mint.eq(displaced_x, truncated_x)
            input_mask = self.expand_dims(input_mask, -1)
        else:
            input_mask = None
            truncated_x = x
        # Get the embeddings.
        # 'embedding' has dynamic shape issue, use gather instead now.
        output_parallel = self.gather(self.weight, truncated_x, 0)
        # Mask the output embedding.
        if self.tensor_parallel_group_size > 1:
            output_parallel = mint.mul(output_parallel, input_mask)

        if self.sequence_parallel:
            output_parallel = output_parallel.swapaxes(0, 1).contiguous()
            output = self.reduce_scatter_to_sp_region(output_parallel)
            output = output.swapaxes(0, 1).contiguous()
        else:
            # Reduce across all the model parallel devices.
            output = self.reduce_from_mp_region(output_parallel)
        return output

    def _vocab_range_from_global_vocab_size(self, global_vocab_size, rank, world_size):
        if global_vocab_size % world_size != 0:
            raise ValueError(f"The vocabulary size is {global_vocab_size},"
                             f"which is not divisible by size of tensor parallel({world_size}).")
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    def sharded_state_dict(self):
        """provide the sharded state dict based on the config"""
        w_shard = (self.tensor_parallel_group_size, 1)
        state_dict = {}
        state_dict[self.weight.name] = {'shape': self.weight.shape,
                                        'shard': w_shard}

        return state_dict
