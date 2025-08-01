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
"""Layers"""
import mindspore.common.dtype as mstype
import mindspore.ops.functional as F
import mindspore.ops.operations as P
from mindspore import Parameter, Tensor, mint, nn, ops
from mindspore.common.initializer import initializer

from mindformers.parallel_core.inference.tensor_parallel.mappings import (gather_from_model_parallel_region,
                                                                          reduce_from_model_parallel_region,
                                                                          reduce_scatter_to_model_parallel_region,
                                                                          scatter_to_model_parallel_region)
from mindformers.parallel_core.inference.tensor_parallel.random import (TENSOR_PARALLEL_GENERATOR,
                                                                        get_rng_tracer)
from mindformers.parallel_core.inference.utils import divide
from mindformers.parallel_core.inference.parallel_state import default_pgs
from mindformers.version_control import check_valid_gmm_op
from mindformers.models.utils import jit

__all__ = ["ColumnParallelLinear", "RowParallelLinear", "VocabParallelEmbedding"]


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
        config (dict): Parallel configuration.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The values
            of str refer to the function `initializer`. Default: 'normal'.
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
        tp_group (ProcessGroup): The process_group this linear layer used. Default: default_pgs.

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
            weight_init="normal",
            bias_init="zeros",
            bias=True,
            gather_output=False,
            stride=1,
            keep_master_weight_for_test=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            embedding_activation_buffer=None,
            grad_output_buffer=None,
            is_expert=False,
            tp_comm_buffer_name=None,
            disable_grad_reduce=False,
            transpose_b=True,
            param_init_type=mstype.float32,
            compute_dtype=mstype.float16,
            expert_num=1,
            tp_group=default_pgs,
    ):
        super(ColumnParallelLinear, self).__init__()
        if stride > 1:
            raise NotImplementedError("For ColumnParallelLinear, `stride > 1` is not supported for now, "
                                      "but got `stride={}`".format(stride))
        if keep_master_weight_for_test:
            raise NotImplementedError("For ColumnParallelLinear, `keep_master_weight_for_test=True` "
                                      "is not supported for now.")
        if skip_bias_add:
            raise NotImplementedError("For ColumnParallelLinear, `skip_bias_add=True` is not supported for now.")
        if embedding_activation_buffer:
            raise NotImplementedError("For ColumnParallelLinear, `embedding_activation_buffer` is not supported "
                                      "for now.")
        if grad_output_buffer:
            raise NotImplementedError("For ColumnParallelLinear, `grad_output_buffer` is not supported for now.")
        if tp_comm_buffer_name:
            raise NotImplementedError("For ColumnParallelLinear, `tp_comm_buffer_name` is not supported for now.")
        if disable_grad_reduce:
            raise NotImplementedError("For ColumnParallelLinear, `disable_grad_reduce=True` is not supported for now.")

        self.input_size = input_size
        self.output_size = output_size
        self.has_bias = bias
        self.gather_output = gather_output
        self.tp_group = tp_group
        self.tensor_parallel_group_size = self.tp_group.size

        self.output_size_per_partition = divide(output_size, self.tensor_parallel_group_size)
        self.is_expert = is_expert
        self.expert_num = expert_num
        self.skip_weight_param_allocation = skip_weight_param_allocation
        self.parallel_config = config
        self.compute_dtype = compute_dtype

        self.sequence_parallel = self.parallel_config.use_sequence_parallel
        self.transpose_b = transpose_b if self.expert_num <= 1 else False

        if self.sequence_parallel and self.tensor_parallel_group_size <= 1:
            self.sequence_parallel = False

        weight_shape = (self.output_size_per_partition, self.input_size) if self.transpose_b else (
            self.input_size, self.output_size_per_partition)
        if self.is_expert and self.expert_num > 1:
            weight_shape = (self.expert_num,) + weight_shape
            if check_valid_gmm_op(gmm_version='GroupedMatmulV4'):
                self.matmul = ops.auto_generate.GroupedMatmulV4()
            elif check_valid_gmm_op(gmm_version='GroupedMatmul'):
                self.matmul = ops.auto_generate.GroupedMatmul(split_item=3, group_type=0)
            else:
                raise RuntimeError(f"Inference of the MoE model relies on the GMM op. "
                                   "Please upgrade to a MindSpore version above 2.3.0.")
        else:
            self.matmul = P.MatMul(transpose_b=self.transpose_b)
        with get_rng_tracer().rng_fork(TENSOR_PARALLEL_GENERATOR):
            if not self.skip_weight_param_allocation:
                self.weight = Parameter(initializer(weight_init, weight_shape, param_init_type), name="weight")

        if self.has_bias:
            self.bias = Parameter(
                initializer(
                    bias_init, (self.output_size_per_partition), param_init_type
                ),
                name="bias",
            )
            self.bias_add = P.Add()

        self.cast = P.Cast()
        self.shape = P.Shape()
        self.reshape = P.Reshape()

    @jit
    def construct(self, input_parallel, weight=None, group_list=None):
        """
        Forward of ColumnParallelLinear.
        Performs a linear transformation considering various parallel modes and data type conversions.
        """

        if weight is None and self.skip_weight_param_allocation:
            raise ValueError("For ColumnParallelLinear, when skip_weight_param_allocation=True,"
                             " weight should be passed to construct(), but got None.")

        origin_dtype = F.dtype(input_parallel)
        if self.skip_weight_param_allocation:
            weight = self.cast(weight, self.compute_dtype)
        else:
            weight = self.cast(self.weight, self.compute_dtype)
        input_parallel = self.cast(input_parallel, self.compute_dtype)

        if self.sequence_parallel:
            input_parallel = input_parallel.swapaxes(0, 1).contiguous()
            input_parallel = gather_from_model_parallel_region(input_parallel, self.tp_group)
            input_parallel = input_parallel.swapaxes(0, 1).contiguous()

        output_shape = self.shape(input_parallel)[:-1] + (self.output_size_per_partition,)
        input_parallel = self.reshape(input_parallel, (-1, self.input_size))
        if self.is_expert and self.expert_num > 1:
            if check_valid_gmm_op(gmm_version='GroupedMatmulV4'):
                output_parallel = self.matmul([input_parallel], [weight], None, None, None, None, None, None,
                                              group_list, split_item=3, group_type=0, group_list_type=1)[0]
            elif check_valid_gmm_op(gmm_version='GroupedMatmul'):
                output_parallel = self.matmul([input_parallel], [weight], None, None, None, None, None,
                                              group_list)[0]
            else:
                raise RuntimeError(f"Inference of the MoE model relies on the GMM op. "
                                   "Please upgrade to a MindSpore version above 2.3.0.")

        else:
            output_parallel = self.matmul(input_parallel, weight)
        if self.has_bias:
            output_parallel = self.bias_add(
                output_parallel, self.cast(self.bias, self.compute_dtype)
            )
        output_parallel = self.cast(output_parallel, origin_dtype)
        output_parallel = self.reshape(output_parallel, output_shape)

        if self.gather_output:
            output = gather_from_model_parallel_region(output_parallel, self.tp_group)
        else:
            output = output_parallel
        return output

    def sharded_state_dict(self):
        """provide the sharded state dict based on the config"""
        w_shard = (self.tensor_parallel_group_size, 1) if self.transpose_b else (1, self.tensor_parallel_group_size)

        if self.is_expert and self.expert_num > 1:
            w_shard = (1, self.tensor_parallel_group_size, 1) if self.transpose_b \
                else (1, 1, self.tensor_parallel_group_size)

        state_dict = {}
        if not self.skip_weight_param_allocation:
            state_dict[self.weight.name] = {'shape': self.weight.shape,
                                            'shard': w_shard}
        if self.has_bias:
            state_dict[self.bias.name] = {'shape': self.bias.shape,
                                          'shard': (self.tensor_parallel_group_size,)}
        return state_dict


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
        config (dict): Parallel configuration.
        input_is_parallel (bool): Specifies whether the input tensor has already been sliced on last dimension.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The values
            of str refer to the function `initializer`. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The values
            of str refer to the function `initializer`. Default: 'zeros'.
        bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        skip_bias_add (bool): Specifies whether the layer doesn't need to add bias. Default: False.
        is_expert (bool): Specifies whether this linear layer is an expert. Default: False.
        transpose_b (bool): Specifies whether the weight parameter will be initialized as a transposed shape.
        param_init_type (dtype.Number): The parameter initialization type. Default: mstype.float32.
        compute_dtype (dtype.Number): The computation type. Default: mstype.float16.
        expert_num (int): The number of expert. Default: 1.
        tp_group (ProcessGroup): The process_group this linear layer used. Default: default_pgs.

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
            weight_init="normal",
            bias_init="zeros",
            bias=True,
            skip_bias_add=False,
            stride=1,
            keep_master_weight_for_test=False,
            is_expert=False,
            tp_comm_buffer_name=None,
            transpose_b=True,
            param_init_type=mstype.float32,
            compute_dtype=mstype.float16,
            expert_num=1,
            delay_allreduce=False,
            tp_group=default_pgs,
    ):
        super(RowParallelLinear, self).__init__()
        if stride > 1:
            raise NotImplementedError("For ColumnParallelLinear, `stride > 1` is not supported for now, "
                                      "but got `stride={}`".format(stride))
        if keep_master_weight_for_test:
            raise NotImplementedError("For ColumnParallelLinear, `keep_master_weight_for_test=True` "
                                      "is not supported for now.")
        if tp_comm_buffer_name:
            raise NotImplementedError("For ColumnParallelLinear, `tp_comm_buffer_name` is not supported for now.")

        self.input_size = input_size
        self.output_size = output_size
        self.has_bias = bias
        self.skip_bias_add = skip_bias_add
        self.input_is_parallel = input_is_parallel
        self.tp_group = tp_group
        self.tensor_parallel_group_size = self.tp_group.size
        self.input_size_per_partition = divide(input_size, self.tensor_parallel_group_size)
        self.parallel_config = config
        self.compute_dtype = compute_dtype
        self.sequence_parallel = self.parallel_config.use_sequence_parallel
        self.expert_num = expert_num
        self.is_expert = is_expert
        self.transpose_b = transpose_b if self.expert_num <= 1 else False
        self.delay_allreduce = delay_allreduce

        if self.sequence_parallel and not self.input_is_parallel:
            raise RuntimeError(
                "To enable `sequence_arallel`, `input_is_parallel` must be `True`"
            )

        if self.delay_allreduce and self.has_bias:
            raise RuntimeError(
                "In RowParallelLinear, `delay_allreduce` and `has_bias` cannot be enabled simultaneously, "
                "otherwise the accuracy will be incorrect"
            )

        weight_shape = (self.output_size, self.input_size_per_partition) if self.transpose_b else (
            self.input_size_per_partition, self.output_size)
        bias_shape = (self.output_size,)
        if self.is_expert and self.expert_num > 1:
            weight_shape = (self.expert_num,) + weight_shape
            bias_shape = (1, self.expert_num, 1) + bias_shape
            if check_valid_gmm_op(gmm_version='GroupedMatmulV4'):
                self.matmul = ops.auto_generate.GroupedMatmulV4()
            elif check_valid_gmm_op(gmm_version='GroupedMatmul'):
                self.matmul = ops.auto_generate.GroupedMatmul(split_item=3, group_type=0)
            else:
                raise RuntimeError(f"Inference of the MoE model relies on the GMM op. "
                                   "Please upgrade to a MindSpore version above 2.3.0.")
        else:
            self.matmul = P.MatMul(transpose_b=self.transpose_b)
        with get_rng_tracer().rng_fork(TENSOR_PARALLEL_GENERATOR):
            self.weight = Parameter(
                initializer(
                    weight_init,
                    weight_shape,
                    param_init_type,
                ),
                name="weight",
            )

        if self.has_bias:
            self.bias = Parameter(initializer(bias_init, bias_shape, param_init_type), name="bias")
            self.bias_add = P.Add()

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.cast = P.Cast()

    def construct(self, input_, group_list=None):
        """
        Forward of RowParallelLinear.
        Performs a linear transformation considering various parallel modes and data type conversions.
        """

        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_, self.tp_group)

        origin_dtype = F.dtype(input_parallel)
        weight = self.cast(self.weight, self.compute_dtype)
        input_parallel = self.cast(input_parallel, self.compute_dtype)
        output_shape = self.shape(input_parallel)[:-1] + (self.output_size,)
        input_parallel = self.reshape(input_parallel, (-1, self.input_size_per_partition))
        if self.is_expert and self.expert_num > 1:
            if check_valid_gmm_op(gmm_version='GroupedMatmulV4'):
                output_parallel = self.matmul([input_parallel], [weight], None, None, None, None, None, None,
                                              group_list, split_item=3, group_type=0, group_list_type=1)[0]
            elif check_valid_gmm_op(gmm_version='GroupedMatmul'):
                output_parallel = self.matmul([input_parallel], [weight], None, None, None, None, None,
                                              group_list)[0]
            else:
                raise RuntimeError(f"Inference of the MoE model relies on the GMM op. "
                                   "Please upgrade to a MindSpore version above 2.3.0.")
        else:
            output_parallel = self.matmul(input_parallel, weight)

        if self.sequence_parallel:
            output_parallel = output_parallel.swapaxes(0, 1).contiguous()
            output = reduce_scatter_to_model_parallel_region(output_parallel, self.tp_group)
            output = output.swapaxes(0, 1).contiguous()
        else:
            if self.delay_allreduce or self.skip_bias_add:
                output = output_parallel
            else:
                output = reduce_from_model_parallel_region(output_parallel, self.tp_group)

        if self.has_bias and not self.skip_bias_add:
            output = self.bias_add(output, self.cast(self.bias, self.compute_dtype))
        output = self.cast(output, origin_dtype)
        output = self.reshape(output, output_shape)
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
        if self.has_bias:
            state_dict[self.bias.name] = {'shape': self.bias.shape,
                                          'shard': (1,)}
        return state_dict


class VocabParallelEmbedding(nn.Cell):
    """
    Embedding parallelized in the vocabulary dimension.

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        parallel_config (Optional[Union[dict, ParallelContextConfig]]):
            Parallel Config For Running Environment. Default: None.
        init_method (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The values
            of str refer to the function `initializer`. Default: 'normal'.
        init_type (dtype.Number): The parameter initialization type. Default: mstype.float32.
        tp_group (ProcessGroup): The process_group this linear layer used. Default: default_pgs.
    """

    def __init__(
            self,
            num_embeddings,
            embedding_dim,
            parallel_config,
            init_method="normal",
            init_type=mstype.float32,
            tp_group=default_pgs,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.sequence_parallel = parallel_config.use_sequence_parallel

        self.tp_group = tp_group
        self.tensor_parallel_group_size = self.tp_group.size
        rank = self.tp_group.rank

        self.vocab_start_index, self.vocab_end_index = self._vocab_range_from_global_vocab_size(
            self.num_embeddings, rank, self.tensor_parallel_group_size
        )
        self.num_embeddings_per_partition = (
            self.vocab_end_index - self.vocab_start_index
        )

        with get_rng_tracer().rng_fork():
            self.embedding_weight = Parameter(
                initializer(
                    init=init_method,
                    shape=(self.num_embeddings_per_partition, self.embedding_dim),
                    dtype=init_type,
                ),
                name="embedding_weight",
            )
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
        output_parallel = self.gather(self.embedding_weight, truncated_x, 0)
        # Mask the output embedding.
        if self.tensor_parallel_group_size > 1:
            output_parallel = mint.mul(output_parallel, input_mask)

        if self.sequence_parallel:
            output_parallel = output_parallel.swapaxes(0, 1).contiguous()
            output = reduce_scatter_to_model_parallel_region(output_parallel, self.tp_group)
            output = output.swapaxes(0, 1).contiguous()
        else:
            # Reduce across all the model parallel devices.
            output = reduce_from_model_parallel_region(output_parallel, self.tp_group)
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
        state_dict[self.embedding_weight.name] = {'shape': self.embedding_weight.shape,
                                                  'shard': w_shard}

        return state_dict
