# Copyright 2022 Huawei Technologies Co., Ltd
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
from mindspore import Parameter, Tensor, nn, ops, mint
from mindspore.common.initializer import initializer, Zero

from mindformers.experimental.distri_cores.create_comm import (
    get_tp_rank,
    get_tp_world_size,
    get_dp_world_size,
)
from mindformers.experimental.distri_cores.tensor_parallel import (
    CopyToModelParallelRegion,
    GatherFromModelParallelRegion,
    GatherFromSequenceParallelRegion,
    ReduceFromModelParallelRegion,
    ReduceScatterToSequenceParallelRegion,
    ScatterToModelParallelRegion,
)
from mindformers.experimental.distri_cores.utils import divide
from mindformers.experimental.distri_cores.random import (
    get_rng_tracer,
    TENSOR_PARALLEL_GENERATOR,
    EXPERT_PARALLEL_GENERATOR,
)

__all__ = ["ColumnParallelLinear", "RowParallelLinear"]


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
        init_method (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The values
            of str refer to the function `initializer`. Default: 'normal'.
        bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        gather_output (bool): Specifies whether gather the output on each tensor parallel rank. Default: False.
        skip_weight_param_allocation (bool): Specifies whether skip the initialization of weight parameter.
            When set True, an weight tensor should be passed to construct function. Default: False.
        is_expert (bool): Specifies whether this linear layer is an expert. Default: False.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The values
            of str refer to the function `initializer`. Default: Zero().
        param_init_dtype (dtype.Number): The parameter initialization type. Default: None.
        compute_dtype (dtype.Number): The computation type. Default: None.
        transpose_b (bool): Specifies whether the weight parameter will be initialized as a transposed shape.
            Default: True.

    Inputs:
        - **input_** (Tensor) - Tensor of shape :math:`(*, input\_size)`. The `input_size` in `Args` should be equal
          to :math:`input\_size` in `Inputs`.
        - **weight** (Tensor) - Tensor of shape :math:`(input\_size, output\_size)`/:math`(output\_size, input\_size)`.
          When `skip_weight_param_allocation=True`, this input must be provided. Default: None.

    Outputs:
        - **output** (Tensor): Result of linear with shape :math:`(*, output\_size)`.
        - **output_bias** (Parameter): Bias parameter when `skip_bias_add=True` with shape :math:`(output\_size)`.

    Raises:
        ValueError: `skip_weight_param_allocation=True` but weight_tensor is not passed to construct function.

    Supported Platforms:
        ``Ascend``
    """
    def __init__(
            self,
            input_size,
            output_size,
            *,
            config,
            init_method,
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
            bias_init=Zero(),
            param_init_dtype=None,
            compute_dtype=None,
            transpose_b=True,
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
        if is_expert:
            raise NotImplementedError("For ColumnParallelLinear, `is_expert=True` is not supported for now.")

        self.input_size = input_size
        self.output_size = output_size
        self.has_bias = bias
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add

        tensor_parallel_group_size = get_tp_world_size()
        self.output_size_per_partition = divide(output_size, tensor_parallel_group_size)

        self.is_expert = is_expert
        self.skip_weight_param_allocation = skip_weight_param_allocation
        self.config = config
        self.param_init_dtype = param_init_dtype if param_init_dtype else self.config.param_init_dtype
        self.compute_dtype = compute_dtype if compute_dtype else self.config.compute_dtype

        self.expert_parallel = self.config.parallel_config.expert_parallel > 1
        self.sequence_parallel = self.config.parallel_config.use_sequence_parallel
        self.use_zero3 = self.config.parallel_config.zero_level == 'z3'
        self.transpose_b = transpose_b
        if self.use_zero3:
            try:
                dp_size = get_dp_world_size()
            except AssertionError:
                raise RuntimeError("When using zero3 optimizer parallel. Data parallel communication "
                                   "need be initialized. Please check 'dp' in order when calling "
                                   "initialize_model_parallel.")
        if self.transpose_b:
            if self.use_zero3 and self.output_size_per_partition % dp_size == 0:
                self.output_size_per_partition = divide(self.output_size_per_partition, dp_size)
            else:
                self.use_zero3 = False
        else:
            if self.use_zero3 and self.input_size % dp_size == 0 and self.output_size_per_partition % dp_size == 0:
                self.input_size = divide(self.input_size, dp_size)
            else:
                self.use_zero3 = False

        if self.sequence_parallel and tensor_parallel_group_size <= 1:
            self.sequence_parallel = False

        mode = EXPERT_PARALLEL_GENERATOR if self.is_expert and self.expert_parallel else TENSOR_PARALLEL_GENERATOR
        weight_shape = (self.output_size_per_partition, self.input_size) if self.transpose_b \
                        else (self.input_size, self.output_size_per_partition)
        with get_rng_tracer().rng_fork(mode):
            if not self.skip_weight_param_allocation:
                self.weight = Parameter(
                    initializer(
                        init_method,
                        weight_shape,
                        self.param_init_dtype,
                    ),
                    name="weight",
                )
        self.matmul = P.BatchMatMul(transpose_b=self.transpose_b)

        if self.has_bias:
            if self.use_zero3 and not self.transpose_b:
                self.output_size_per_partition = divide(self.output_size_per_partition, dp_size)
            self.bias = Parameter(
                initializer(
                    bias_init, (self.output_size_per_partition), self.param_init_dtype
                ),
                name="bias",
            )

        self.explicit_expert_comm = self.is_expert and (
            self.sequence_parallel or self.expert_parallel
        )

        self.copy_to_mp_region = CopyToModelParallelRegion()
        self.gather_from_mp_region = GatherFromModelParallelRegion()
        self.gather_from_sp_region = GatherFromSequenceParallelRegion()

    def construct(self, input_, weight=None):
        """construct method."""
        if weight is None and self.skip_weight_param_allocation:
            raise ValueError("For ColumnParallelLinear, when skip_weight_param_allocation=True,"
                             " weight should be passed to construct(), but got None.")

        if weight and not self.skip_weight_param_allocation:
            raise ValueError("For ColumnParallelLinear, when skip_weight_param_allocation=False,"
                             "weight should not be passed to construct(), but got {}".format(weight))

        if self.sequence_parallel or self.explicit_expert_comm:
            input_parallel = input_
        else:
            input_parallel = self.copy_to_mp_region(input_)

        origin_dtype = F.dtype(input_parallel)
        if self.skip_weight_param_allocation:
            weight = self.cast(weight, self.compute_dtype)
        else:
            weight = self.cast(self.weight, self.compute_dtype)
        input_parallel = self.cast(input_parallel, self.compute_dtype)

        if self.sequence_parallel:
            input_parallel = input_parallel.swapaxes(0, 1).contiguous()
            input_parallel = self.gather_from_sp_region(input_parallel)
            input_parallel = input_parallel.swapaxes(0, 1).contiguous()
        output_parallel = self.matmul(input_parallel, weight)

        bias = self.cast(self.bias, self.compute_dtype) if self.has_bias and not self.skip_bias_add else None

        if self.has_bias and not self.skip_bias_add:
            output_parallel = mint.add(output_parallel, bias)
        output_parallel = self.cast(output_parallel, origin_dtype)

        if self.gather_output:
            output = self.gather_from_mp_region(output_parallel)
        else:
            output = output_parallel

        output_bias = self.bias if self.has_bias and self.skip_bias_add else None

        return output, output_bias

    def sharded_state_dict(self):
        """provide the sharded state dict based on the config"""
        tp_size = get_tp_world_size()
        w_shard = (tp_size, 1) if self.transpose_b else (1, tp_size)
        state_dict = {}
        opt_weight_shard_step = get_tp_world_size() if self.use_zero3 else 0
        try:
            opt_weight_shard_size = get_dp_world_size() if self.use_zero3 else -1
        except AssertionError:
            opt_weight_shard_size = -1
        if not self.skip_weight_param_allocation:
            state_dict[self.weight.name] = {'shape': self.weight.shape,
                                            'shard': w_shard,
                                            'opt_weight_shard_step': opt_weight_shard_step,
                                            'opt_weight_shard_size': opt_weight_shard_size}
        if self.has_bias:
            state_dict[self.bias.name] = {'shape': self.bias.shape,
                                          'shard': (tp_size,),
                                          'opt_weight_shard_step': opt_weight_shard_step,
                                          'opt_weight_shard_size': opt_weight_shard_size}
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
        init_method (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The values
            of str refer to the function `initializer`. Default: 'normal'.
        bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        input_is_parallel (bool): Specifies whether the input tensor has already been sliced on last dimension.
        is_expert (bool): Specifies whether this linear layer is an expert. Default: False.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The values
            of str refer to the function `initializer`. Default: Zero().
        param_init_dtype (dtype.Number): The parameter initialization type. Default: None.
        compute_dtype (dtype.Number): The computation type. Default: None.
        transpose_b (bool): Specifies whether the weight parameter will be initialized as a transposed shape.
            Default: True.

    Inputs:
        - **input_** (Tensor) - Tensor of shape :math:`(*, output\_size)`. The `input_size` in `Args` should be equal
          to :math:`in\_channels` in `Inputs`.

    Outputs:
        - **output** (Tensor): Result of linear with shape :math:`(*, output\_size)`.
        - **output_bias** (Parameter): Bias parameter when `skip_bias_add=True` with shape :math:`(output\_size)`.

    Supported Platforms:
        ``Ascend``
    """
    def __init__(
            self,
            input_size,
            output_size,
            *,
            config,
            init_method,
            bias,
            input_is_parallel,
            skip_bias_add,
            stride=1,
            keep_master_weight_for_test=False,
            is_expert=False,
            tp_comm_buffer_name=None,
            bias_init=Zero(),
            param_init_dtype=None,
            compute_dtype=None,
            transpose_b=True,
        ):
        super(RowParallelLinear, self).__init__()
        if skip_bias_add:
            raise NotImplementedError("For ColumnParallelLinear, `skip_bias_add=True` is not supported for now.")
        if stride > 1:
            raise NotImplementedError("For ColumnParallelLinear, `stride > 1` is not supported for now, "
                                      "but got `stride={}`".format(stride))
        if keep_master_weight_for_test:
            raise NotImplementedError("For ColumnParallelLinear, `keep_master_weight_for_test=True` "
                                      "is not supported for now.")
        if is_expert:
            raise NotImplementedError("For RowParallelLinear, `is_expert=True` is not supported for now.")
        if tp_comm_buffer_name:
            raise NotImplementedError("For ColumnParallelLinear, `tp_comm_buffer_name` is not supported for now.")

        self.input_size = input_size
        self.output_size = output_size
        self.has_bias = bias
        self.input_is_parallel = input_is_parallel
        self.skip_bias_add = skip_bias_add

        tensor_parallel_group_size = get_tp_world_size()
        self.input_size_per_partition = divide(input_size, tensor_parallel_group_size)

        self.config = config
        self.param_init_dtype = param_init_dtype if param_init_dtype else self.config.param_init_dtype
        self.compute_dtype = compute_dtype if compute_dtype else self.config.compute_dtype
        self.is_expert = is_expert
        self.expert_parallel = self.config.parallel_config.expert_parallel > 1
        self.sequence_parallel = self.config.parallel_config.use_sequence_parallel
        self.use_zero3 = self.config.parallel_config.zero_level == 'z3'
        self.transpose_b = transpose_b
        if self.use_zero3:
            try:
                dp_size = get_dp_world_size()
            except AssertionError:
                raise RuntimeError("When using zero3 optimizer parallel. Data parallel communication "
                                   "need be initialized. Please check 'dp' in order when calling "
                                   "initialize_model_parallel.")
        if self.transpose_b:
            if self.use_zero3 and self.output_size % dp_size == 0:
                self.output_size = divide(self.output_size, dp_size)
        else:
            if self.use_zero3 and self.input_size_per_partition % dp_size == 0 and self.output_size % dp_size == 0:
                self.input_size_per_partition = divide(self.input_size_per_partition, dp_size)
            else:
                self.use_zero3 = False

        if self.sequence_parallel and not self.input_is_parallel:
            raise RuntimeError(
                "To enable `sequence_arallel`, `input_is_parallel` must be `True`"
            )

        mode = EXPERT_PARALLEL_GENERATOR if self.is_expert and self.expert_parallel else TENSOR_PARALLEL_GENERATOR
        weight_shape = (self.output_size, self.input_size_per_partition) if self.transpose_b \
                        else (self.input_size_per_partition, self.output_size)
        with get_rng_tracer().rng_fork(mode):
            self.weight = Parameter(
                initializer(
                    init_method,
                    weight_shape,
                    self.param_init_dtype,
                ),
                name="weight",
            )
        self.matmul = P.BatchMatMul(transpose_b=self.transpose_b)

        if self.has_bias:
            if self.use_zero3 and not self.transpose_b:
                self.output_size = divide(self.output_size, dp_size)
            self.bias = Parameter(
                initializer(bias_init, (self.output_size), self.param_init_dtype), name="bias"
            )

        self.explicit_expert_comm = self.is_expert and (
            self.sequence_parallel or self.expert_parallel
        )

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.cast = P.Cast()

        self.scatter_to_mp_region = ScatterToModelParallelRegion()
        self.reduce_scatter_to_sp_region = ReduceScatterToSequenceParallelRegion()
        self.reduce_from_mp_region = ReduceFromModelParallelRegion()

    def construct(self, input_):
        """construct method"""
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = self.scatter_to_mp_region(input_)

        origin_dtype = F.dtype(input_parallel)
        weight = self.cast(self.weight, self.compute_dtype)
        input_parallel = self.cast(input_parallel, self.compute_dtype)
        output_parallel = self.matmul(input_parallel, weight)
        if self.explicit_expert_comm:
            output = output_parallel
        elif self.sequence_parallel:
            output_parallel = output_parallel.swapaxes(0, 1).contiguous()
            output = self.reduce_scatter_to_sp_region(output_parallel)
            output = output.swapaxes(0, 1).contiguous()
        else:
            output = self.reduce_from_mp_region(output_parallel)

        if self.has_bias and not self.skip_bias_add:
            output = mint.add(output, self.cast(self.bias, self.compute_dtype))
        output = ops.cast(output, origin_dtype)
        output_bias = self.bias if self.has_bias and self.skip_bias_add else None

        return output, output_bias

    def sharded_state_dict(self):
        """provide the sharded state dict based on the config"""
        tp_size = get_tp_world_size()
        w_shard = (1, tp_size) if self.transpose_b else (tp_size, 1)
        state_dict = {}
        opt_weight_shard_step = get_tp_world_size() if self.use_zero3 else 0
        try:
            opt_weight_shard_size = get_dp_world_size() if self.use_zero3 else -1
        except AssertionError:
            opt_weight_shard_size = -1
        state_dict[self.weight.name] = {'shape': self.weight.shape,
                                        'shard': w_shard,
                                        'opt_weight_shard_step': opt_weight_shard_step,
                                        'opt_weight_shard_size': opt_weight_shard_size}
        if self.has_bias:
            state_dict[self.bias.name] = {'shape': self.bias.shape,
                                          'shard': (1,),
                                          'opt_weight_shard_step': opt_weight_shard_step,
                                          'opt_weight_shard_size': opt_weight_shard_size}
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
    """

    def __init__(
            self,
            num_embeddings,
            embedding_dim,
            *,
            init_method,
            reduce_scatter_embeddings=False,
            config,
            param_init_dtype=None
        ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.reduce_scatter_embeddings = reduce_scatter_embeddings
        self.param_init_dtype = param_init_dtype if param_init_dtype else config.param_init_dtype

        self.tensor_model_parallel_size = get_tp_world_size()

        (
            self.vocab_start_index,
            self.vocab_end_index,
        ) = self._vocab_range_from_global_vocab_size(
            self.num_embeddings, get_tp_rank(), self.tensor_model_parallel_size
        )
        self.num_embeddings_per_partition = (
            self.vocab_end_index - self.vocab_start_index
        )

        with get_rng_tracer().rng_fork():
            self.weight = Parameter(
                initializer(
                    init=init_method,
                    shape=(self.num_embeddings_per_partition, self.embedding_dim),
                    dtype=self.param_init_dtype,
                ),
                name="weight",
            )
        self.gather = ops.Gather()
        self.reduce_from_mp_region = ReduceFromModelParallelRegion()
        self.reduce_scatter_to_sp_region = ReduceScatterToSequenceParallelRegion()

    def construct(self, x):
        """ construct. """
        if self.tensor_model_parallel_size > 1:
            # Build the mask.
            input_mask = (x < self.vocab_start_index).astype(mstype.int32) | (
                x >= self.vocab_end_index
            ).astype(mstype.int32)
            input_mask = input_mask.astype(mstype.bool_)
            # Mask the input.
            masked_input = x.copy() - self.vocab_start_index
            # masked_input[input_mask] = 0
            masked_input = ops.masked_fill(
                masked_input, input_mask, Tensor(0, masked_input.dtype)
            )
        else:
            masked_input = x
        # Get the embeddings.
        output_parallel = self.gather(self.weight, masked_input, 0)
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:
            # output_parallel[input_mask, :] = 0.0
            output_parallel = ops.masked_fill(
                output_parallel,
                input_mask.expand_dims(2),
                Tensor(0.0, output_parallel.dtype),
            )

        if self.reduce_scatter_embeddings:
            output_parallel = output_parallel.swapaxes(0, 1).contiguous()
            output = self.reduce_scatter_to_sp_region(output_parallel)
            output = output.swapaxes(0, 1).contiguous()
        else:
            # Reduce across all the model parallel devices.
            output = self.reduce_from_mp_region(output_parallel)
        return output

    # pylint: disable=W0613
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
        tp_size = get_tp_world_size()
        w_shard = (tp_size, 1)
        state_dict = {}
        state_dict[self.weight.name] = {'shape': self.weight.shape,
                                        'shard': w_shard,
                                        'opt_weight_shard_step': 0,
                                        'opt_weight_shard_size': -1}

        return state_dict
