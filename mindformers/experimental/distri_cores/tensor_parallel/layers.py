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
from mindspore import Parameter, Tensor, nn, ops, mint, hal
from mindspore.common.initializer import initializer, Zero

from mindformers.experimental.distri_cores.create_comm import (
    get_tp_rank,
    get_tp_world_size,
    get_dp_world_size,
    get_tp_group,
    get_stream
)
from mindformers.experimental.distri_cores.tensor_parallel import (
    CopyToModelParallelRegion,
    GatherFromModelParallelRegion,
    ReduceFromModelParallelRegion,
    ReduceScatterToSequenceParallelRegion,
    ScatterToModelParallelRegion,
    GatherFromSequenceParallelRegion
)
from mindformers.experimental.distri_cores.utils import divide
from mindformers.experimental.distri_cores.random import (
    get_rng_tracer,
    TENSOR_PARALLEL_GENERATOR,
    EXPERT_PARALLEL_GENERATOR,
)

__all__ = [
    "ColumnParallelLinear",
    "RowParallelLinear",
    "VocabParallelEmbedding",
    "LinearWithGradAccumulationAndAsyncCommunication"
]


class LinearWithGradAccumulationAndAsyncCommunication(nn.Cell):
    r"""
    Linear execution with asynchronous communication in backprop.

    The gradient of weight is calculated simultaneously with
    all reduce communication of input gradient under tensor parallel condition.

    For sequence parallel, the calculation of weight gradient is overlapped with
    reduce scatter communication of input gradient.

    Args:
        bias (bool): Specifies whether the layer uses a bias vector.
        gradient_accumulation_fusion (bool): Specifies whether accumulate gradient in backprop.
        sequence_parallel (bool): Specifies whether sequence parallel is enabled.
        grad_output_buffer (Tensor): Buffer used to save output gradients.
        wgrad_deferral_limit (int): Limit on the number of micro-batches. Default: 0.
        allreduce_dgrad (bool): Specifies whether calculation and communication are overlapped.
        transpose_b (bool): use transposed weight shape for initialization and compute.

    """
    def __init__(
            self,
            bias,
            gradient_accumulation_fusion,
            sequence_parallel,
            grad_output_buffer=None,
            wgrad_deferral_limit=0,
            allreduce_dgrad=None,
            transpose_b=True,
        ):
        super(LinearWithGradAccumulationAndAsyncCommunication, self).__init__()
        if grad_output_buffer:
            raise NotImplementedError("For LinearWithGradAccumulationAndAsyncCommunication, "
                                      "`grad_output_buffer` is not supported for now.")
        if wgrad_deferral_limit != 0:
            raise NotImplementedError("For LinearWithGradAccumulationAndAsyncCommunication, "
                                      "`wgrad_deferral_limit != 0` is not supported for now.")
        self.use_bias = bias
        self.allreduce_dgrad = allreduce_dgrad
        self.sequence_parallel = sequence_parallel
        self.gradient_accumulation_fusion = gradient_accumulation_fusion
        self.transpose_b = transpose_b

        self.matmul = P.BatchMatMul(transpose_b=self.transpose_b)
        self.matmul_g_in = P.BatchMatMul(transpose_a=False, transpose_b=not self.transpose_b)
        self.matmul_g_w = P.BatchMatMul(transpose_a=True, transpose_b=False)
        if get_tp_world_size() > 1:
            self.gather_from_sp_region = ops.AllGather(group=get_tp_group())
            self.reduce_scatter_to_sp_region = ops.ReduceScatter(group=get_tp_group())
            self.reduce_from_mp_region = ops.AllReduce(group=get_tp_group())
        self.reduce_sum = P.ReduceSum()
        self.stream = get_stream()
        self.input_parallel = None
        self.weight_param = None

    # pylint: disable=C0111
    def construct(self, x, weight, bias, weight_param=None):
        if bias is None:
            self.use_bias = False
        self.weight_param = weight_param
        if self.sequence_parallel:
            self.input_parallel = x.swapaxes(0, 1).contiguous()
            self.input_parallel = self.gather_from_sp_region(self.input_parallel.contiguous())
            self.input_parallel = self.input_parallel.swapaxes(0, 1).contiguous()
        else:
            self.input_parallel = x

        output_parallel = self.matmul(self.input_parallel, weight)
        if self.use_bias:
            output_parallel = mint.add(
                output_parallel, bias
            )

        return output_parallel

    # pylint: disable=W0613, C0111
    def bprop(self, *args):
        dout = args[-1]
        weight = args[1]
        weight_param = args[3]
        grad_input = self.matmul_g_in(dout, weight).reshape(self.input_parallel.shape)

        if self.allreduce_dgrad:
            self.stream.wait_stream(hal.current_stream())
            with hal.StreamCtx(self.stream):
                grad_input = self.reduce_from_mp_region(grad_input)

        if self.sequence_parallel:
            assert not self.allreduce_dgrad
            grad_input = grad_input.swapaxes(0, 1).contiguous()
            self.stream.wait_stream(hal.current_stream())
            with hal.StreamCtx(self.stream):
                grad_input = self.reduce_scatter_to_sp_region(grad_input.contiguous())

        if self.transpose_b:
            grad_weight = self.matmul_g_w(dout, self.input_parallel)
        else:
            grad_weight = self.matmul_g_w(self.input_parallel, dout)

        if len(dout.shape) > 2:     # b,s,h
            grad_weight = self.reduce_sum(grad_weight, 0)
            dout = dout.sum(axis=0)

        grad_weight = grad_weight.reshape(weight.shape)
        grad_bias = dout.sum(axis=0) if self.use_bias else None

        if self.gradient_accumulation_fusion and self.weight_param is not None and \
            isinstance(self.weight_param, Parameter):
            if hasattr(self.weight_param, 'grad_accumulated'):
                origin_dtype = None
                if grad_weight.dtype != self.weight_param.dtype:
                    grad_weight = ops.cast(grad_weight, self.weight_param.dtype)
                    origin_dtype = grad_weight.dtype
                self.weight_param.grad_view[:] = mint.add(self.weight_param.grad_view, grad_weight)
                self.weight_param.grad_accumulated = True
                if origin_dtype:
                    grad_weight = ops.cast(grad_weight, origin_dtype)

        if self.sequence_parallel or self.allreduce_dgrad:
            hal.current_stream().wait_stream(self.stream)

        if self.sequence_parallel:
            grad_input = grad_input.swapaxes(0, 1).contiguous()

        grad_weight_param = F.full(weight_param.shape,
                                   0, dtype=weight_param.dtype) if weight_param is not None else None

        return grad_input, grad_weight, grad_bias, grad_weight_param


class LinearWithFrozenWeight(nn.Cell):
    r"""
    Linear execution with frozen weight.

    The gradient of weight is not calculated during backward propagation.

    Args:
        bias (bool): Specifies whether the layer uses a bias vector.
        allreduce_dgrad (bool): Specifies whether calculation and communication are overlapped.

    """
    def __init__(self, bias, allreduce_dgrad, transpose_b=True):
        super(LinearWithFrozenWeight, self).__init__()
        self.bias = bias
        self.allreduce_dgrad = allreduce_dgrad
        self.transpose_b = transpose_b
        self.matmul = P.BatchMatMul(transpose_b=self.transpose_b)
        self.matmul_g_in = P.BatchMatMul(transpose_a=False, transpose_b=not self.transpose_b)
        if get_tp_world_size() > 1:
            self.reduce_from_mp_region = ops.AllReduce(group=get_tp_group())
        if bias:
            self.bias_add = P.Add()

    def construct(self, input_, weight, bias):
        output = self.matmul(input_, weight)
        if self.bias and bias is not None:
            output = self.bias_add(output, bias)
        return output

    # pylint: disable=W0613
    def bprop(self, x, weight, bias, out, dout):
        grad_input = self.matmul_g_in(dout, weight).reshape(x.shape)
        grad_bias = F.full(bias.shape, 0, dtype=bias.dtype) if self.bias else None
        if self.allreduce_dgrad:
            grad_input = self.reduce_from_mp_region(grad_input)
        return grad_input, None, grad_bias


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
        self.allreduce_dgrad = (
            tensor_parallel_group_size > 1 and not self.sequence_parallel and not disable_grad_reduce
        )
        if self.allreduce_dgrad and self.sequence_parallel:
            raise RuntimeError(
                "`allreduce_dgrad` and `sequence_parallel` cannot be enabled at the same time."
            )
        self.gradient_accumulation_fusion = config.parallel_config.gradient_accumulation_fusion
        self.forward_impl_ = LinearWithGradAccumulationAndAsyncCommunication(
            bias=(self.has_bias and not self.skip_bias_add),
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            sequence_parallel=False if self.explicit_expert_comm else self.sequence_parallel,
            allreduce_dgrad=False if self.explicit_expert_comm else self.allreduce_dgrad,
            transpose_b=self.transpose_b
        )

        self.frozen_weight_forward_impl_ = LinearWithFrozenWeight(
            bias=(self.has_bias and not self.skip_bias_add),
            allreduce_dgrad=False if self.explicit_expert_comm else self.allreduce_dgrad,
            transpose_b=self.transpose_b
        )

    def construct(self, input_, weight=None):
        """construct method."""
        if weight is None and self.skip_weight_param_allocation:
            raise ValueError("For ColumnParallelLinear, when skip_weight_param_allocation=True,"
                             " weight should be passed to construct(), but got None.")

        if weight is not None and not self.skip_weight_param_allocation:
            raise ValueError("For ColumnParallelLinear, when skip_weight_param_allocation=False,"
                             "weight should not be passed to construct(), but got {}".format(weight))

        if (
                self.sequence_parallel
                or self.explicit_expert_comm
                or self.allreduce_dgrad
        ):
            input_parallel = input_
        else:
            input_parallel = self.copy_to_mp_region(input_)

        origin_dtype = F.dtype(input_parallel)
        if self.skip_weight_param_allocation:
            weight_requires_grad = not isinstance(weight, Parameter) or weight.requires_grad
            weight_param = weight
            weight = ops.cast(weight, self.compute_dtype)
        else:
            weight_requires_grad = self.weight.requires_grad
            weight_param = self.weight
            weight = ops.cast(self.weight, self.compute_dtype)
        input_parallel = ops.cast(input_parallel, self.compute_dtype)

        bias = ops.cast(self.bias, self.compute_dtype) if self.has_bias and not self.skip_bias_add else None

        if not weight_requires_grad:
            if self.sequence_parallel:
                input_parallel = input_parallel.swapaxes(0, 1).contiguous()
                input_parallel = self.gather_from_sp_region(input_parallel)
                input_parallel = input_parallel.swapaxes(0, 1).contiguous()
            output_parallel = self.frozen_weight_forward_impl_(input_parallel, weight, bias)
        else:
            output_parallel = self.forward_impl_(input_parallel, weight, bias, weight_param=weight_param)

        output_parallel = ops.cast(output_parallel, origin_dtype)

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
        self.gradient_accumulation_fusion = config.parallel_config.gradient_accumulation_fusion

        self.forward_impl_ = LinearWithGradAccumulationAndAsyncCommunication(
            bias=False,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            sequence_parallel=False,
            allreduce_dgrad=False,
            transpose_b=self.transpose_b
        )

        self.frozen_weight_forward_impl_ = LinearWithFrozenWeight(
            bias=False,
            allreduce_dgrad=False,
            transpose_b=self.transpose_b
        )

    def construct(self, input_):
        """construct method"""
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = self.scatter_to_mp_region(input_)

        origin_dtype = F.dtype(input_parallel)
        weight = self.cast(self.weight, self.compute_dtype)
        weight_param = self.weight if self.gradient_accumulation_fusion else None
        input_parallel = ops.cast(input_parallel, self.compute_dtype)
        if self.weight.requires_grad:
            output_parallel = self.forward_impl_(input_parallel, weight, bias=None, weight_param=weight_param)
        else:
            output_parallel = self.frozen_weight_forward_impl_(input_parallel, weight, bias=None)
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
