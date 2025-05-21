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
"""Batched linear units for tensor parallelism."""
__all__ = ["ColumnParallelBatchedLinear", "RowParallelBatchedLinear"]

from typing import List, Optional, Callable

from mindspore import nn, Tensor
from mindspore.context import ParallelMode
from mindspore.common import dtype
from mindspore.common.parameter import Parameter
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindspore.ops.auto_generate import Cast, BatchMatMulExt, Reshape, Transpose

from mindformers.parallel_core.transformer_config import TransformerConfig


class ColumnParallelBatchedLinear(nn.Cell):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Args:
        input_size (int): The number of input units.
        output_size (int): The number of output units.
        config (TransformerConfig): The config of the transformer model.
        bias_init (str): The initialization method for bias. Default: 'zeros'.
        bias (bool): Whether to include bias in the linear layer. Default: True.
        gather_output (bool): Whether to gather the output. Default: False.
        stride (int): The stride of the linear layer. Default: 1.
        keep_master_weight_for_test (bool): Whether to keep master weight for test. Default: False.
        skip_bias_add (bool): Whether to skip bias add. Default: False.
        skip_weight_param_allocation (bool): Whether to skip weight parameter allocation. Default: False.
        embedding_activation_buffer (List[Tensor]): The buffer for embedding activation. Default: None.
        grad_output_buffer (List[Tensor]): The buffer for gradient output. Default: None.
        is_expert (bool): Whether to use expert mode. Default: False.
        tp_comm_buffer_name (str): The name of the tensor parallel communication buffer. Default: None.
        disable_grad_reduce (bool): Whether to disable gradient reduce. Default: False.
        transpose_b (bool): Whether to transpose the weight matrix. Default: True.
        compute_dtype (dtype): The data type of the computation. Default: dtype.float16.
        init_method (Callable): The initialization method. Default: None
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 config: TransformerConfig,
                 init_method: Callable = None,
                 bias: bool = False,
                 gather_output: bool = False,
                 stride: int = 1,
                 keep_master_weight_for_test: bool = False,
                 skip_bias_add: bool = False,
                 skip_weight_param_allocation: bool = False,
                 embedding_activation_buffer: Optional[List[Tensor]] = None,
                 grad_output_buffer: Optional[List[Tensor]] = None,
                 is_expert: bool = True,
                 tp_comm_buffer_name: str = None,
                 disable_grad_reduce: bool = False,
                 transpose_b: bool = True,
                 compute_dtype: dtype = dtype.float16,
                 bias_init: Callable = None
                 ):
        super(ColumnParallelBatchedLinear, self).__init__()
        if gather_output:
            raise NotImplementedError("For ColumnParallelBatchedLinear, `gather_output` is not supported for now")
        if stride > 1:
            raise NotImplementedError("For ColumnParallelBatchedLinear, `stride > 1` is not supported for now")
        if keep_master_weight_for_test:
            raise NotImplementedError(
                "For ColumnParallelBatchedLinear, `keep_master_weight_for_test` is not supported for now")
        if embedding_activation_buffer is not None:
            raise NotImplementedError(
                "For ColumnParallelBatchedLinear, `embedding_activation_buffer` is not supported for now")
        if grad_output_buffer is not None:
            raise NotImplementedError("For ColumnParallelBatchedLinear, `grad_output_buffer` is not supported for now")
        if tp_comm_buffer_name is not None:
            raise NotImplementedError("For ColumnParallelBatchedLinear, `tp_comm_buffer_name` is not supported for now")
        if disable_grad_reduce:
            raise NotImplementedError("For ColumnParallelBatchedLinear, `disable_grad_reduce` is not supported for now")
        if bias:
            raise NotImplementedError("For ColumnParallelBatchedLinear, `bias` is not supported for now")
        if skip_bias_add:
            raise NotImplementedError("For ColumnParallelBatchedLinear, `skip_bias_add` is not supported for now")
        if bias_init:
            raise NotImplementedError("For ColumnParallelBatchedLinear, `bias_init` is not supported for now")
        self.input_size = input_size
        self.output_size = output_size
        self.config = config
        self.init_method = init_method
        self.compute_dtype = compute_dtype
        self.cast = Cast()
        self.transpose_b = transpose_b
        self.skip_weight_param_allocation = skip_weight_param_allocation
        self.params_dtype = config.params_dtype
        self.bias = None
        # expert config
        self.expert_flag = is_expert and config.num_moe_experts > 1
        if not self.expert_flag:
            raise ValueError("For ColumnParallelBatchedLinear, `is_expert` should be True "
                             "and `num_moe_experts` should be larger than 1")
        self.num_moe_experts = config.num_moe_experts
        dp = config.data_parallel_size if config.data_parallel_size is not None else 1
        tp = config.tensor_model_parallel_size if config.tensor_model_parallel_size is not None else 1
        cp = config.context_parallel_size if config.context_parallel_size is not None else 1
        ep = config.expert_model_parallel_size if config.expert_model_parallel_size is not None else 1
        etp = config.expert_tensor_parallel_size if config.expert_tensor_parallel_size is not None else 1
        self.dp_moe = dp * cp // ep
        self.etp_flag = etp != 1
        if self.etp_flag:
            self.tp_moe = etp
            self.dp_moe = dp * cp * tp // (ep * self.tp_moe)
        self.outer_batch = self.dp_moe

        if self.skip_weight_param_allocation:
            self.weight = None
        else:
            weight_shape = [output_size, input_size] if transpose_b else [input_size, output_size]
            # add expert dimension
            weight_shape = [config.num_moe_experts] + weight_shape
            self.weight = Parameter(init_method(weight_shape), name='weight')

        self.bmm = BatchMatMulExt()
        self.transpose = Transpose()
        self.in_transpose = Transpose()
        self.out_transpose = Transpose()
        self.reshape = Reshape()

        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.sharding_propagation(config)
        else:
            self.shard(config)

    def construct(self, input_: Tensor, weight: Tensor = None) -> tuple[Tensor, Tensor]:
        """Forward of ColumnParallelBatchedLinear.

        Args:
            input_ (Tensor): The input tensor.
            weight (Tensor): The weight tensor. Default: None.

        Returns:
            - output (Tensor): The output tensor.
            - bias (Tensor): The bias
        """
        # [S, B, H] -> [B, S, H]
        input_ = self.in_transpose(input_, (1, 0, 2))
        output_shape = input_.shape[:-1] + (self.output_size,)
        input_ = self.reshape(input_, (-1, self.input_size))

        ori_dtype = input_.dtype
        if weight is None:
            if self.skip_weight_param_allocation:
                raise ValueError("For ColumnParallelBatchedLinear, when `skip_weight_param_allocation` is enabled,"
                                 " `weight` is required, but got None")
            weight = self.weight
        weight = self.cast(weight, self.compute_dtype)
        input_ = self.cast(input_, self.compute_dtype)

        input_ = self.reshape(input_, (self.outer_batch * self.num_moe_experts, -1, self.input_size))
        if self.transpose_b:
            weight = self.transpose(weight, (0, 2, 1))
        input_ = self.bmm(input_, weight)

        bias = self.bias
        input_ = self.cast(input_, ori_dtype)
        output = self.reshape(input_, output_shape)
        output = self.out_transpose(output, (1, 0, 2))
        return output, bias

    def shard(self, config: TransformerConfig) -> None:
        """Shard the operators in ColumnParallelBatchedLinear.

        Args:
            config (TransformerConfig): The config of the transformer model.
        """
        dp = config.data_parallel_size if config.data_parallel_size is not None else 1
        tp = config.tensor_model_parallel_size if config.tensor_model_parallel_size is not None else 1
        cp = config.context_parallel_size if config.context_parallel_size is not None else 1
        ep = config.expert_model_parallel_size if config.expert_model_parallel_size is not None else 1

        if self.etp_flag:
            dp = self.dp_moe * tp // self.tp_moe
            tp = self.tp_moe
        else:
            dp = self.dp_moe
        if self.transpose_b:
            self.bmm.shard(((dp * ep, 1, 1), (ep, tp, 1)))
        else:
            self.bmm.shard(((dp * ep, 1, 1), (ep, 1, tp)))

        transpose_in_strategy = ((ep, 1, tp),)
        self.transpose.shard(in_strategy=transpose_in_strategy)
        self.in_transpose.shard(((cp, dp, tp),))
        self.out_transpose.shard(((dp, cp, tp),))

    def sharding_propagation(self, config: TransformerConfig) -> None:
        """Shard the operators in ColumnParallelBatchedLinear in sharding propagation mode.

        Args:
            config (TransformerConfig): The config of the transformer model.
        """
        tp = config.tensor_model_parallel_size if config.tensor_model_parallel_size is not None else 1
        ep = config.expert_model_parallel_size if config.expert_model_parallel_size is not None else 1

        if self.etp_flag:
            dp = self.dp_moe * tp // self.tp_moe
            tp = self.tp_moe
        else:
            dp = self.dp_moe

        if self.transpose_b:
            self.bmm.shard(((dp * ep, 1, 1), (ep, tp, 1)))
        else:
            self.bmm.shard(((dp * ep, 1, 1), (ep, 1, tp)))


class RowParallelBatchedLinear(nn.Cell):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along its first dimension and X along
    its second dimension. A = transpose([A_1 .. A_p]) X = [X_1, ..., X_p]

    Args:
        input_size (int): The number of input units.
        output_size (int): The number of output units.
        config (TransformerConfig): The config of the transformer model.
        bias_init (str): The initialization method for bias. Default: 'zeros'.
        bias (bool): Whether to include bias in the linear layer. Default: True.
        input_is_parallel (bool): Whether the input is already parallelized. Default: False.
        skip_bias_add (bool): Whether to skip bias add. Default: False.
        stride (int): The stride of the linear layer. Default: 1.
        keep_master_weight_for_test (bool): Whether to keep master weight for test. Default: False.
        is_expert (bool): Whether to use expert mode. Default: False.
        tp_comm_buffer_name (str): The name of the tensor parallel communication buffer. Default: None.
        transpose_b (bool): Whether to transpose the weight matrix. Default: True.
        compute_dtype (dtype): The data type of the computation. Default: dtype.float16.
        init_method (Callable): The initialization method. Default: None
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 config: TransformerConfig,
                 init_method: Callable = None,
                 bias: bool = False,
                 input_is_parallel: bool = False,
                 skip_bias_add: bool = False,
                 stride: int = 1,
                 keep_master_weight_for_test: bool = False,
                 is_expert: bool = True,
                 tp_comm_buffer_name: str = None,
                 transpose_b: bool = True,
                 compute_dtype: dtype = dtype.float16,
                 bias_init: Callable = None
                 ):
        super(RowParallelBatchedLinear, self).__init__()
        if input_is_parallel:
            raise NotImplementedError("For RowParallelBatchedLinear, `input_is_parallel` is not supported for now")
        if stride > 1:
            raise NotImplementedError("For RowParallelBatchedLinear, `stride > 1` is not supported for now")
        if keep_master_weight_for_test:
            raise NotImplementedError("For RowParallelBatchedLinear, `keep_master_weight_for_test` "
                                      "is not supported for now")
        if tp_comm_buffer_name is not None:
            raise NotImplementedError("For RowParallelBatchedLinear, `tp_comm_buffer_name` is not supported for now")
        if bias:
            raise NotImplementedError("For ColumnParallelBatchedLinear, `bias` is not supported for now")
        if skip_bias_add:
            raise NotImplementedError("For ColumnParallelBatchedLinear, `skip_bias_add` is not supported for now")
        if bias_init:
            raise NotImplementedError("For ColumnParallelBatchedLinear, `bias_init` is not supported for now")
        self.input_size = input_size
        self.output_size = output_size
        self.config = config
        self.init_method = init_method
        self.transpose_b = transpose_b
        self.compute_dtype = compute_dtype
        self.params_dtype = config.params_dtype
        self.cast = Cast()
        self.bias = None
        # expert config
        self.expert_flag = is_expert and config.num_moe_experts > 1
        if not self.expert_flag:
            raise ValueError("For RowParallelBatchedLinear, `is_expert` should be True "
                             "and `num_moe_experts` should be larger than 1")
        self.num_moe_experts = config.num_moe_experts
        dp = config.data_parallel_size if config.data_parallel_size is not None else 1
        tp = config.tensor_model_parallel_size if config.tensor_model_parallel_size is not None else 1
        cp = config.context_parallel_size if config.context_parallel_size is not None else 1
        ep = config.expert_model_parallel_size if config.expert_model_parallel_size is not None else 1
        etp = config.expert_tensor_parallel_size if config.expert_tensor_parallel_size is not None else 1
        self.dp_moe = dp * cp // ep
        self.etp_flag = etp != 1
        if self.etp_flag:
            self.tp_moe = etp
            self.dp_moe = dp * cp * tp // (ep * self.tp_moe)
        self.outer_batch = self.dp_moe

        weight_shape = [output_size, input_size] if transpose_b else [input_size, output_size]
        # add expert dimension
        weight_shape = [config.num_moe_experts] + weight_shape
        self.weight = Parameter(init_method(weight_shape), name='weight')

        self.bmm = BatchMatMulExt()
        self.transpose = Transpose()
        self.in_transpose = Transpose()
        self.out_transpose = Transpose()
        self.reshape = Reshape()

        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.sharding_propagation(config)
        else:
            self.shard(config)

    def construct(self, input_: Tensor) -> tuple[Tensor, Tensor]:
        """Forward of RowParallelLinear.

        Args:
            input_ (Tensor): The input tensor.

        Returns:
            - output (Tensor): The output tensor.
            - bias (Tensor): The bias
        """
        # [S, B, H] -> [B, S, H]
        input_ = self.in_transpose(input_, (1, 0, 2))
        output_shape = input_.shape[:-1] + (self.output_size,)
        input_ = self.reshape(input_, (-1, self.input_size))

        ori_dtype = input_.dtype
        weight = self.cast(self.weight, self.compute_dtype)
        input_ = self.cast(input_, self.compute_dtype)

        input_ = self.reshape(input_, (self.outer_batch * self.num_moe_experts, -1, self.input_size))
        if self.transpose_b:
            weight = self.transpose(weight, (0, 2, 1))
        input_ = self.bmm(input_, weight)

        bias = self.bias
        input_ = self.cast(input_, ori_dtype)
        output = self.reshape(input_, output_shape)
        output = self.out_transpose(output, (1, 0, 2))
        return output, bias

    def shard(self, config: TransformerConfig) -> None:
        """Shard the operators in ColumnParallelBatchedLinear.

        Args:
            config (TransformerConfig): The config of the transformer model.
        """
        dp = config.data_parallel_size if config.data_parallel_size is not None else 1
        tp = config.tensor_model_parallel_size if config.tensor_model_parallel_size is not None else 1
        cp = config.context_parallel_size if config.context_parallel_size is not None else 1
        ep = config.expert_model_parallel_size if config.expert_model_parallel_size is not None else 1

        if self.etp_flag:
            dp = self.dp_moe * tp // self.tp_moe
            tp = self.tp_moe
        else:
            dp = self.dp_moe
        if self.transpose_b:
            self.bmm.shard(((dp * ep, 1, 1), (ep, 1, tp)))
        else:
            self.bmm.shard(((dp * ep, 1, tp), (ep, tp, 1)))

        self.transpose.shard(((ep, 1, tp),))
        self.in_transpose.shard(((cp, dp, tp),))
        self.out_transpose.shard(((dp, cp, tp),))

    def sharding_propagation(self, config: TransformerConfig) -> None:
        """Shard the operators in RowParallelLinear in sharding propagation mode.

        Args:
            config (TransformerConfig): The config of the transformer model.
        """
        dp = config.data_parallel_size if config.data_parallel_size is not None else 1
        tp = config.tensor_model_parallel_size if config.tensor_model_parallel_size is not None else 1
        ep = config.expert_model_parallel_size if config.expert_model_parallel_size is not None else 1

        if self.etp_flag:
            dp = self.dp_moe * tp // self.tp_moe
            tp = self.tp_moe
        else:
            dp = self.dp_moe

        if self.transpose_b:
            self.bmm.shard(((dp * ep, 1, 1), (ep, 1, tp)))
        else:
            self.bmm.shard(((dp * ep, 1, tp), (ep, tp, 1)))
