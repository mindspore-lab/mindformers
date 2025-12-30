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
"""Linear units for tensor parallelism."""
__all__ = [
    "Linear",
]

from typing import Callable

from mindspore import nn, Tensor, mint, ops
from mindspore.common.parameter import Parameter

from mindformers.models.utils import convert_mstype
from mindformers.parallel_core.utils.init_method import init_method_zero


class Linear(nn.Cell):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Args:
        input_size (int): The number of input units.
        output_size (int): The number of output units.
        compute_dtype (str): The data type of the computation (e.g., 'bf16', 'float16').
        params_dtype (str): The data type of the parameters (e.g., 'float32').
        init_method (Callable): The initialization method. Default: None.
        bias (bool): Whether to include bias in the linear layer. Default: True.
        skip_bias_add (bool): Whether to skip bias add. Default: False.
        skip_weight_param_allocation (bool): Whether to skip weight parameter allocation. Default: False.
        bias_init (Callable): The initialization method for bias. Default: None.
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 compute_dtype: str,
                 params_dtype: str,
                 init_method: Callable = None,
                 bias: bool = True,
                 skip_bias_add: bool = False,
                 skip_weight_param_allocation: bool = False,
                 bias_init: Callable = None
                 ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.init_method = init_method
        self.skip_bias_add = skip_bias_add
        self.skip_weight_param_allocation = skip_weight_param_allocation
        self.has_bias = bias
        self.params_dtype = convert_mstype(params_dtype)
        self.compute_dtype = convert_mstype(compute_dtype)

        # use_cpu_initialization configuration is not supported for now.
        if skip_weight_param_allocation:
            self.weight = None
        else:
            # Weight is stored as (output_size, input_size) and transposed at runtime
            weight_shape = (output_size, input_size)
            self.weight = Parameter(init_method(weight_shape), name='weight')

        if self.has_bias:
            if bias_init is None:
                bias_init = init_method_zero(self.params_dtype)
            self.bias = Parameter(bias_init((output_size,)), name='bias')
        else:
            self.bias = None

        self.matmul = mint.matmul
        self.transpose = mint.transpose
        self.cast = ops.cast
        if not skip_bias_add:
            self.add = mint.add

    def construct(self, input_: Tensor, weight: Tensor = None) -> tuple[Tensor, Tensor]:
        """Forward of Linear.

        Args:
            input_ (Tensor): The input tensor.
            weight (Tensor): The weight tensor. Default: None.

        Returns:
            - output (Tensor): The output tensor.
            - bias (Tensor): The bias
        """
        if weight is None:
            if self.skip_weight_param_allocation:
                raise ValueError("For Linear, when `skip_weight_param_allocation` is enabled,"
                                 " `weight` is required, but got None")
            weight = self.weight

        ori_dtype = input_.dtype

        weight = self.cast(weight, self.compute_dtype)
        input_ = self.cast(input_, self.compute_dtype)

        # Transpose weight from (output_size, input_size) to (input_size, output_size)
        weight = self.transpose(weight, 1, 0)

        # Directly use 3D input: (batch, seq, input_size) @ (input_size, output_size) -> (batch, seq, output_size)
        output = self.matmul(input_, weight)

        if not self.skip_bias_add and self.has_bias:
            bias = self.cast(self.bias, self.compute_dtype)
            output = self.add(output, bias)
            bias = None
        else:
            bias = self.bias

        output = self.cast(output, ori_dtype)

        return output, bias
