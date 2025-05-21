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
Linear units for tensor parallelism.
"""
__all__ = [
    "RowParallelLinear",
]

from typing import Callable

try:
    from mindformer._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator
from mindspore import nn, Tensor
from mindspore.context import ParallelMode
from mindspore.common import dtype
from mindspore.common.parameter import Parameter
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindspore.ops.auto_generate import Cast, MatMulExt, AddExt, Reshape, Transpose, IndexSelect

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.utils.init_method import init_method_zero



class RowParallelLinear(nn.Cell):
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
                 bias: bool = True,
                 input_is_parallel: bool = False,
                 skip_bias_add: bool = False,
                 stride: int = 1,
                 keep_master_weight_for_test: bool = False,
                 is_expert: bool = False,
                 tp_comm_buffer_name: str = None,
                 transpose_b: bool = True,
                 compute_dtype: dtype = dtype.float16,
                 bias_init: Callable = None
                 ):
        super(RowParallelLinear, self).__init__()
        if input_is_parallel:
            raise NotImplementedError("For RowParallelLinear, `input_is_parallel` is not supported for now")
        if stride > 1:
            raise NotImplementedError("For RowParallelLinear, `stride > 1` is not supported for now")
        if keep_master_weight_for_test:
            raise NotImplementedError("For RowParallelLinear, `keep_master_weight_for_test` is not supported for now")
        if is_expert:
            raise NotImplementedError("For RowParallelLinear, `is_expert` is not supported for now")
        if tp_comm_buffer_name is not None:
            raise NotImplementedError("For RowParallelLinear, `tp_comm_buffer_name` is not supported for now")
        self.input_size = input_size
        self.output_size = output_size
        self.config = config
        self.init_method = init_method
        self.transpose_b = transpose_b
        self.skip_bias_add = skip_bias_add
        self.compute_dtype = compute_dtype
        self.params_dtype = config.params_dtype
        self.cast = Cast()
        self.has_bias = bias

        self.transpose = Transpose()

        weight_shape = (output_size, input_size) if transpose_b else (input_size, output_size)
        self.weight = Parameter(init_method(weight_shape), name='weight')

        if self.has_bias:
            if bias_init is None:
                bias_init = init_method_zero(self.params_dtype)
            self.bias = Parameter(bias_init((output_size,)), name='bias')
        else:
            self.bias = None

        self.matmul = MatMulExt()

        if not skip_bias_add:
            self.add = AddExt()
        self.reshape = Reshape()

        if config is not None:
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
        output_shape = input_.shape[:-1] + (self.output_size,)
        input_ = self.reshape(input_, (-1, self.input_size))

        ori_dtype = input_.dtype
        weight = self.cast(self.weight, self.compute_dtype)
        input_ = self.cast(input_, self.compute_dtype)

        if self.transpose_b:
            weight = self.transpose(weight, (1, 0))

        input_ = self.matmul(input_, weight)

        if not self.skip_bias_add and self.has_bias:
            bias = self.cast(self.bias, self.compute_dtype)
            input_ = self.add(input_, bias)
            bias = None
        else:
            bias = self.bias

        input_ = self.cast(input_, ori_dtype)
        output = self.reshape(input_, output_shape)
        return output, bias

    def shard(self, config: TransformerConfig) -> None:
        """Shard the operators in RowParallelLinear.

        Args:
            config (TransformerConfig): The config of the transformer model.
        """
        dp = config.data_parallel_size if config.data_parallel_size is not None else 1
        cp = config.context_parallel_size if config.context_parallel_size is not None else 1
        tp = config.tensor_model_parallel_size if config.tensor_model_parallel_size is not None else 1

        matmul_in_strategy = ((dp * cp, tp), (tp, 1))
        self.matmul.shard(in_strategy=matmul_in_strategy)
        if self.transpose_b:
            self.transpose.shard(((1, tp),))

        if not self.skip_bias_add:
            add_in_strategy = ((dp * cp, 1), (1,))
            self.add.shard(in_strategy=add_in_strategy)

    def sharding_propagation(self, config: TransformerConfig) -> None:
        """Shard the operators in RowParallelLinear in sharding propagation mode.

        Args:
            config (TransformerConfig): The config of the transformer model.
        """
        dp = config.data_parallel_size if config.data_parallel_size is not None else 1
        cp = config.context_parallel_size if config.context_parallel_size is not None else 1
        tp = config.tensor_model_parallel_size if config.tensor_model_parallel_size is not None else 1

        if self.transpose_b:
            weight_strategy = (1, tp)
        else:
            weight_strategy = (tp, 1)
        matmul_in_strategy = ((dp * cp, tp), weight_strategy)
        self.matmul.shard(in_strategy=matmul_in_strategy)
