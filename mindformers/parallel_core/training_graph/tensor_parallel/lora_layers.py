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
"""mindformers lora layer"""

from typing import List, Optional, Callable

from mindspore.common import dtype
from mindspore.common.initializer import initializer
from mindspore.common.tensor import Tensor
from mindspore.context import ParallelMode
from mindspore.ops.auto_generate import Cast, AddExt, Reshape, Mul
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

try:
    from mindspore._checkparam import Validator, Rel

    INC_LEFT = Rel.INC_LEFT
except ImportError:
    import mindspore._checkparam as Validator

    INC_LEFT = Validator.INC_LEFT

from mindformers.parallel_core.model_parallel_config import ModelParallelConfig
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.training_graph.transformer.dropout import Dropout
from mindformers.parallel_core.training_graph.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    LinearNoTP,
)

__all__ = ["ColumnParallelLinearWithLoRA", "RowParallelLinearWithLoRA"]


def init_lora_method(method, param_init_dtype: dtype = dtype.float32):
    """Init method based on N(0, sigma)."""

    def init_(tensor_shape) -> Tensor:
        return initializer(method, tensor_shape, param_init_dtype)

    return init_


# pylint: disable=C0103
class ColumnParallelLinearWithLoRA(ColumnParallelLinear):
    r"""
    The LoRA layer with weight sliced in column dimension.

    Args:
        input_size (int): The number of input units.
        output_size (int): The number of output units.
        config (TransformerConfig): The config of the transformer model.
        init_method (Callable): The initialization method. Default: None
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
        bias_init (str): The initialization method for bias. Default: 'zeros'.
        lora_rank (int): The value of rank. Default: 8.
        lora_alpha (int):  The alpha value for lora. Default: 32.
        lora_dropout (float):  The dropout rate for lora. Default: 0.0.
        lora_a_init (Union[Tensor, str, Initializer, numbers.Number]):  The trainable lora_a weight_init parameter.
            The values of str refer to the function `initializer`. Default: 'normal'.
        lora_b_init (Union[Tensor, str, Initializer, numbers.Number]):  The trainable lora_b weight_init parameter.
            The values of str refer to the function `initializer`. Default: 'zeros'.

    Inputs:
        - **input_** (Tensor) - Tensor of shape :math:`(*, input\_size)`. The `input_size` in `Args` should be equal
          to :math:`input\_size` in `Inputs`.
        - **weight** (Tensor) - Tensor of shape :math:`(input\_size, output\_size)`/:math`(output\_size, input\_size)`.
          Default: None.

    Outputs:
        - **output** (Tensor): Result of linear with shape :math:`(*, output\_size)`.
        - **output_bias** (Tensor): Bias parameter when `skip_bias_add=True` with shape :math:`(output\_size)`.

    Raises:
        ValueError: `skip_weight_param_allocation=True` but weight_tensor is not passed to construct function.

    """

    def __init__(
            self,
            input_size: int,
            output_size: int,
            config: TransformerConfig,
            init_method: Callable = None,
            bias: bool = True,
            gather_output: bool = False,
            stride: int = 1,
            keep_master_weight_for_test: bool = False,
            skip_bias_add: bool = False,
            skip_weight_param_allocation: bool = False,
            embedding_activation_buffer: Optional[List[Tensor]] = None,
            grad_output_buffer: Optional[List[Tensor]] = None,
            is_expert: bool = False,
            tp_comm_buffer_name: str = None,
            disable_grad_reduce: bool = False,
            transpose_b: bool = True,
            compute_dtype: dtype = dtype.float16,
            bias_init: Callable = None,
            lora_rank: int = 8,
            lora_alpha: int = 32,
            lora_dropout: float = 0.0,
            lora_a_init="normal",
            lora_b_init="zeros",
        ):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            config=config,
            init_method=init_method,
            bias=bias,
            gather_output=gather_output,
            stride=stride,
            keep_master_weight_for_test=keep_master_weight_for_test,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=skip_weight_param_allocation,
            embedding_activation_buffer=embedding_activation_buffer,
            grad_output_buffer=grad_output_buffer,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            disable_grad_reduce=disable_grad_reduce,
            transpose_b=transpose_b,
            compute_dtype=compute_dtype,
            bias_init=bias_init,
        )
        # LoRA parameters
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = Dropout(drop_prob=lora_dropout)
        self.scaling = self.lora_alpha / self.lora_rank

        # LoRA layers
        self.lora_A = LinearNoTP(
            input_size=input_size,
            output_size=lora_rank,
            config=config,
            init_method=init_lora_method(lora_a_init, config.params_dtype),
            skip_bias_add=True,
            bias=False,
            compute_dtype=self.compute_dtype,
        )
        self.lora_B = ColumnParallelLinear(
            input_size=lora_rank,
            output_size=output_size,
            config=config,
            init_method=init_lora_method(lora_b_init, config.params_dtype),
            skip_bias_add=True,
            bias=False,
            compute_dtype=self.compute_dtype,
        )

        # Operations
        self.cast = Cast()
        self.reshape = Reshape()
        self.add = AddExt()
        self.mul = Mul()

        if config is not None:
            if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
                self.sharding_propagation(config)
            else:
                self.shard_lora(config)

    def construct(self, input_: Tensor, weight: Tensor = None) -> tuple[Tensor, Tensor]:
        """Forward of ColumnParallelLinearWithLoRA.

        Args:
            input_ (Tensor): The input tensor.

        Returns:
            - output (Tensor): The output tensor.
            - bias (Optional[Tensor]): The bias
        """
        # Get original output from base_layer
        base_output, bias = super().construct(input_, weight)

        # Data type operation
        ori_dtype = input_.dtype

        # Shape operations
        x_shape = input_.shape

        # LoRA result
        input_dropped = self.lora_dropout(input_)
        lora_intermediate, _ = self.lora_A(input_dropped)
        lora_result, _ = self.lora_B(lora_intermediate)
        scaling = self.cast(self.scaling, self.compute_dtype)
        lora_result = self.mul(lora_result, scaling)

        # Add LoRA result to base output
        output = self.add(base_output, lora_result)

        # Shape restore
        if len(x_shape) != 2:
            out_shape = x_shape[:-1] + (output.shape[-1],)
            output = self.reshape(output, out_shape)
        output = self.cast(output, ori_dtype)

        return output, bias

    def shard_lora(self, config: TransformerConfig) -> None:
        """Shard the operators in ColumnParallelLinearWithLoRA.

        Args:
            config (TransformerConfig): The config of the transformer model.

        """
        dp = config.data_parallel_size if config.data_parallel_size is not None else 1
        tp = config.tensor_model_parallel_size if config.tensor_model_parallel_size is not None else 1
        cp = config.context_parallel_size if config.context_parallel_size is not None else 1
        # shard of lora operator
        self.lora_dropout.shard((cp, dp, tp))
        self.mul.shard(((cp, dp, tp), ()))
        self.add.shard(((cp, dp, tp), (cp, dp, tp)))


class RowParallelLinearWithLoRA(RowParallelLinear):
    r"""
    The LoRA layer with weight sliced in column dimension.

    Args:
        input_size (int): The number of input units.
        output_size (int): The number of output units.
        config (TransformerConfig): The config of the transformer model.
        init_method (Callable): The initialization method. Default: None
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
        bias_init (str): The initialization method for bias. Default: 'zeros'.
        lora_rank (int): The value of rank. Default: 8.
        lora_alpha (int):  The alpha value for lora. Default: 32.
        lora_dropout (float):  The dropout rate for lora. Default: 0.0.
        lora_a_init (Union[Tensor, str, Initializer, numbers.Number]):  The trainable lora_a weight_init parameter.
            The values of str refer to the function `initializer`. Default: 'normal'.
        lora_b_init (Union[Tensor, str, Initializer, numbers.Number]):  The trainable lora_b weight_init parameter.
            The values of str refer to the function `initializer`. Default: 'zeros'.

    Inputs:
        - **input_** (Tensor) - Tensor of shape :math:`(*, input\_size)`. The `input_size` in `Args` should be equal
          to :math:`input\_size` in `Inputs`.
        - **weight** (Tensor) - Tensor of shape :math:`(input\_size, output\_size)`/:math`(output\_size, input\_size)`.
          Default: None.

    Outputs:
        - **output** (Tensor): Result of linear with shape :math:`(*, output\_size)`.
        - **output_bias** (Tensor): Bias parameter when `skip_bias_add=True` with shape :math:`(output\_size)`.

    Raises:
        ValueError: `skip_weight_param_allocation=True` but weight_tensor is not passed to construct function.

    """

    def __init__(
            self,
            input_size: int,
            output_size: int,
            config: ModelParallelConfig,
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
            bias_init: Callable = None,
            lora_rank: int = 8,
            lora_alpha: int = 32,
            lora_dropout: float = 0.0,
            lora_a_init="normal",
            lora_b_init="zeros",
        ):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            config=config,
            init_method=init_method,
            bias=bias,
            input_is_parallel=input_is_parallel,
            skip_bias_add=skip_bias_add,
            stride=stride,
            keep_master_weight_for_test=keep_master_weight_for_test,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            transpose_b=transpose_b,
            compute_dtype=compute_dtype,
            bias_init=bias_init,
        )

        # LoRA parameters
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = Dropout(drop_prob=lora_dropout)
        self.scaling = self.lora_alpha / self.lora_rank

        # LoRA layers
        input_size = input_size
        output_size = output_size
        self.lora_A = RowParallelLinear(
            input_size=input_size,
            output_size=lora_rank,
            config=config,
            init_method=init_lora_method(lora_a_init, config.params_dtype),
            skip_bias_add=True,
            bias=False,
            compute_dtype=self.compute_dtype,
        )
        self.lora_B = LinearNoTP(
            input_size=lora_rank,
            output_size=output_size,
            config=config,
            init_method=init_lora_method(lora_b_init, config.params_dtype),
            skip_bias_add=True,
            bias=False,
            compute_dtype=self.compute_dtype,
        )

        # Operations
        self.cast = Cast()
        self.reshape = Reshape()
        self.add = AddExt()
        self.mul = Mul()

        if config is not None:
            if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
                self.sharding_propagation(config)
            else:
                self.shard_lora(config)

    def construct(self, input_: Tensor) -> tuple[Tensor, Tensor]:
        """Forward of RowParallelLinearWithLoRA.

        Args:
            input_ (Tensor): The input tensor.

        Returns:
            - output (Tensor): The output tensor.
            - bias (Optional[Tensor]): The bias
        """
        # Get original output from base_layer
        base_output, bias = super().construct(input_)

        # Data type operation
        ori_dtype = input_.dtype

        # Shape operations
        x_shape = input_.shape

        # LoRA result
        input_dropped = self.lora_dropout(input_)
        lora_intermediate, _ = self.lora_A(input_dropped)
        lora_result, _ = self.lora_B(lora_intermediate)
        scaling = self.cast(self.scaling, self.compute_dtype)
        lora_result = self.mul(lora_result, scaling)

        # Add LoRA result to base output
        output = self.add(base_output, lora_result)

        # Shape restore
        if len(x_shape) != 2:
            out_shape = x_shape[:-1] + (output.shape[-1],)
            output = self.reshape(output, out_shape)
        output = self.cast(output, ori_dtype)

        return output, bias

    def shard_lora(self, config: TransformerConfig) -> None:
        """Shard the operators in RowParallelLinearWithLoRA.

        Args:
            config (TransformerConfig): The config of the transformer model.

        """
        dp = config.data_parallel_size if config.data_parallel_size is not None else 1
        tp = config.tensor_model_parallel_size if config.tensor_model_parallel_size is not None else 1
        cp = config.context_parallel_size if config.context_parallel_size is not None else 1
        # shard of lora operator
        self.lora_dropout.shard((cp, dp, tp))
        self.mul.shard(((cp, dp, tp), ()))
        self.add.shard(((cp, dp, tp), (cp, dp, tp)))
