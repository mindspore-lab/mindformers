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
"""Activation functions for transformer."""
__all__ = ["FusedSwiGlu", "GELU", "SiLU"]

from mindspore import nn, Tensor
from mindspore import mint
from mindspore import ops


class FusedSwiGlu(nn.Cell):
    """
    Computes SwiGLU (Swish-Gated Linear Unit activation function) of input tensor.
    SwiGLU is a variant of the :class:`mindspore.ops.GLU` activation function, it is defined as:

    .. math::
        {SwiGLU}(a, b)= Swish(a) \\otimes b

    where :math:`a` is the first half of the `input` matrices and :math:`b` is the second half,
    Swish(a)=a :math:`\\sigma` (a), :math:`\\sigma` is the :func:`mindspore.ops.sigmoid` activation function
    and :math:`\\otimes` is the Hadamard product.

    Args:
        input (Tensor): Tensor to be split. It has shape :math:`(\\ast_1, N, \\ast_2)`
            where `*` means, any number of additional dimensions. :math:`N` must be divisible by 2.
        dim (int, optional): the axis to split the input. It must be int. Default: ``-1`` , the last axis of `input`.

    Returns:
        Tensor, the same dtype as the `input`, with the shape :math:`(\\ast_1, M, \\ast_2)` where :math:`M=N/2`.

    Raises:
        TypeError: If dtype of `input` is not float16, float32 or bfloat16.
        TypeError: If `input` is not a Tensor.
        RuntimeError: If the dimension specified by `dim` is not divisible by 2.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindformers.parallel_core.training_pynative.layers.activation import FusedSwiGlu
        >>> input = Tensor([[-0.12, 0.123, 31.122], [2.1223, 4.1212121217, 0.3123]], dtype=mindspore.float32)
        >>> output = FusedSwiGlu()(input, 0)
        >>> print(output)
        [[-0.11970687 0.2690224 9.7194 ]]
    """

    def __init__(self):
        super().__init__()
        self.swiglu = ops.swiglu

    def construct(self, x: Tensor, dim=-1) -> Tensor:
        """Apply the fused SwiGLU activation."""
        return self.swiglu(x, dim)


class GELU(nn.Cell):
    """
    Gaussian Error Linear Units activation function.

    GeLU is described in the paper `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_.
    And also please refer to `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
    <https://arxiv.org/abs/1810.04805>`_.

    When `approximate` argument is `none`, GELU is defined as follows:

    .. math::
        GELU(x_i) = x_i*P(X < x_i),

    where :math:`P` is the cumulative distribution function of the standard Gaussian distribution,
    :math:`x_i` is the input element.

    When `approximate` argument is `tanh`, GELU is estimated with:

    .. math::
        GELU(x_i) = 0.5 * x_i * (1 + \\tanh(\\sqrt(2 / \\pi) * (x_i + 0.044715 * x_i^3)))

    Args:
        input (Tensor): The input of the activation function GeLU, the data type is float16, float32 or float64.

    Keyword Args:
        approximate (str, optional): the gelu approximation algorithm to use.
        Acceptable vaslues are ``'none'`` and ``'tanh'`` .
            Default: ``'none'`` .

    Returns:
        Tensor, with the same type and shape as `input`.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If dtype of `input` is not bfloat16, float16, float32 or float64.
        ValueError: If `approximate` value is neither `none` nor `tanh`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindformers.parallel_core.training_pynative.layers.activation import GELU
        >>> input = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> result = GELU()(input)
        >>> print(result)
        [[-1.58655241e-01  3.99987316e+00 -0.00000000e+00]
         [ 1.95449972e+00 -1.41860323e-06  9.0000000e+00]]
        >>> result = GELU(approximate="tanh")(input)
        >>> print(result)
        [[-1.58808023e-01  3.99992990e+00 -3.10779147e-21]
         [ 1.95459759e+00 -2.29180174e-07  9.0000000e+00]]
    """

    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate
        self.gelu = mint.nn.functional.gelu

    def construct(self, x: Tensor) -> Tensor:
        """Apply GELU activation function."""
        return self.gelu(x, approximate=self.approximate)


class SiLU(nn.Cell):
    """
    Computes Sigmoid Linear Unit of input element-wise. The SiLU function is defined as:

    .. math::

        \\text{SiLU}(x) = x * \\sigma(x),

    where :math:`x` is an element of the input, :math:`\\sigma(x)` is Sigmoid function.

    .. math::

        \\text{sigma}(x_i) = \\frac{1}{1 + \\exp(-x_i)},

    Args:
        input (Tensor): `input` is :math:`x` in the preceding formula. Input with the data type
            float16 or float32.
        inplace (bool, optional): If it is ``True``, enable the in place update function. Default value: ``False``.

    Returns:
        Tensor, with the same type and shape as the `input`.

    Raises:
        TypeError: If dtype of `input` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> import numpy as np
        >>> from mindformers.parallel_core.training_pynative.layers.activation import SiLU
        >>> input = Tensor(np.array([-1, 2, -3, 2, -1]), mindspore.float16)
        >>> output = SiLU()(input)
        >>> print(output)
        [-0.269  1.762  -0.1423  1.762  -0.269]
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.silu = mint.nn.functional.silu

    def construct(self, x: Tensor) -> Tensor:
        """Apply SiLU activation function."""
        return self.silu(x, inplace=self.inplace)


ACTIVATION_MAP = {
    'gelu': GELU,
    'silu': SiLU,
    'fusedswiglu': FusedSwiGlu
}


def get_activation(activation_name, *args, **kwargs):
    """Create and return an activation Cell by name."""
    activation_name = activation_name.lower()
    if activation_name not in ACTIVATION_MAP:
        raise ValueError(
            f"Activation '{activation_name}' is not supported. "
            f"Supported activations are: {list(ACTIVATION_MAP.keys())}"
        )

    # activation should be a Cell in Static
    activation = ACTIVATION_MAP[activation_name]
    return activation(*args, **kwargs)
