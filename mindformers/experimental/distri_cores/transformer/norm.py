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
"""Normalization"""
import mindspore.ops.operations as P
import mindspore.common.dtype as mstype

from mindspore import nn, Parameter, Tensor
from mindspore.common.initializer import initializer

__all__ = ["get_norm"]


class LayerNorm(nn.Cell):
    r"""
    Layer norm operation.

    Args:
        normalized_shape (tuple): The shape of the input tensor
        eps (float): The epsilon value of the denominator. Default 1e-5.
        param_init_type: The param init type.

    Inputs:
        - **x** (Tensor) - Tensor of shape (batch, seq_length, hidden_size).

    Outputs:
        - Tensor with shape (batch, seq_length, hidden_size).
    """

    def __init__(self, normalized_shape, eps=1e-5, param_init_type=mstype.float32):
        super(LayerNorm, self).__init__()
        if param_init_type not in [mstype.float32, mstype.float16, mstype.bfloat16]:
            raise TypeError("The type of parameter 'param_init_type' should in [float32, float16], "
                            "but got the type : {}.".format(type(param_init_type)))
        self.layer_norm = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=eps)
        self.gamma = Parameter(initializer('ones', normalized_shape, param_init_type), name="gamma",
                               parallel_optimizer=False)
        self.beta = Parameter(initializer('zeros', normalized_shape, param_init_type), name="beta",
                              parallel_optimizer=False)

    def construct(self, x):
        """construct method"""
        output, _, _ = self.layer_norm(x, self.gamma, self.beta)
        return output


class RMSNorm(nn.Cell):
    r"""
    A self-defined RMSNorm operation using reduce mean.

    Args:
        dim (tuple): The shape of the input tensor
        eps (float): The epsilon value of the denominator. Default 1e-5.
        param_init_type: The param init type.

    Inputs:
        - **x** (Tensor) - Tensor of shape (batch, seq_length, hidden_size).

    Outputs:
        - Tensor with shape (batch, seq_length, hidden_size).
    """

    def __init__(self, dim, eps=1e-6, param_init_type=mstype.float32):
        super(RMSNorm, self).__init__()
        self.eps = Tensor(float(eps), dtype=param_init_type)
        self.weight = Parameter(initializer("ones", (dim,), dtype=param_init_type))

        self.square = P.Square()
        self.mean = P.ReduceMean(keep_dims=True)
        self.add = P.Add()
        self.rsqrt = P.Rsqrt()
        self.mul = P.Mul()
        self.mul2 = P.Mul()

    def construct(self, x):
        """Forward of RMSNorm."""
        norm_factor = self.square(x)
        norm_factor = self.mean(norm_factor, -1)
        norm_factor = self.add(norm_factor, self.eps)
        norm_factor = self.rsqrt(norm_factor)
        output = self.mul(x, norm_factor)
        output = self.mul2(output, self.weight)
        return output


class FusedRMSNorm(nn.Cell):
    r"""
    A RMSNorm fused kernel implementation.

    Args:
        dim (tuple): The shape of the input tensor
        eps (float): The epsilon value of the denominator. Default 1e-5.
        param_init_type: The param init type.

    Inputs:
        - **x** (Tensor) - Tensor of shape (batch, seq_length, hidden_size).

    Outputs:
        - Tensor with shape (batch, seq_length, hidden_size).
    """

    def __init__(self, dim, eps=1.e-6, compute_type=mstype.float32):
        super(FusedRMSNorm, self).__init__()
        self.eps = eps
        self.compute_type = compute_type
        self.weight = Parameter(initializer("ones", (dim,), dtype=mstype.float32), parallel_optimizer=False)

        self.norm = P.RmsNorm(eps)
        self.cast = P.Cast()
        self.rcast = P.Cast()

    def construct(self, x):
        """Forward of FusedRMSNorm."""
        original_type = x.dtype
        output = self.norm(self.cast(x, self.compute_type), self.weight)[0]
        return self.rcast(output, original_type)


def get_norm(config):
    r"""
    Get normalization layer.

    Args:
        config: The config of the model.

    Returns:
        callable, the normalization layer.
    """
    if config.normalization == "LayerNorm":
        return LayerNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon)
    if config.normalization == "RMSNorm":
        return RMSNorm(dim=config.hidden_size,
                       eps=config.layernorm_epsilon)
    if config.normalization == "FusedRMSNorm":
        return FusedRMSNorm(dim=config.hidden_size, eps=config.layernorm_epsilon)

    raise Exception(f"unsupported norm type '{config.normalization}'.")
