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
"""Fused normalization layers for transformer models.

This module provides fused implementations of commonly used normalization
layers, including LayerNorm and RMSNorm, implemented as MindSpore `nn.Cell`s.

These fused variants are designed for transformer-style models and support:
    - Parameter dtype control (e.g., fp32 parameters)
    - Computation dtype control (e.g., fp16/bf16 compute)
    - Factory-style construction via `FusedNorm`

Currently supported normalization types:
    - FusedLayerNorm
    - FusedRMSNorm
"""

__all__ = ["get_norm_cls"]

from mindspore import nn, dtype, Parameter
from mindspore.ops import cast, rms_norm
from mindspore.common.initializer import initializer
from mindspore.mint.nn.functional import layer_norm


class FusedLayerNorm(nn.Cell):
    """
    Fused Layer Normalization cell.

    This class implements Layer Normalization over the last dimension
    of the input tensor using MindSpore's fused `layer_norm` functional
    interface.

    The implementation supports:
        - Explicit parameter dtype (gamma / beta)
        - Separate computation dtype for numerical stability
        - Automatic casting back to the original input dtype

    Args:
        dim (Union[int, tuple[int], list[int]]):
            Shape of the normalized dimension(s), typically the hidden size.
        eps (float, optional):
            Small constant added to variance for numerical stability.
            Default: 1e-5.
        params_dtype (mindspore.dtype, optional):
            Data type of learnable parameters (gamma, beta).
            Default: mindspore.float32.
        compute_dtype (mindspore.dtype, optional):
            Data type used during LayerNorm computation.
            Default: mindspore.float32.

    Inputs:
        x (Tensor):
            Input tensor of shape (..., hidden_size). Layer normalization
            is applied over the last dimension.

    Outputs:
        Tensor:
            Normalized tensor with the same shape and dtype as the input.
    """

    def __init__(self, dim, eps=1e-5, params_dtype=dtype.float32, compute_dtype=dtype.float32):
        super().__init__()
        self.params_dtype = params_dtype
        self.compute_type = compute_dtype
        self.eps = eps

        self.layer_norm = layer_norm
        self.gamma = Parameter(
            initializer('ones', dim, self.params_dtype),
            name="gamma"
        )
        self.beta = Parameter(
            initializer('zeros', dim, self.params_dtype),
            name="beta"
        )
        self.cast = cast

    def construct(self, x):
        """Apply fused Layer Normalization."""
        original_type = x.dtype
        output = self.layer_norm(self.cast(x, self.compute_type), x.shape[-1], self.gamma, self.beta, self.eps)
        output = self.cast(output, original_type)
        return output


class FusedRMSNorm(nn.Cell):
    """
    Fused RMS Normalization cell.

    This class implements RMSNorm using MindSpore's fused `RmsNorm` operator.
    RMSNorm normalizes inputs based on the root mean square of activations
    and applies a learnable scale parameter.

    Compared to LayerNorm, RMSNorm:
        - Does not use a bias term
        - Often provides better numerical stability in large models

    Args:
        dim (Union[int, tuple[int], list[int]]):
            Shape of the normalized dimension(s), typically the hidden size.
        eps (float, optional):
            Small constant added for numerical stability.
            Default: 1e-5.
        params_dtype (mindspore.dtype, optional):
            Data type of the learnable weight parameter.
            Default: mindspore.float32.
        compute_dtype (mindspore.dtype, optional):
            Data type used during RMSNorm computation.
            Default: mindspore.float32.

    Inputs:
        x (Tensor):
            Input tensor of shape (..., hidden_size). Normalization
            is applied over the last dimension.

    Outputs:
        Tensor:
            Normalized tensor with the same shape and dtype as the input.
    """

    def __init__(self, dim, eps=1e-5, params_dtype=dtype.float32, compute_dtype=dtype.float32):
        super().__init__()
        self.params_dtype = params_dtype
        self.compute_type = compute_dtype

        self.eps = eps
        self.weight = Parameter(initializer('ones', dim, self.params_dtype))

        self.norm = rms_norm
        self.cast = cast

    def construct(self, x):
        """Apply fused RMS Normalization."""
        original_type = x.dtype
        output = self.norm(self.cast(x, self.compute_type), self.weight, self.eps)[0]
        output = self.cast(output, original_type)
        return output


def get_norm_cls(
        normalization,
        fused_norm=True,
):
    """
    Supported normalization types:
        - "LayerNorm": returns a `FusedLayerNorm`
        - "RMSNorm": returns a `FusedRMSNorm`

    Args:
        normalization (str):
            Name of the normalization type. Must be either
            "LayerNorm" or "RMSNorm".
        fused_norm (bool):
            Whether to use fused normalization or not.

    Returns:
        nn.Cell:
            A class of `FusedLayerNorm` or `FusedRMSNorm`.

    Raises:
        ValueError:
            If an unsupported normalization type is specified.
    """

    if fused_norm:
        if normalization == "LayerNorm":
            return FusedLayerNorm
        if normalization == "RMSNorm":
            return FusedRMSNorm
        raise ValueError("Only 'LayerNorm' and 'RMSNorm' are currently supported.")
    raise ValueError("Only fused normalization layers are supported.")
