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
"""Dropout layer implementation based on MindSpore mint functional API.

This module provides a lightweight `Dropout` wrapper implemented as a
MindSpore `nn.Cell`. It is designed to integrate with parallel and
transformer-style training pipelines, while delegating the core
dropout computation to the mint functional interface.
"""

__all__ = ["Dropout"]

from mindspore import nn
from mindspore.mint.nn.functional import dropout


class Dropout(nn.Cell):
    """
    Dropout layer.

    This layer randomly zeroes some elements of the input tensor during
    training with probability `drop_prob`. During evaluation, the input
    is returned unchanged.

    The implementation is a thin wrapper around
    `mindspore.mint.nn.functional.dropout` and respects the module's
    `training` state.

    Args:
        drop_prob (float, optional):
            Probability of dropping an element. Must be a float in the range [0.0, 1.0]. A value of 0.0 disables
            dropout, while 1.0 drops all elements during training. Default: 0.5.

    Raises:
        TypeError:
            If `drop_prob` is not a float.
        ValueError:
            If `drop_prob` is not in the range [0.0, 1.0].

    Inputs:
        x (Tensor):
            Input tensor of arbitrary shape.

    Outputs:
        Tensor:
            Output tensor with the same shape and dtype as the input.
    """

    def __init__(self, drop_prob: float = 0.5):
        super().__init__()
        # Validate drop_prob input
        if not isinstance(drop_prob, float):
            raise TypeError(f"drop_prob must be a float, but got {type(drop_prob).__name__}")
        if drop_prob < 0.0 or drop_prob > 1.0:
            raise ValueError(f"drop_prob must be in the range [0.0, 1.0], but got {drop_prob}")
        self.p = drop_prob
        self.use_dropout = drop_prob != 0
        self.dropout = dropout

    def construct(self, x):
        """
        Apply dropout to the input tensor.

        Args:
            x (Tensor):
                Input tensor.

        Returns:
            Tensor:
                Tensor after applying dropout during training,
                or the original input during evaluation.
        """
        out = self.dropout(x, self.p, self.training)
        return out
