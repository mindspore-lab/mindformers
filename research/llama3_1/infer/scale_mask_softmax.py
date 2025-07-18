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
DEPRECATED MODULE

This module is deprecated and will be removed in future releases.
ScaleMaskSoftmax
"""
import mindspore.ops.functional as F

from mindspore import mint, ops, nn
from mindspore.common import dtype as mstype


class ScaleMaskSoftmax(nn.Cell):
    r"""
    fused operation: scaling + mask + softmax

    Args:
        mask_func: mask function to be applied.
        scale: scaling factor used in input tensor scaling.
        softmax_compute_type: softmax in performed precision.

    Inputs:
        - **x** (Tensor) - The input tensor
        - **mask** (Tensor) - The mask tensor

    Outputs:
        - The output tensor.
    """

    def __init__(self, mask_func, scale=None, softmax_compute_type=mstype.float32):
        super().__init__()
        self.mask_func = mask_func
        self.softmax_compute_type = softmax_compute_type
        self.scale = scale

        if self.scale is not None and self.softmax_compute_type != mstype.float32:
            raise ValueError("softmax should be in fp32 when scaled")

    def construct(self, x, mask):
        """construct method"""
        origin_dtype = F.dtype(x)
        if self.softmax_compute_type != origin_dtype:
            x = ops.cast(x, self.softmax_compute_type)

        if self.scale is not None:
            x = x * self.scale
        masked_input = self.mask_func(x, mask) if mask is not None else x

        probs = mint.nn.functional.softmax(masked_input, dim=-1)

        if self.softmax_compute_type != origin_dtype:
            probs = ops.cast(probs, origin_dtype)

        return probs
