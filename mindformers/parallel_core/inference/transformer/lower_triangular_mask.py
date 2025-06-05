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
"""LowerTriangularMask Module"""

__all__ = ['LowerTriangularMaskWithDynamic']

import numpy as np

from mindspore import nn, Tensor, ops
import mindspore.common.dtype as mstype


class LowerTriangularMaskWithDynamic(nn.Cell):
    r"""
    Generates a strictly lower triangular mask for attention.

    Args:
        seq_length (int): Maximum sequence length for the mask.
        compute_type (mstype): Data type for computations.
            Defaults to float16.
        pad_token_id (int): Padding token ID. Defaults to 0.

    Inputs:
        - **positions** (Tensor) - Token positions tensor of shape [batch_size, seq_length].
          Only used in decode mode.

    Outputs:
        - **mask** (Tensor) - Lower triangular mask tensor.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(self, seq_length, compute_type=mstype.float16, pad_token_id=0):
        super().__init__()
        self.compute_dtype = compute_type
        self.pad_token_id = pad_token_id
        self.seq_length = seq_length
        self.is_prefill = True
        mask_coeff = 1.0 if self.compute_dtype is mstype.bfloat16 else -10000.0
        full_mask = np.ones(shape=(self.seq_length, self.seq_length), dtype=np.int8)
        self.pa_lower_triangle_mask = Tensor(np.triu(full_mask, 1),
                                             dtype=self.compute_dtype) * -10000
        self.fa_lower_triangle_mask = Tensor(np.triu(np.ones(shape=(128, 128), dtype=np.float16), 1) * mask_coeff,
                                             dtype=self.compute_dtype)
        self.gather = ops.Gather()

    def construct(self, positions):
        """Forward process of the CausalMask"""
        if self.is_prefill:
            return self.prefill()

        return self.decode(positions)

    def prefill(self):
        """Generate the lower triangular mask for prefill."""
        return self.fa_lower_triangle_mask

    def decode(self, positions):
        """Generate dynamic mask based on token positions for decode."""
        return self.gather(self.pa_lower_triangle_mask, positions, 0)
