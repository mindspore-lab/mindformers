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
"""Attention Mask Generate"""
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops, mint
import mindspore.common.dtype as mstype


class CausalMaskGenerate(nn.Cell):
    """Generate causal attention mask from input tokens or masks.

    Generates attention masks for causal language modeling. Supports two mask types:

    1. Upper triangular mask (default): Shape [bs, 1, seq_len, seq_len]
       When is_dynamic=False: seq_len equals seq_length (fixed)
       When is_dynamic=True: seq_len is the actual input sequence length (variable)
       Output mask generated from tokens or masks input
       Example for seq_len=3 (showing the last two dimensions):
       0 1 1
       0 0 1
       0 0 0

    2. Upper triangular mask with compression (use_attn_mask_compression=True):
       Shape (2048, 2048), pre-computed for efficiency
       Example pattern (3x3 for illustration, actual size is 2048x2048):
       0 1 1
       0 0 1
       0 0 0

    Args:
        seq_length (int): The length of the input sequence.
        compute_type (mstype): The compute type of the input tensor. Default: mstype.float16.
        is_dynamic (bool): Whether the input_ids is dynamic. Default: False.
        pad_token_id (int): The pad token id. Default: 0.
        use_attn_mask_compression (bool): Whether to use the attention mask compression. Default: False.
    """

    def __init__(self,
                 seq_length: int,
                 compute_type: mstype = mstype.float16,
                 is_dynamic: bool = False,
                 pad_token_id: int = 0,
                 use_attn_mask_compression: bool = False
                 ):
        super().__init__()
        self.compute_type = compute_type
        self.is_dynamic = is_dynamic
        self.pad_token_id = pad_token_id
        self.use_attn_mask_compression = use_attn_mask_compression
        self.seq_length = seq_length
        self.one = Tensor([1.0], dtype=compute_type)
        if use_attn_mask_compression:
            if seq_length < 2048:
                raise ValueError("seq_length should be larger than 2048 when use mask_compression")
            self.lower_triangle_mask = Tensor(np.triu(np.ones((2048, 2048), dtype=np.int8), k=1), dtype=ms.uint8)
        else:
            self.lower_triangle_mask = Tensor(np.tril(np.ones(shape=(seq_length, seq_length), dtype=np.int8)),
                                              dtype=compute_type)
        self.cast = ops.cast
        self.reshape = mint.reshape
        self.not_equal = mint.not_equal
        self.expand_dim = mint.unsqueeze
        self.slice = ops.strided_slice
        self.mul = mint.mul
        self.sub = mint.sub

    def construct(self, tokens=None, masks=None):
        """Forward process of the CausalMask

        Args:
            tokens (Tensor): The input tokens. Default: None.
            masks (Tensor): The input masks. Default: None.

        Returns:
            Tensor, the upper triangle attention mask carrying 0 and 1 values
        """
        if self.use_attn_mask_compression:
            attention_mask = self.lower_triangle_mask
            return attention_mask
        if tokens is not None:
            bs = tokens.shape[0]
            seq_len = tokens.shape[1]
            input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), self.compute_type)
        else:
            bs = masks.shape[0]
            seq_len = masks.shape[1]
            input_mask = self.cast(masks, self.compute_type)
        shape_right = (bs, 1, seq_len)

        # Mask the padded inputs
        mask_right = self.reshape(input_mask, shape_right)
        attention_mask = mask_right
        if not self.is_dynamic:
            lower_triangle = self.expand_dim(self.lower_triangle_mask, 0)
        else:
            lower_triangle_mask = self.slice(self.lower_triangle_mask, (0, 0), (seq_len, seq_len), (1, 1))
            lower_triangle = self.expand_dim(lower_triangle_mask, 0)

        # the returned shape is [bs, 1, seq_len, seq_len] (seq_len may differ from seq_length when is_dynamic=True)
        attention_mask = self.mul(attention_mask, lower_triangle)
        attention_mask = self.sub(self.one, attention_mask)
        attention_mask = self.expand_dim(attention_mask, 1)
        attention_mask = self.cast(attention_mask, mstype.uint8)
        return attention_mask
