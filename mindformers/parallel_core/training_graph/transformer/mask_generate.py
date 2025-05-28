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
"""Attention Mask Generate"""
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor
from mindspore.ops import operations as P
from mindspore.context import ParallelMode
import mindspore.common.dtype as mstype
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindformers.parallel_core.transformer_config import TransformerConfig


class CausalMaskGenerate(nn.Cell):
    """Get the upper triangular matrix from the input_ids.

    Args:
        seq_length (int): The length of the input sequence.
        compute_type (mstype): The compute type of the input tensor. Default: mstype.float16.
        is_dynamic (bool): Whether the input_ids is dynamic. Default: False.
        pad_token_id (int): The pad token id. Default: 0.
        use_flash_attention (bool): Whether to use the flash attention. Default: False.
        use_prompt_flash_attention (bool): Whether to use the prompt flash attention. Default: False.
        use_incre_flash_attention (bool): Whether to use the incremental flash attention. Default: False.
        use_attn_mask_compression (bool): Whether to use the attention mask compression. Default: False.
    """

    def __init__(self,
                 seq_length: int,
                 compute_type: mstype = mstype.float16,
                 is_dynamic: bool = False,
                 pad_token_id: int = 0,
                 use_flash_attention: bool = False,
                 use_prompt_flash_attention: bool = False,
                 use_incre_flash_attention: bool = False,
                 use_attn_mask_compression: bool = False,
                 config: TransformerConfig = None
                 ):
        super().__init__()
        self.dtype = compute_type
        self.is_dynamic = is_dynamic
        self.pad_token_id = pad_token_id
        self.use_flash_attention = use_flash_attention
        self.use_attn_mask_compression = use_attn_mask_compression
        self.seq_length = seq_length
        self.use_prompt_flash_attention = use_prompt_flash_attention
        self.use_incre_flash_attention = use_incre_flash_attention
        self.is_first_iteration = True
        self.multiply_data = Tensor([-10000.0], dtype=compute_type)
        self.one = Tensor([1.0], dtype=compute_type)
        if use_attn_mask_compression:
            if seq_length < 2048:
                raise ValueError("seq_length should be larger than 2048 when use mask_compression")
            self.lower_triangle_mask = ms.Tensor(np.triu(np.ones((2048, 2048), dtype=np.int8), k=1), dtype=ms.uint8)
        else:
            self.lower_triangle_mask = Tensor(np.tril(np.ones(shape=(seq_length, seq_length), dtype=np.int8)),
                                              dtype=compute_type)
        self.shape = P.Shape()
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.not_equal = P.NotEqual()
        self.expand_dim = P.ExpandDims()
        self.slice = P.StridedSlice()
        self.mul = P.Mul()
        self.sub = P.Sub()
        self.expand_dim_post = P.ExpandDims()
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.sharding_propagation(config)
        else:
            self.shard(config)

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
            bs = self.shape(tokens)[0]
            seq_len = self.shape(tokens)[1]
            input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), self.dtype)
        else:
            bs = self.shape(masks)[0]
            seq_len = self.shape(masks)[1]
            input_mask = self.cast(masks, self.dtype)
        shape_right = (bs, 1, seq_len)

        # Mask the padded inputs
        mask_right = self.reshape(input_mask, shape_right)
        attention_mask = mask_right
        if not self.is_dynamic:
            lower_triangle = self.expand_dim(self.lower_triangle_mask, 0)
        else:
            lower_triangle_mask = self.slice(self.lower_triangle_mask, (0, 0), (seq_len, seq_len), (1, 1))
            lower_triangle = self.expand_dim(lower_triangle_mask, 0)

        # the returned shape is [bs, 1, seq_length, seq_length]
        attention_mask = self.mul(attention_mask, lower_triangle)
        attention_mask = self.sub(self.one, attention_mask)
        attention_mask = self.expand_dim_post(attention_mask, 1)
        if self.use_flash_attention or self.use_prompt_flash_attention:
            attention_mask = self.cast(attention_mask, mstype.uint8)
        return attention_mask

    def shard(self, config: TransformerConfig):
        """sharding operators
        """
        dp = config.data_parallel_size if config.data_parallel_size is not None else 1
        self.not_equal.shard(((dp, 1), ()))
        self.expand_dim.shard(((1, 1),))
        self.mul.shard(((dp, 1, 1), (1, 1, 1)))
        self.sub.shard(((1,), (dp, 1, 1)))
        self.expand_dim_post.shard(((dp, 1, 1),))

    def sharding_propagation(self, config: TransformerConfig):
        pass
