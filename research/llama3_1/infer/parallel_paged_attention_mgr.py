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
Paged Attention Manager for inference.
"""
import math

import mindspore.common.dtype as mstype
from mindspore import ops as P
from mindspore import nn
from mindformers.parallel_core.inference.utils import create_empty_parameter


class ParallelPagedAttentionMgr(nn.Cell):
    """Paged Attention Manager."""
    def __init__(self,
                 n_heads,
                 head_dim,
                 n_kv_heads,
                 kv_shape,
                 seq_length=-1,
                 compute_dtype=mstype.float16,
                 npu_mem_size=2):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.seq_length = seq_length
        self.is_first_iteration = True
        self.scale_value = 1 / math.sqrt(self.head_dim)
        self.key_cache = None
        self.value_cache = None
        self.npu_mem_size = npu_mem_size
        if self.npu_mem_size > 0:
            self.key_cache = create_empty_parameter(
                shape=kv_shape,
                dtype=compute_dtype,
                device="Ascend",
                name="key_cache",
                requires_grad=False,
            )
            self.value_cache = create_empty_parameter(
                shape=kv_shape,
                dtype=compute_dtype,
                device="Ascend",
                name="value_cache",
                requires_grad=False,
            )

        self.reshape_and_cache = P.auto_generate.ReshapeAndCache()
        self.paged_attention = P.auto_generate.PagedAttention(self.n_heads,
                                                              self.scale_value,
                                                              self.n_kv_heads)
        self.paged_attention_with_alibi = P.auto_generate.PagedAttentionMask(self.n_heads,
                                                                             self.scale_value,
                                                                             self.n_kv_heads)

    def construct(self, key, value, slot_mapping, _, key_cache=None, value_cache=None):
        """The forward compute of KVCache for Paged Attention."""
        if self.npu_mem_size == -1:
            return self.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
        return self.reshape_and_cache(key, value, self.key_cache, self.value_cache, slot_mapping)

    def paged_attn(self, query, batch_valid_length, block_tables, attn_mask=None, q_seq_lens=None,
                   key_cache=None, value_cache=None):
        if self.npu_mem_size == -1:
            return self._paged_attn(query, batch_valid_length, block_tables, attn_mask, q_seq_lens,
                                    key_cache, value_cache)
        return self._paged_attn(query, batch_valid_length, block_tables, attn_mask, q_seq_lens,
                                self.key_cache, self.value_cache)

    def _paged_attn(self, query, batch_valid_length, block_tables, attn_mask=None, q_seq_lens=None,
                    key_cache=None, value_cache=None):
        """The forward compute of Paged Attention."""
        return self.paged_attention(query, key_cache, value_cache, block_tables, batch_valid_length,
                                    None, None, attn_mask, q_seq_lens)
