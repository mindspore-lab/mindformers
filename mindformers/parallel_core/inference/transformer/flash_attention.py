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
"""Flash Attention Layer"""
__all__ = ['FlashAttention']

from mindspore import ops
from mindspore.nn.cell import Cell
from mindspore.ops.operations.nn_ops import FlashAttentionScore


class FlashAttention(Cell):
    """Flash Attention Layer.

    This function contains the flash attention primitives used in FlashAttention (see paper) and PagedAttention.
    `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness <https://arxiv.org/pdf/2205.14135.pdf>`

    Args：
        - head_num (int): Number of attention heads.
        - kv_cache_shape (tuple): Shape of the key-value cache tensor.
        - head_dim (Optional[int]): Dimension of each attention head. Default: None.
        - kv_head_num (Optional[int]): Number of key-value heads. Default: None
        - keep_prob (float): Dropout keep probability. Default: 1.0.
        - scale_value (float): Scaling factor for attention scores. Default: 1.0.
        - pre_tokens (int): Number of previous tokens to consider. Default: 2147483647.
        - next_tokens (int): Number of next tokens to consider. Default: 2147483647.
        - sparse_mode (int): Mode for sparse attention. Default: 0.
        - compute_dtype (mstype): Data type for computation. Default: mstype.float16.
        - input_layout (str): Layout of input tensors("TH" or "TND"). Default: "TH".


    Inputs:
        - **query** (Tensor[float16, bfloat16]) - The query tensor.
        - **key** (Tensor[float16, bfloat16]) - The key tensor.
        - **value** (Tensor[float16, bfloat16]) - The value tensor.
        - **slot_mapping** (Tensor) - Store token cache physical slot index.
        - **block_tables** (Tensor) - The block mapping table with data type of int32.
        - **batch_valid_length** (Tensor) -  In incremental inference, a tensor used for calculating the index
          of the previous step. It is of int32 type and has a shape of [batch_size].
        - **context_lens_tensor** (Tensor) - The context length of each sequence with data type of int32.
        - **actual_seq_qlen** (Union[List[int64], Tuple[int64], None]) - Size of query corresponding to each batch,
          array with increasing values and the last value equal to T1.
        - **actual_seq_kvlen** (Union[List[int64], Tuple[int64], None]) - Size of key and value corresponding to each
          batch, array with increasing values and the last value equal to T2.
        - **attn_mask** (Union[Tensor, None]) - The attention mask tensor.
        - **padding_mask** (None) - Reserved parameter. Not implemented yet.
        - **prefix** (Union[Tensor[int64], None]) - N value of each Batch in the prefix sparse calculation scenario.
          Not implemented yet. Input tensor of shape :math:`(B,)`.
        - **key_cache** (Tensor, optional) - Key cache for incremental inference.
        - **value_cache** (Tensor, optional) - Value cache for incremental inference.

    Outputs:
        - **attention_out** (Tensor[float16, bfloat16]) - The output of attention, its shape, and data type
          are the same as the query.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(
            self,
            head_num,
            head_dim=None,
            kv_head_num=None,
            keep_prob=1.0,
            scale_value=1.0,
            pre_tokens=2147483647,
            next_tokens=2147483647,
            sparse_mode=0,
            input_layout="TH"
    ):
        super().__init__()
        self.head_num = head_num
        self.hidden_size_per_attention_head = head_dim
        self.kv_head_num = kv_head_num
        self.sparse_mode = sparse_mode
        self.is_prefill = True
        self.input_layout = input_layout

        self.reshape_and_cache = ops.auto_generate.ReshapeAndCache()

        self.flash_attention = FlashAttentionScore(
            head_num=head_num,
            keep_prob=keep_prob,
            scale_value=scale_value,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            inner_precise=0,
            input_layout=self.input_layout,
            sparse_mode=self.sparse_mode)

        self.paged_attention = ops.auto_generate.PagedAttention(
            self.head_num, scale_value, self.kv_head_num)

    def construct(self,
                  query,
                  key,
                  value,
                  slot_mapping=None,
                  block_tables=None,
                  batch_valid_length=None,
                  context_lens_tensor=None,
                  q_seq_lens=None,
                  actual_seq_qlen=None,
                  actual_seq_kvlen=None,
                  attn_mask=None,
                  padding_mask=None,
                  prefix=None,
                  key_cache=None,
                  value_cache=None):
        """Forward process of the FlashAttention."""

        bs, seq_len, _ = query.shape

        self.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

        if self.is_prefill:

            if self.input_layout == "TH":
                query = query.reshape((-1, self.head_num * self.hidden_size_per_attention_head))
                key = key.reshape((-1, self.kv_head_num * self.hidden_size_per_attention_head))
                value = value.reshape((-1, self.kv_head_num * self.hidden_size_per_attention_head))

            if self.input_layout == "TND":
                query = query.reshape((-1, self.head_num, self.hidden_size_per_attention_head))
                key = key.reshape((-1, self.kv_head_num, self.hidden_size_per_attention_head))
                value = value.reshape((-1, self.kv_head_num, self.hidden_size_per_attention_head))

            _, _, _, output = self.flash_attention(query, key, value,
                                                   None, None,
                                                   padding_mask, attn_mask,
                                                   prefix, actual_seq_qlen,
                                                   actual_seq_kvlen)
            context_layer = output
        else:
            context_layer = self.paged_attention(query, key_cache, value_cache,
                                                 block_tables, batch_valid_length, None,
                                                 None, attn_mask,
                                                 q_seq_lens)

        core_attn_out = context_layer.reshape(
            (bs, seq_len, self.head_num * self.hidden_size_per_attention_head))

        return core_attn_out
