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
"""Infer Attention Layer"""
import math

import mindspore.common.dtype as mstype
from mindspore import ops
from mindspore.common.tensor import Tensor
from mindspore.nn.cell import Cell
from mindspore.ops import operations as P

from mindformers.modules import PagedAttentionMgr
from mindformers.modules.flash_attention import FlashAttention


class InferAttention(Cell):
    """Infer Attention Layer.

    This function contains the InferAttention primitives used with FlashAttention and PagedAttention for kbk infer.

    B -- Batch size
    S1 -- Sequence length of query. The value ranges from 1 to 32768 and is a multiple of 16.
    S2 -- Sequence length of key and value. The value ranges from 1 to 32768 and is a multiple of 16.
    N1 -- Num heads of query
    N2 -- Num heads of key and value, and N2 must be a factor of N1
    D -- Head size. Support value: 64, 80, 96, 120, 128 and 256.
    H1 -- Hidden size of query, which equals to N1 * D
    H2 -- Hidden size of key and value, which equals to N2 * D
    Args:
        n_head (int): The head num of query.
        head_dim (int): The dim of head.
        keep_prob (float): The keep probability of dropout. Default: 1.0.
        scale_value (float): The scale factor of score. Default: 1.0.
        pre_tokens (int): Parameter for sparse computation, represents how many tokens are counted forward.
        When sparse_mode is set to 1, 2, 3, or 5, this parameter does not take effect. Default: 2147483647.
        next_tokens (int): Parameter for sparse computation, represents how many tokens are counted backward.
        When sparse_mode is set to 1, 2, 3, or 5, this parameter does not take effect. Default: 2147483647.
        input_layout (str): Specifies the layout of input `query`, key and value. The value can be "BSH" or "BNSD".
        Default: "BSH".
        sparse_mode (int): Indicates sparse mode. Default 0.

            - 0: Indicates the defaultMask mode. If attn_mask is not passed, the mask operation is not performed,
              and preTokens and nextTokens(internally assigned as INT_MAX) are ignored. If passed in, the full attn_mask
              matrix (S1 * S2) needs to be passed in, indicating that the part between preTokens and nextTokens needs to
              be calculated.
            - 1: Represents allMask, that is, passing in the complete attn_mask matrix.
            - 2: Representing the leftUpCausal mode corresponds to the lower triangle scenario divided by the left
              vertex, and the optimized attn_mask matrix (2048*2048) is required.
            - 3: Representing the rightDownCausal model corresponds to the lower triangle scene divided by the lower
              right vertex, and the optimized attn_mask matrix (2048*2048) is required.
            - 4: Represents the band scenario, that is, the part between counting preTokens and nextTokens, and the
              optimized attn_mask matrix (2048*2048) is required..
            - 5: Represents the prefix scenario, that is, on the basis of rightDownCasual, a matrix with length S1 and
              width N is added to the left side. The value of N is obtained by the new input prefix, and the N value of
              each Batch axis is different. Not implemented yet.
            - 6: Represents the global scenario, not implemented yet.
            - 7: Represents the dilated scenario, not implemented yet.
            - 8: Represents the block_local scenario, not implemented yet.
        block_size (int): Block size for paged attention.
        num_blocks (int): Block num for paged attention.
        use_attention_mask (bool): The value is True if attention_mask is passed. Default: False.
        use_alibi_mask (bool): The value is True if alibi_mask is passed. Default: False.
        use_rope_rotary_emb (bool): If use rotary embedding. Default True.
        rotary_cos_format (int): Choose the rotary embedding cos format. Default 0.
        parallel_config (ParallelConfig): Parallel config for infer attention. Default None.
        compute_dtype (mstype): Compute dtype for infer attention. Default mstype.float16.


    Inputs:
        - **query** (Tensor[float16, bfloat16]) - The query tensor.
          Input tensor of shape :math:`(B, S1, H1)` or `(B, N1, S1, D)`.
        - **key** (Tensor[float16, bfloat16]) - The key tensor.
          Input tensor of shape :math:`(B, S2, H2)` or `(B, N2, S2, D)`.
        - **value** (Tensor[float16, bfloat16]) - The value tensor.
          Input tensor of shape :math:`(B, S2, H2)` or `(B, N2, S2, D)`.
        - **batch_valid_length** (Tensor): Int32 tensor with shape [batch_size] the past calculated the index.
          Used for incremental prediction when the use_past is True. Default None.
        - **block_tables** (Tensor[int64]) - Store mapping tables for each sequence.
        - **slot_mapping** (Tensor[int32]) - Store token cache physical slot index.
        - **freqs_cos** (Tensor[float16, bfloat16]) - The precompute freqs cos for rotary position embedding used in
          attention, shape is (seq_len, head_dim).
        - **freqs_sin** (Tensor[float16, bfloat16]) - The precompute freqs sin for rotary position embedding used in
          attention, shape is (seq_len, head_dim).
        - **attn_mask** (Union[Tensor[uint8], None]) - The attention mask tensor. For each element, 0 indicates
          retention and 1 indicates discard. Input tensor of shape :math:`(B, N1, S1, S2)`, `(B, 1, S1, S2)`, `(S1, S2)`
          or (2048, 2048).
        - **alibi_mask** (Union[Tensor[float16, bfloat16], None]) - The position embedding code. If S is greater than
          1024 and the mask of the lower triangle is used, enter only the inverse 1024 lines of the lower triangle for
          memory optimization.
          Input tensor of shape :math: `(B, N1, S1, S2)`, `(1, N1, S1, S2)`, `(B, N1, 1024, S2)`, `(1, N1, 1024, S2)`
          or (1024, 1024).

    Outputs:
        - **attention_out** (Tensor[float16, bfloat16]) - The output of attention, its shape, and data type
          are the same as the query.

    Supported Platforms:
        ``Ascend910B``

    Examples:
        >>> import numpy as np
        >>> import math
        >>> from mindspore import dtype as mstype
        >>> from mindspore import Tensor
        >>> from mindformers.modules.infer_attention import InferAttention
        >>> bsz, head_num, seq_len, head_dim = 1, 16, 4096, 128
        >>> n_kv_head = 16
        >>> block_size = 1024
        >>> num_blocks = 16
        >>> hidden_size = head_num * head_dim
        >>> query = Tensor(np.ones((bsz, seq_len, hidden_size)), mstype.float16)
        >>> key = Tensor(np.ones((bsz, seq_len, hidden_size)), mstype.float16)
        >>> value = Tensor(np.ones((bsz, seq_len, hidden_size)), mstype.float16)
        >>> batch_valid_length = Tensor(np.ones((bsz, 1)), mstype.int32)
        >>> block_tables = Tensor(np.ones((bsz, num_blocks)), mstype.int64)
        >>> slot_mapping = Tensor(np.ones((bsz,)), mstype.int32)
        >>> freqs_cos = Tensor(np.ones((seq_len, head_dim)), mstype.float16)
        >>> freqs_sin = Tensor(np.ones((seq_len, head_dim)), mstype.float16)
        >>> attn_mask = Tensor(np.ones((bsz, 1, seq_len, seq_len)), mstype.uint8)
        >>> infer_attention = InferAttention(head_num,
                                             head_dim,
                                             n_kv_head,
                                             scale_value=1. / math.sqrt(head_dim),
                                             pre_tokens=65536,
                                             next_tokens=0,
                                             block_size=block_size,
                                             num_blocks=num_blocks)
        >>> output = infer_attention(query, key, value, batch_valid_length, block_tables, slot_mapping, attn_mask)
        >>> print(output.shape)
        (1, 4096, 2048)
    """

    def __init__(self,
                 n_head,
                 head_dim,
                 n_kv_head,
                 keep_prob=1.0,
                 scale_value=1.0,
                 pre_tokens=2147483647,
                 next_tokens=2147483647,
                 input_layout="BSH",
                 sparse_mode=0,
                 block_size=16,
                 num_blocks=1024,
                 use_attention_mask=False,
                 use_alibi_mask=False,
                 use_rope_rotary_emb=True,
                 rotary_cos_format=0,
                 parallel_config=None,
                 compute_dtype=mstype.float16
                 ):
        super(InferAttention, self).__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.n_kv_head = n_kv_head
        self.keep_prob = keep_prob
        self.scale_value = scale_value
        self.pre_tokens = pre_tokens
        self.next_tokens = next_tokens
        self.input_layout = input_layout
        self.sparse_mode = sparse_mode
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.use_attention_mask = use_attention_mask
        self.use_alibi_mask = use_alibi_mask
        self.use_rope_rotary_emb = use_rope_rotary_emb
        self.rotary_cos_format = rotary_cos_format
        self.parallel_config = parallel_config
        self.compute_dtype = compute_dtype
        self.is_first_iteration = True
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.batch_matmul = P.BatchMatMul()
        self.batch_matmul_q_k = P.BatchMatMul(transpose_b=True)
        self.mul = P.Mul()
        self.add = P.Add()
        self.shape = P.Shape()
        self.tile = P.Tile()
        self.softmax = P.Softmax()
        self.cast = P.Cast()
        self.inv_norm_factor = Tensor(1.0 / math.sqrt(self.head_dim), dtype=compute_dtype)
        self.n_rep = self.n_head // self.n_kv_head

        dp = 1 if parallel_config is None else parallel_config.data_parallel
        mp = 1 if parallel_config is None else parallel_config.model_parallel
        self.flash_attention = FlashAttention(head_num=self.n_head,
                                              pre_tokens=self.pre_tokens,
                                              next_tokens=self.next_tokens,
                                              input_layout=self.input_layout,
                                              keep_prob=self.keep_prob,
                                              scale_value=self.scale_value,
                                              sparse_mode=self.sparse_mode,
                                              use_attention_mask=self.use_attention_mask,
                                              use_alibi_mask=self.use_alibi_mask,
                                              dp=dp,
                                              mp=mp)

        self.paged_attention_mgr = PagedAttentionMgr(self.n_head,
                                                     self.head_dim,
                                                     n_kv_heads=self.n_kv_head,
                                                     block_size=self.block_size,
                                                     num_blocks=self.num_blocks,
                                                     compute_dtype=self.compute_dtype)
        self.paged_attention_mgr.shard(parallel_config)
        self.apply_rotary_pos_emb = ops.ApplyRotaryPosEmb(self.rotary_cos_format)
        self.apply_rotary_pos_emb.shard(((dp, 1, mp), (dp, 1, mp), (1, 1), (1, 1), (dp,)))

    def _core_attention(self, query, key, value, attn_mask, alibi_mask=None):
        """
        Get the weighted score along the seq_length

        Inputs:
            query: the query matrix
            key: the key matrix
            value: the value matrix
            attn_mask: the attention mask adder matrix with shape (batch_size,
            1, seq_length, seq_length)
            alibi_mask: the alibi matrix
        Outputs:
            weighted_values: Tensor, the weighted sum scores
        """
        # q, k: [bs, n_head, seq/1, head_dim], [bs, n_head, seq, head_dim]
        score = self.batch_matmul_q_k(query, key)
        # score: [bs, n_head, seq/1, seq]
        score = self.mul(score, self.inv_norm_factor)
        if alibi_mask is not None:
            score = self.add(score, alibi_mask)
        score = self.add(attn_mask, score)

        attention_probs = self.softmax(self.cast(score, mstype.float32))
        # score, v: [bs, n_head, seq/1, seq], [bs, n_head, seq, head_dim]
        weighted_values = self.batch_matmul(self.cast(attention_probs, self.compute_dtype), value)
        # [bs, n_head, seq/1, head_dim]
        weighted_values = self._merge_heads(weighted_values)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        return weighted_values

    def _repeat_kv(self, x, rep):
        if rep == 1:
            return x
        bs, n_kv_head, seqlen, head_dim = self.shape(x)
        x = self.reshape(x, (bs, n_kv_head, 1, seqlen * head_dim))
        x = self.tile(x, (1, 1, rep, 1))
        x = self.reshape(x, (bs, n_kv_head * rep, seqlen, head_dim))
        return x

    def _merge_heads(self, x):
        """
        convert a 4d input to a 2d or 3d output

        Inputs:
            x: input tensor

        Output:
            x_merge: the 2d output
        """
        # [bs, n_head, seq/1, head_dim]
        x = self.transpose(x, (0, 2, 1, 3))  # dp,mp,1,1 -> dp,1,mp,1
        # [bs, seq/1, n_head, head_dim]
        bs, seq_len, n_head, head_dim = self.shape(x)
        # [bs, seq/1, hidden_dim]
        new_shape = (bs, seq_len, n_head * head_dim)
        x_merge = self.reshape(x, new_shape)
        return x_merge

    def construct(self, query, key, value, batch_valid_length, block_tables, slot_mapping, freqs_cos=None,
                  freqs_sin=None, attn_mask=None, alibi_mask=None):
        """Forward process of the Infer Attention Cell"""
        if self.use_rope_rotary_emb:
            # ROPE currently only supported float16 data type.
            freqs_cos = self.cast(freqs_cos, mstype.float16)
            freqs_sin = self.cast(freqs_sin, mstype.float16)
            query, key = self.apply_rotary_pos_emb(query, key, freqs_cos, freqs_sin, batch_valid_length)
        key_out = self.paged_attention_mgr(key, value, slot_mapping)
        query = ops.depend(query, key_out)

        if self.is_first_iteration:
            if self.input_layout == "BSH":
                context_layer = self.flash_attention(query, key, value, attn_mask, alibi_mask)
            else:
                bs, seq_len, _ = query.shape
                query = self.transpose(self.reshape(query, (bs, seq_len, self.n_head, self.head_dim)), (0, 2, 1, 3))
                key = self.transpose(self.reshape(key, (bs, seq_len, self.n_kv_head, self.head_dim)), (0, 2, 1, 3))
                value = self.transpose(self.reshape(value, (bs, seq_len, self.n_kv_head, self.head_dim)), (0, 2, 1, 3))
                key = self._repeat_kv(key, self.n_rep)
                value = self._repeat_kv(value, self.n_rep)
                context_layer = self._core_attention(query, key, value, attn_mask, alibi_mask)
                return context_layer
        else:
            if self.use_alibi_mask:
                context_layer = self.paged_attention_mgr.paged_attn_with_alibi(query, batch_valid_length, block_tables,
                                                                               alibi_mask)
            else:
                context_layer = self.paged_attention_mgr.paged_attn(query, batch_valid_length, block_tables)
        return context_layer
