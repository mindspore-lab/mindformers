# Copyright 2026 Huawei Technologies Co., Ltd
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

import math
from typing import Union

import mindspore.common.dtype as mstype
import mindspore as ms
from mindspore import ops, mint
from mindspore.common.tensor import Tensor
from mindspore.nn.cell import Cell

from mindformers.parallel_core.transformer_config import TransformerConfig, MLATransformerConfig


class FlashAttention(Cell):
    """
    FlashAttention Layer.

    This class implements the FlashAttention mechanism for fast and memory-efficient attention computation.
    It supports multiple attention types, mask modes, and is optimized for parallel training including
    tensor and context parallelism.

    Reference:
        "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
        https://arxiv.org/abs/2205.14135

    Args:
        config (Union[TransformerConfig, MLATransformerConfig]): Configuration object containing model hyperparameters,
            including number of heads, and more.
        layer_number (int): The index of the current layer within the transformer stack.
        softmax_scale (float, optional): Scaling factor for the attention logits before softmax.
            If None, it defaults to 1 / sqrt(head_dim).

    Inputs:
        - **query** (Tensor): The query tensor with shape (B, S1, H1) or (B, N1, S1, D).
        - **key** (Tensor): The key tensor with shape (B, S2, H2) or (B, N2, S2, D).
        - **value** (Tensor): The value tensor with shape (B, S2, H2) or (B, N2, S2, D).
        - **attn_mask** (Tensor): Attention mask. A value of 0 keeps the element;
          a value of 1 masks it out. Shape can vary based on attention mode.
        - **alibi_mask** (Tensor, optional): Positional bias tensor for ALiBi attention.
          Used for large sequences and causal masks.
        - **prefix** (Tensor, optional): Prefix lengths for prefix attention mode.
          Not implemented yet.
        - **padding_mask** (None): Reserved for future use.
        - **actual_seq_qlen** (Tensor[int32], optional): Actual valid sequence lengths of the query.
        - **actual_seq_kvlen** (Tensor[int32], optional): Actual valid sequence lengths of the key/value.

    Outputs:
        - **attention_out** (Tensor): The attention output tensor with the same shape and type as `query`.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(self,
                 config: Union[TransformerConfig, MLATransformerConfig],
                 layer_number,
                 softmax_scale: float = None,
                 ):
        super().__init__()

        # FA (Flash Attention) is an optimized version of DotProductAttention in Megatron v0.12.0,
        # with nearly identical computational precision.

        self.config = config
        self.layer_number = max(1, layer_number)

        projection_size = self.config.kv_channels * self.config.num_attention_heads

        if config.multi_latent_attention:
            hidden_size_per_attention_head = config.qk_head_dim + config.qk_pos_emb_head_dim
        else:
            hidden_size_per_attention_head = projection_size // config.num_attention_heads

        # MindSpore FlashAttentionScore
        self.head_num = config.num_attention_heads
        self.input_layout = config.input_layout
        self.sparse_mode = config.sparse_mode
        self.pre_tokens = 2147483647
        self.next_tokens = 0
        self.scalar_value = 1. / math.sqrt(hidden_size_per_attention_head) if softmax_scale is None else softmax_scale
        self.inner_precise = 0

        self.flash_attention = ops.flash_attention_score

        # Note: only support config.apply_query_key_layer_scaling be set False
        # FusedScaleMaskSoftmax does not require implementation.

        self.use_alibi_mask = config.use_alibi_mask

        if self.use_alibi_mask:
            self.alibi_rescale_factor = Tensor([1.0 / self.scalar_value], dtype=mstype.float16)
            self.alibi_rescale_mul = mint.mul

        self.bnsd_transpose = mint.permute
        self.bsh_transpose = mint.permute
        self.merge_head_transpose = mint.permute
        self.reshape = mint.reshape
        self.fa_out_transpose = mint.permute
        self.cast = ops.cast

    def construct(self,
                  query: Tensor,
                  key: Tensor,
                  value: Tensor,
                  attention_mask: Tensor,
                  alibi_mask=None,
                  prefix=None,
                  padding_mask=None,
                  actual_seq_qlen=None,
                  actual_seq_kvlen=None):
        """Forward process of the AttentionMaskMF"""
        if attention_mask is not None:
            attention_mask = self.cast(attention_mask, ms.uint8)

        if self.input_layout == "TND":
            output = self.flash_attention(query=query,
                                          key=key,
                                          value=value,
                                          head_num=self.head_num,
                                          real_shift=alibi_mask,
                                          padding_mask=padding_mask,
                                          attn_mask=attention_mask,
                                          prefix=prefix,
                                          actual_seq_qlen=actual_seq_qlen,
                                          actual_seq_kvlen=actual_seq_kvlen,
                                          scalar_value=self.scalar_value,
                                          pre_tokens=self.pre_tokens,
                                          next_tokens=self.next_tokens,
                                          inner_precise=self.inner_precise,
                                          input_layout=self.input_layout,
                                          sparse_mode=self.sparse_mode)
            return output

        q_seq_len, bsz = query.shape[:2]
        kv_seq_len = key.shape[0]
        if self.input_layout == "BNSD":
            query = self.bnsd_transpose(query, (1, 2, 0, 3))
            key = self.bnsd_transpose(key, (1, 2, 0, 3))
            value = self.bnsd_transpose(value, (1, 2, 0, 3))
        elif self.input_layout == "BSH":
            query = self.bsh_transpose(query, (1, 0, 2))
            key = self.bsh_transpose(key, (1, 0, 2))
            value = self.bsh_transpose(value, (1, 0, 2))
        else:
            query = self.reshape(query, (q_seq_len, bsz, -1))
            key = self.reshape(key, (kv_seq_len, bsz, -1))
            value = self.reshape(key, (kv_seq_len, bsz, -1))
        if self.use_alibi_mask:
            alibi_mask = self.alibi_rescale_mul(alibi_mask, self.cast(self.alibi_rescale_factor, alibi_mask.dtype))

        output = self.flash_attention(query=query,
                                      key=key,
                                      value=value,
                                      head_num=self.head_num,
                                      real_shift=alibi_mask,
                                      padding_mask=padding_mask,
                                      attn_mask=attention_mask,
                                      prefix=prefix,
                                      scalar_value=self.scalar_value,
                                      pre_tokens=self.pre_tokens,
                                      next_tokens=self.next_tokens,
                                      inner_precise=self.inner_precise,
                                      input_layout=self.input_layout,
                                      sparse_mode=self.sparse_mode)
        if self.input_layout == "BNSD":
            output = self._merge_heads(output)
        elif self.input_layout == "BSH":
            output = self.fa_out_transpose(output, (1, 0, 2))
        return output

    def _merge_heads(self, x):
        """
        Convert a 4D input tensor to a 3D output tensor.

        Inputs:
            x: input tensor

        Output:
            x_merge: the 3D output tensor
        """
        x = self.merge_head_transpose(x, (0, 2, 1, 3))  # dp,tp,cp,1 -> dp,cp,tp,1
        bs, seq_len, n_head, head_dim = x.shape
        new_shape = (bs, seq_len, n_head * head_dim)
        x_merge = self.reshape(x, new_shape)
        x_merge = self.fa_out_transpose(x_merge, (1, 0, 2))
        return x_merge
