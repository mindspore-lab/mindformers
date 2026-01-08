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
"""Transformer Attention"""
import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import Union
from mindspore import nn, ops, mint
import mindspore.common.dtype as mstype

from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.pynative.base_models.common.embeddings.rope_utils import ApplyRotaryPosEmb


@dataclass
class SelfAttentionSubmodules:
    """
    Configuration class for specifying the submodules of a self-attention.
    """

    linear_qkv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None


class Attention(nn.Cell):
    """Attention layer abstract class.

    This layer only contains common modules required for the "self attn" and
    "cross attn" specializations.

    Args:
        config (TransformerConfig): The config of the transformer model.
        submodules (SelfAttentionSubmodules): The submodules used to construct
            the Attention layer, such as ColumnParallelLinear and RowParallelLinear for query and
            key-value projections.
        layer_number (int): Number which indicates the index of this transformer layer in the
            whole transformer block.

    Inputs:
        - **hidden_states** (Tensor) - Tensor of shape :math:`(B, S, H)`.
        - **attention_mask** (Tensor) - Tensor of attention mask.
        - **key_value_states** (Tensor) - Tensor of encoder output used for cross attention. Default: None.
        - **rotary_pos_emb** (Tensor) - Tensor of rotary position embedding. Default: None.
        - **prefix_keys_values** (Union[Tensor[float16, bfloat16], None]) - The prefix keys values.
        - **actual_seq_len** (Tensor[int32], optional):  Used to automatically generate attention mask.

    Outputs:
        - **output** (Tensor) - Tensor of shape :math:`(B, S, H)`.
        - **bias** (Tensor) - Tensor of shape :math:`(B, S, H)`.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(
            self,
            config: TransformerConfig,
            submodules: Union[SelfAttentionSubmodules],
            layer_number: int,
    ):
        super().__init__()

        self.config = config
        self.init_method = config.init_method
        self.output_layer_init_method = config.output_layer_init_method
        self.compute_dtype = self.config.compute_dtype
        self.hidden_size = self.config.hidden_size
        self.use_flash_attention = self.config.use_flash_attention
        self.parallel_config = self.config
        self.use_attn_mask_compression = self.config.use_attn_mask_compression
        self.input_layout = config.input_layout

        # For normal attention without groups, num_query_groups == num_attention_heads,
        # so these two will be the same
        self.use_gqa = self.config.num_query_groups < self.config.num_attention_heads
        self.num_heads = self.config.num_attention_heads
        self.kv_num_heads = self.config.num_query_groups if self.use_gqa else self.num_heads
        self.head_dim = self.config.kv_channels
        self.q_hidden_size = self.head_dim * self.num_heads
        self.kv_hidden_size = self.head_dim * self.kv_num_heads

        # Not Support Graph Mode and key/value with different hidden size for now.
        # attention_hidden_size and num_attention_heads must be evenly divisible
        # by num_heads and tp respectively to enable correct tensor splitting.

        self.n_rep = self.num_heads // self.kv_num_heads
        self.layer_number = max(1, layer_number)
        self.norm_factor = math.sqrt(self.head_dim)
        self.seq_length = config.seq_length
        self.pre_tokens = 2147483647 if self.config.attention_pre_tokens is None else self.config.attention_pre_tokens
        self.next_tokens = 0 if self.config.attention_next_tokens is None else self.config.attention_next_tokens
        self.keep_prob = 1.0 if self.config.attention_dropout is None else 1 - self.config.attention_dropout
        self.use_attention_mask = True if self.config.use_attention_mask is None else self.config.use_attention_mask

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"For 'MultiHeadAttention', the class variable 'hidden_size' must be a multiple "
                             f"of 'num_heads', but got the hidden_size is {self.hidden_size} "
                             f"and the num_heads is {self.num_heads}.")

        self.core_attention = build_module(
            submodules.core_attention,
            config=self.config,
            layer_number=self.layer_number,
        )

        # Output
        self.linear_proj = build_module(
            submodules.linear_proj,
            input_size=self.q_hidden_size,
            output_size=self.config.hidden_size,
            compute_dtype=self.compute_dtype,
            params_dtype=self.config.params_dtype,
            init_method=self.output_layer_init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
        )

        self.apply_rotary_pos_emb = ApplyRotaryPosEmb(self.parallel_config)

        # after rotary
        self.split_qkv = mint.split
        self.cast = ops.cast
        self.reshape = mint.reshape
        self.bs_transpose = mint.permute
        self.tnd_transpose = mint.permute
        self.tile_kv = mint.tile
        self.cat = mint.concat

    @abstractmethod
    def get_query_key_value_tensors(self, hidden_states, key_value_states):
        """
        This method needs to be implemented based on whether the derived class
        is "self-attn" or "cross-attn".
        """

    def sbh2tnd(self, x, num_heads):
        """
        Convert a input tensor from SBH/SBND layout to TND layout.

        Inputs:
            x: input tensor
            num_head: the attention head number of x

        Output:
            x_merge: the TND output tensor
        """
        seq_len, bs = x.shape[:2]
        x = self.reshape(x, (seq_len, bs, num_heads, -1))
        x = self.tnd_transpose(x, (1, 0, 2, 3))
        x = self.reshape(x, (bs * seq_len, num_heads, -1))
        return x

    def construct(
            self,
            hidden_states,
            attention_mask,
            key_value_states=None,
            rotary_pos_emb=None,
            prefix_keys_values=None,
            actual_seq_len=None
    ):
        """ Construct function of attention block."""
        ori_dtype = hidden_states.dtype
        seq_len, bs, _ = hidden_states.shape

        # apply query, key, value projection
        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)

        # transpose and reshape
        query = self.reshape(query, (seq_len, bs, self.num_heads, self.head_dim))
        key = self.reshape(key, (seq_len, bs, self.kv_num_heads, self.head_dim))

        # apply rotary position embedding
        if rotary_pos_emb is not None:
            query = self.apply_rotary_pos_emb(query, rotary_pos_emb)
            key = self.apply_rotary_pos_emb(key, rotary_pos_emb)

        value = self.reshape(value, (seq_len, bs, self.kv_num_heads, self.head_dim))
        key, value = self._cat_prefix(key, value, prefix_keys_values)

        if self.input_layout == "TND":
            query = self.sbh2tnd(query, self.num_heads)
            key = self.sbh2tnd(key, self.kv_num_heads)
            value = self.sbh2tnd(value, self.kv_num_heads)

        if not self.use_flash_attention:
            key = self._repeat_kv(key, self.n_rep)
            value = self._repeat_kv(value, self.n_rep)
            context_layer = self.core_attention(query, key, value, attention_mask)
        else:
            if attention_mask is not None:
                attention_mask = self.cast(attention_mask, mstype.uint8)

            if query.dtype not in (mstype.float16, mstype.bfloat16):
                query = self.cast(query, mstype.float16)
            if key.dtype not in (mstype.float16, mstype.bfloat16):
                key = self.cast(key, mstype.float16)
            if value.dtype not in (mstype.float16, mstype.bfloat16):
                value = self.cast(value, mstype.float16)

            output = self.core_attention(
                query, key, value, attention_mask,
                actual_seq_qlen=actual_seq_len, actual_seq_kvlen=actual_seq_len
            )

            if self.input_layout == "TND":
                output = self.reshape(output, (bs, seq_len, -1))
                context_layer = self.bs_transpose(output, (1, 0, 2))
            else:
                context_layer = self.cast(output, self.compute_dtype)

        # apply output projection
        output, bias = self.linear_proj(context_layer)
        output = self.cast(output, ori_dtype)

        return output, bias

    def _cat_prefix(self, key, value, prefix_keys_values):
        '''
        Concatenate prefix_keys_values to key and value.
        prefix_keys_values: shape (2, bs, pre_len, num_heads * kv_channels)
        '''
        if prefix_keys_values is not None:
            _, bs, n_kv_head, head_dim = key.shape
            past_key = prefix_keys_values[0]
            past_value = prefix_keys_values[1]
            past_key = self.reshape(past_key, (-1, bs, n_kv_head, head_dim))
            past_value = self.reshape(past_value, (-1, bs, n_kv_head, head_dim))
            past_key = self.cast(past_key, self.compute_dtype)
            past_value = self.cast(past_value, self.compute_dtype)
            key = self.cat((past_key, key), dim=2)
            value = self.cat((past_value, value), dim=2)
        return key, value

    def _repeat_kv(self, x, rep):
        if rep == 1:
            return x
        bs, n_kv_head, seqlen, head_dim = x.shape
        x = self.reshape(x, (bs, n_kv_head, 1, seqlen * head_dim))
        x = self.tile_kv(x, (1, 1, rep, 1))
        x = self.reshape(x, (bs, n_kv_head * rep, seqlen, head_dim))
        return x


class SelfAttention(Attention):
    """Self-attention layer class

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.

    Args:
        config (TransformerConfig): The config of the transformer model.
        submodules (SelfAttentionSubmodules): The submodules used to construct the SelfAttention layer,
            such as ColumnParallelLinear and RowParallelLinear for query and key-value projections.
        layer_number (int): Number which indicates the index of this transformer layer in the
            whole transformer block.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(
            self,
            config: TransformerConfig,
            submodules: SelfAttentionSubmodules,
            layer_number: int,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
        )

        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.hidden_size,
            self.q_hidden_size + 2 * self.kv_hidden_size,
            compute_dtype=self.compute_dtype,
            params_dtype=self.config.params_dtype,
            init_method=self.init_method,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
        )

        if submodules.q_layernorm is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                dim=self.head_dim,
                compute_dtype=self.compute_dtype,
                params_dtype=self.config.params_dtype,
                eps=self.config.layernorm_epsilon
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                compute_dtype=self.compute_dtype,
                params_dtype=self.config.params_dtype,
                dim=self.head_dim,
                eps=self.config.layernorm_epsilon
            )
        else:
            self.k_layernorm = None

        self.reshape_concat = mint.reshape

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        seq_len, bs, _ = hidden_states.shape

        qkv, _ = self.linear_qkv(hidden_states)
        qkv = self.cast(qkv, self.compute_dtype)
        new_tensor_shape = (seq_len, bs, -1, (self.n_rep + 2) * self.head_dim)
        mixed_x_layer = self.reshape_concat(qkv, new_tensor_shape)
        query, key, value = self.split_qkv(mixed_x_layer,
                                           (self.head_dim * self.n_rep, self.head_dim, self.head_dim), 3)

        if self.q_layernorm is not None:
            orig_query_shape = query.shape
            query = self.q_layernorm(query.reshape(hidden_states.shape[:-1] +
                                                   (-1, self.head_dim,)))
            query = query.reshape(orig_query_shape)

        if self.k_layernorm is not None:
            orig_query_shape = key.shape
            key = self.k_layernorm(key.reshape(hidden_states.shape[:-1] +
                                               (-1, self.head_dim,)))
            key = key.reshape(orig_query_shape)

        return query, key, value
