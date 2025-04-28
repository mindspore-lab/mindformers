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
"""SelfAttention."""
from abc import abstractmethod
import math
from typing import Union
from mindspore import Tensor, mint, nn, ops

from mindformers.experimental.infer.core import get_attn_mask_func
from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig
from mindformers.experimental.infer.transformer.rotary_embedding import RotaryEmbedding
from mindformers.experimental.infer.core.utils import get_tp_world_size
from mindformers.experimental.graph.transformer.spec_utils import (
    ModuleSpec, build_module
)
from mindformers.experimental.parallel_core.pynative.transformer.scale_mask_softmax import (
    ScaleMaskSoftmax
)
from mindformers.experimental.parallel_core.pynative.utils import divide

__all__ = [
    'SelfAttentionSubmodules',
    'DotProductAttention',
    'Attention',
    'SelfAttention'
]


class SelfAttentionSubmodules:
    """Configuration class for specifying the submodules of a self-attention."""

    def __init__(self,
                 linear_qkv: Union[ModuleSpec, type] = None,
                 core_attention: Union[ModuleSpec, type] = None,
                 linear_proj: Union[ModuleSpec, type] = None,
                 q_layernorm: Union[ModuleSpec, type] = None,
                 k_layernorm: Union[ModuleSpec, type] = None):
        self.linear_qkv = linear_qkv
        self.core_attention = core_attention
        self.linear_proj = linear_proj
        self.q_layernorm = q_layernorm
        self.k_layernorm = k_layernorm


class DotProductAttention(nn.Cell):
    """
    Get the weighted score along the seq_length.

    Args:
        layer_number (int): Number which indicates the index of this transformer layer in the
            whole transformer block.
        config (dict): Configuration.
        attn_type (str): Attention type. Support ['self_attn', 'cross_attn']. Default: 'self_attn'.

    Inputs:
        - **query** (Tensor) - Tensor of query matrix.
        - **key** (Tensor) - Tensor of key matrix.
        - **value** (Tensor) - Tensor of value matrix.
        - **attention_mask** (Tensor) - Tensor of attention mask matrix.

    Outputs:
        - **attn_output** (Tensor) - Tensor of shape :math:`(B, S, H)`.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(self, config: TransformerConfig, layer_number, attn_mask_type=None):
        super().__init__()
        if attn_mask_type:
            raise NotImplementedError(
                "For CoreAttention, `attn_mask_type` is not supported for now."
            )
        if config.context_parallel > 1:
            raise NotImplementedError(
                "For CoreAttention, cp is not supported for now."
            )

        self.config = config
        self.layer_number = max(1, layer_number)
        self.compute_dtype = self.config.compute_dtype
        self.softmax_compute_dtype = self.config.softmax_compute_dtype

        self.apply_query_key_layer_scaling = self.config.apply_query_key_layer_scaling
        self.num_heads = self.config.num_attention_heads
        self.query_projection_size = self.config.hidden_size
        self.hidden_size_per_attention_head = getattr(config, 'kv_channels', divide(
            self.query_projection_size, self.num_heads
        ))
        self.num_query_groups = (self.num_heads
                                 if config.num_query_groups is None else
                                 config.num_query_groups)
        self.use_gqa = (self.num_heads != self.num_query_groups)
        if self.use_gqa:
            self.repeat_num = divide(self.num_heads, self.num_query_groups)

        self.tp_group_size = get_tp_world_size()
        self.hidden_size_per_partition = divide(self.query_projection_size,
                                                self.tp_group_size)

        coeff = None
        self.softmax_scale = Tensor(1.0 / math.sqrt(self.hidden_size_per_attention_head), dtype=self.compute_dtype)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.softmax_scale /= coeff

        self.mask_func = get_attn_mask_func(self.config.mask_func_type)
        self.scale_mask_softmax = ScaleMaskSoftmax(
            self.mask_func, softmax_compute_type=self.softmax_compute_dtype)

        self.attention_dropout = mint.nn.Dropout(
            p=self.config.attention_dropout_rate)

    def construct(self, query_layer, key_layer, value_layer, attention_mask):
        """Forward process of the CoreAttention."""
        bs, seq_len, _ = query_layer.shape
        # [B, S, H] --> [B, S, N, D]
        query_layer = query_layer.reshape(bs, seq_len, -1, self.hidden_size_per_attention_head)
        key_layer = key_layer.reshape(bs, seq_len, -1, self.hidden_size_per_attention_head)
        value_layer = value_layer.reshape(bs, seq_len, -1, self.hidden_size_per_attention_head)
        # [B, S, N_kv, D] --> [B, S, N, D]
        if self.use_gqa:
            key_layer = mint.repeat_interleave(key_layer,
                                               repeats=self.repeat_num,
                                               dim=2)
            value_layer = mint.repeat_interleave(value_layer,
                                                 repeats=self.repeat_num,
                                                 dim=2)
        # [B, S, N, D] --> [B, N, S, D]
        query_layer = mint.transpose(query_layer, -3, -2)
        key_layer = mint.transpose(key_layer, -3, -2)
        value_layer = mint.transpose(value_layer, -3, -2)
        # [B, N, S, D] --> [B * N, S, D]
        query_layer = query_layer.reshape(-1, seq_len, self.hidden_size_per_attention_head)
        key_layer = key_layer.reshape(-1, seq_len, self.hidden_size_per_attention_head)
        value_layer = value_layer.reshape(-1, seq_len, self.hidden_size_per_attention_head)

        # score: [B * N, S_q, S_k]
        score = mint.bmm(query_layer, mint.transpose(key_layer, -2, -1))
        score = mint.mul(score, self.softmax_scale)

        # attention scores and attention mask [B * N, S_q, S_k]
        attention_probs = self.scale_mask_softmax(score, attention_mask)

        attention_probs = self.attention_dropout(attention_probs)

        # [B * N, S_q, S_k] * [B * N, S_v, D] -> [B * N, S_q, D]
        core_attn_out = mint.bmm(attention_probs, value_layer)
        # [B * N, S_q, D] -> [B, N, S_q, D]
        core_attn_out = core_attn_out.reshape(bs, -1, seq_len, self.hidden_size_per_attention_head)

        core_attn_out = mint.transpose(core_attn_out, -3, -2).reshape(
            bs, seq_len, self.hidden_size_per_partition)
        return core_attn_out


class Attention(nn.Cell):
    """
    Attention block.

    Args:
        layer_index (int): Number which indicates the index of this transformer layer in the
            whole transformer block.
        submodules (SelfAttentionSubmodules): submodules of SelfAttention
        config (TransformerConfig): Configuration.

    Inputs:
        - **hidden_states** (Tensor) - Tensor of shape :math:`(B, S, H)`.
        - **attention_mask** (Tensor) - Tensor of attention mask.
        - **encoder_output** (Tensor) - Tensor of encoder output used for cross attention. Default: None.
        - **rotary_pos_emb** (Tensor) - Tensor of rotary position embedding. Default: None.

    Outputs:
        - **output** (Tensor) - Tensor of shape :math:`(B, S, H)`.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(
            self,
            config,
            submodules: Union[SelfAttentionSubmodules],
            layer_number: int,
            attn_mask_type: str = None,
            attention_type: str = None,
            cp_comm_type: str = None
    ):
        super().__init__(config)

        if attn_mask_type:
            raise NotImplementedError(
                "For Attention, 'attn_mask_type' is not supported for now."
            )
        if attention_type:
            raise NotImplementedError(
                "For Attention, 'attention_type' is not supported for now."
            )
        if cp_comm_type:
            raise NotImplementedError(
                "For Attention, 'cp_comm_type' is not supported for now."
            )

        self.config = config
        self.submodules = submodules
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type

        self.param_init_type = self.config.param_init_type
        self.compute_dtype = self.config.compute_dtype
        self.is_prefill = True

        self.num_heads = self.config.num_attention_heads
        self.num_query_groups = (self.num_heads
                                 if config.num_query_groups is None else
                                 config.num_query_groups)
        self.query_projection_size = self.config.hidden_size
        self.hidden_size_per_attention_head = getattr(config, 'kv_channels', divide(
            self.query_projection_size, self.num_heads
        ))
        self.kv_projection_size = self.hidden_size_per_attention_head * self.num_query_groups
        self.n_rep = divide(self.num_heads, self.num_query_groups)

        self.use_flash_attention = self.config.use_flash_attention
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)

        self.tp_group_size = get_tp_world_size()

        self.cast = ops.Cast()

        self.key_hidden_size = self.hidden_size_per_attention_head
        self.val_hidden_size = self.hidden_size_per_attention_head

        self.num_attention_heads_per_partition = divide(self.num_heads,
                                                        self.tp_group_size)

        self.use_gqa = (self.num_heads != self.num_query_groups)

        if self.use_gqa:
            self._check_gqa_valid()
            self.num_query_groups_per_partition = divide(self.num_query_groups,
                                                         self.tp_group_size)
            self.repeat_num = divide(self.num_heads, self.num_query_groups)
        else:
            self.num_query_groups_per_partition = self.num_attention_heads_per_partition

        self.hidden_size_per_partition = divide(self.query_projection_size,
                                                self.tp_group_size)
        self.kv_hidden_size_per_partition = divide(self.kv_projection_size,
                                                   self.tp_group_size)

        if self.use_flash_attention:
            kv_cache_shape = (self.config.num_blocks, self.config.block_size,
                              self.num_query_groups_per_partition, self.hidden_size_per_attention_head)
            self.flash_attention = build_module(
                submodules.core_attention,
                head_num=self.num_attention_heads_per_partition,
                kv_cache_shape=kv_cache_shape,
                head_dim=self.hidden_size_per_attention_head,
                kv_head_num=self.num_query_groups_per_partition,
                scale_value=1.0 / self.norm_factor,
                next_tokens=0,
                compute_dtype=self.compute_dtype,
            )
        else:
            self.core_attention = build_module(submodules.core_attention,
                                               config=self.config,
                                               layer_number=self.layer_number)

        self.rotary_embedding = RotaryEmbedding(kv_channels=self.hidden_size_per_attention_head,
                                                rotary_cos_format=2)

        # submodules.linear_proj: RowParallelLinear
        self.linear_proj = build_module(
            submodules.linear_proj,
            self.query_projection_size,
            self.query_projection_size,
            input_is_parallel=True,
            config=self.config,
            bias=self.config.out_proj_has_bias,
            transpose_b=True,
            param_init_type=self.param_init_type,
            compute_dtype=self.compute_dtype,
        )

    def _check_gqa_valid(self):
        """check whether the config is valid for grouped-query-attention"""
        if self.num_heads % self.num_query_groups != 0:
            raise ValueError(f"num_heads must be divisible by kv_num_heads, "
                             f"but got num_heads {self.num_heads} "
                             f"and kv_num_heads {self.num_query_groups}")
        if self.num_query_groups % self.tp_group_size != 0:
            raise ValueError(
                f"kv_num_heads must be divisible by tp_group_size, "
                f"but got kv_num_heads {self.num_query_groups} "
                f"and kv_num_heads {self.tp_group_size}")

    def construct(self,
                  x,
                  batch_valid_length=None,
                  kv_cache=None,
                  block_tables=None,
                  slot_mapping=None,
                  rotary_pos_cos=None,
                  rotary_pos_sin=None,
                  actual_seq_qlen=None,
                  actual_seq_kvlen=None,
                  context_lens_tensor=None,
                  q_seq_lens=None,
                  attn_mask=None,
                  alibi_mask=None):
        """Forward process of the SelfAttention."""
        # hidden_states: [B, S, H]
        ori_dtype = x.dtype

        # apply query, key, value projection
        query, key, value = self.get_query_key_value_tensors(x)

        # [B, S, H]
        if rotary_pos_cos is not None and rotary_pos_sin is not None:
            query, key = self.rotary_embedding(query, key, rotary_pos_cos,
                                               rotary_pos_sin,
                                               batch_valid_length)

        if self.use_flash_attention:
            core_attn_out = self.flash_attention(
                query, key, value, kv_cache, slot_mapping, block_tables,
                batch_valid_length, context_lens_tensor, q_seq_lens,
                actual_seq_qlen, actual_seq_kvlen, attn_mask, alibi_mask)
        else:
            core_attn_out = self.core_attention(query, key, value, attn_mask)

        # apply output projection
        output = self.linear_proj(core_attn_out)
        output = self.cast(output, ori_dtype)
        return output

    @abstractmethod
    def get_query_key_value_tensors(self, x):
        """
        This method needs to be implemented based on whether the derived class
        is "self-attn" or "cross-attn".
        """


class SelfAttention(Attention):
    """
    SelfAttention block.

    Args:
        layer_index (int): Number which indicates the index of this transformer layer in the
            whole transformer block.
        submodules (SelfAttentionSubmodules): submodules of SelfAttention
        config (TransformerConfig): Configuration.

    Inputs:
        - **hidden_states** (Tensor) - Tensor of shape :math:`(B, S, H)`.
        - **attention_mask** (Tensor) - Tensor of attention mask.
        - **encoder_output** (Tensor) - Tensor of encoder output used for cross attention. Default: None.
        - **rotary_pos_emb** (Tensor) - Tensor of rotary position embedding. Default: None.

    Outputs:
        - **output** (Tensor) - Tensor of shape :math:`(B, S, H)`.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(self,
                 config,
                 submodules: SelfAttentionSubmodules,
                 layer_number,
                 attn_mask_type=None,
                 cp_comm_type: str = None):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=None,
            cp_comm_type=cp_comm_type
        )
        if attn_mask_type:
            raise NotImplementedError(
                "For SelfAttention, `attn_mask_type` is not supported for now."
            )
        if cp_comm_type:
            raise NotImplementedError(
                "For SelfAttention, 'cp_comm_type' is not supported for now."
            )

        self.linear_qkv = build_module(
            self.submodules.linear_qkv,
            self.query_projection_size,
            self.query_projection_size + 2 * self.kv_projection_size,
            config=self.config,
            bias=self.config.qkv_has_bias,
            gather_output=False,
            transpose_b=True,
            param_init_type=self.param_init_type,
            compute_dtype=self.compute_dtype,
        )

        if submodules.q_layernorm is not None:
            self.q_layernorm = build_module(
                self.submodules.q_layernorm,
                self.hidden_size_per_attention_head,
                eps=self.config.layernorm_epsilon,
                compute_type=self.config.layernorm_compute_type,
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                self.submodules.k_layernorm,
                self.hidden_size_per_attention_head,
                eps=self.config.layernorm_epsilon,
                compute_type=self.config.layernorm_compute_type,
            )
        else:
            self.k_layernorm = None

        self.cast = ops.Cast()

    def get_query_key_value_tensors(self, x):
        qkv = self.cast(self.linear_qkv(x), self.compute_dtype)
        query, key, value = mint.split(qkv,
                                       (self.hidden_size_per_partition,
                                        self.kv_hidden_size_per_partition,
                                        self.kv_hidden_size_per_partition), -1)

        if self.q_layernorm is not None:
            orig_query_shape = query.shape
            query = self.q_layernorm(query.reshape(x.shape[:-1] + (-1, self.hidden_size_per_attention_head,)))
            query = query.reshape(orig_query_shape)

        if self.k_layernorm is not None:
            orig_query_shape = key.shape
            key = self.k_layernorm(key.reshape(x.shape[:-1] + (-1, self.hidden_size_per_attention_head,)))
            key = key.reshape(orig_query_shape)

        return query, key, value
