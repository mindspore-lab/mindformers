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
"""Multi-Latent Attention."""

__all__ = [
    'MLASelfAttentionSubmodules',
    'MultiLatentAttention',
    'MLASelfAttention'
]

import math
from dataclasses import dataclass
from typing import Union

from mindspore import mint
from mindspore.ops import operations as P

from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.inference.utils import divide, get_tp_world_size
from mindformers.parallel_core.transformer_config import MLATransformerConfig
from mindformers.parallel_core.inference.tensor_parallel.mappings import GatherFromModelParallelRegion
from mindformers.parallel_core.inference.transformer.attention import Attention
from mindformers.parallel_core.inference.base_models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from mindformers.parallel_core.inference.base_models.common.embeddings.yarn_rotary_pos_embedding import (
    YaRNScalingRotaryEmbedding,
    _yarn_get_mscale
)


@dataclass
class MLASelfAttentionSubmodules:
    """Submodules for the MLA self-attention layer."""

    linear_q_proj: Union[ModuleSpec, type] = None
    linear_qkv_down_proj: Union[ModuleSpec, type] = None
    linear_q_up_proj: Union[ModuleSpec, type] = None
    linear_kv_down_proj: Union[ModuleSpec, type] = None
    linear_kv_up_proj: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    kv_layernorm: Union[ModuleSpec, type] = None


#pylint: disable=W0223
#pylint: disable=E1121
class MultiLatentAttention(Attention):
    """Multi-Latent Attention layer abstract class.

    This layer only contains common modules required for the "self attn" and
    "cross attn" specializations.
    """
    def __init__(
            self,
            config: MLATransformerConfig,
            submodules: Union[MLASelfAttentionSubmodules],
            layer_number: int,
            attn_mask_type: str = None,
            attention_type: str = None,
            cp_comm_type: str = None
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            cp_comm_type=cp_comm_type
        )
        if attn_mask_type:
            raise NotImplementedError(
                "For MultiLatentAttention, `attn_mask_type` is not supported for now."
            )
        if attention_type:
            raise NotImplementedError(
                "For MultiLatentAttention, `attention_type` is not supported for now."
            )
        if cp_comm_type is not None:
            raise NotImplementedError(
                "For MultiLatentAttention, `cp_comm_type` is not supported for now."
            )

        self.config = config
        self.use_flash_attention = self.config.use_flash_attention
        self.query_projection_size = self.config.v_head_dim * self.config.num_attention_heads
        self.q_head_dim = self.config.qk_head_dim + self.config.qk_pos_emb_head_dim
        # Overwrite the base class kv shape to support MLA inference
        self.key_hidden_size = self.q_head_dim
        self.val_hidden_size = self.config.v_head_dim

        self.num_attention_heads = self.config.num_attention_heads
        self.compute_dtype = self.config.compute_dtype
        self.cast = P.Cast()
        self.depend = P.Depend()
        self.dim_slice_4d = P.Slice()

        self.gather_from_mp_region = GatherFromModelParallelRegion()

        mscale = _yarn_get_mscale(self.config.rotary_scaling_factor, self.config.mscale)
        self.softmax_scale = mscale * mscale / math.sqrt(self.q_head_dim)

        if self.config.rope_type == "rope":
            self.rotary_pos_emb = RotaryEmbedding(
                self.config.qk_pos_emb_head_dim,
                rotary_percent=self.config.rotary_percent,
                rotary_base=self.config.rotary_base,
                rotary_cos_format=2,
            )
        elif self.config.rope_type == "yarn":
            self.rotary_pos_emb = YaRNScalingRotaryEmbedding(
                self.config.qk_pos_emb_head_dim,
                rotary_base=self.config.rotary_base,
                scaling_factor=self.config.rotary_scaling_factor,
                original_max_position_embeddings=self.config.max_position_embeddings,
                beta_fast=self.config.beta_fast,
                beta_slow=self.config.beta_slow,
                mscale=self.config.mscale,
                mscale_all_dim=self.config.mscale_all_dim,
                rotary_cos_format=2,
            )
        else:
            raise ValueError(
                f"Unsupported RoPE type: {self.config.rope_type}, supported types are "
                "'rope' and 'yarn'"
            )

        self.tp_group_size = get_tp_world_size()
        self.num_attention_heads_per_partition = divide(self.num_attention_heads,
                                                        self.tp_group_size)
        self.hidden_size_per_attention_head = getattr(config, 'kv_channels', divide(
            self.config.hidden_size, self.num_attention_heads
        ))
        self.kv_num_heads_per_partition = self.num_attention_heads_per_partition

        if self.use_flash_attention:
            self.core_attention = build_module(
                submodules.core_attention,
                head_num=self.num_attention_heads_per_partition,
                head_dim=self.q_head_dim,
                kv_head_num=self.kv_num_heads_per_partition,
                scale_value=self.softmax_scale,
                next_tokens=0,
                pa_kv_head_num=1,
                pa_mla_v_dim=512
                )
        else:
            raise NotImplementedError("For MLA, only `use_flash_attention=True` is supported")

        # RowParallelLinear
        self.linear_proj = build_module(
            submodules.linear_proj,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
            transpose_b=True,
            compute_dtype=self.compute_dtype
        )

    def construct(
            self,
            hidden_states,
            attention_mask,
            key_value_states=None,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            position_ids=None,
            batch_valid_length=None,
            block_tables=None,
            slot_mapping=None,
            q_seq_lens=None,
            actual_seq_qlen=None,
            actual_seq_kvlen=None,
            context_lens_tensor=None,
            key_cache=None,
            value_cache=None
    ):
        """Forward process of the CoreAttention."""

        query, key, value, out_absorb = self.get_query_key_value_tensors(
            hidden_states,
            batch_valid_length,
            rotary_pos_cos,
            rotary_pos_sin,
            slot_mapping,
            key_cache
        )

        # ==================================
        # core attention computation
        # ==================================
        bs, seq_len, _ = query.shape
        if self.is_prefill:
            core_attn_out = self.core_attention(
                query, key, value, slot_mapping, block_tables, batch_valid_length,
                context_lens_tensor, q_seq_lens, actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen, attn_mask=attention_mask,
                key_cache=key_cache, value_cache=value_cache)

            core_attn_out = core_attn_out.reshape(bs, seq_len, self.num_attention_heads_per_partition, self.q_head_dim)
            core_attn_out = self.dim_slice_4d(core_attn_out, (0, 0, 0, 0),
                                              (bs, seq_len, self.num_attention_heads_per_partition,
                                               self.config.v_head_dim))
            core_attn_out = core_attn_out.reshape(bs, seq_len,
                                                  self.num_attention_heads_per_partition * self.config.v_head_dim)

            # ==================================
            # apply output projection. [b, s, h]
            # ==================================
            output = self.linear_proj(core_attn_out)
        else:
            core_attn_out = self.core_attention(
                query, key, value, slot_mapping, block_tables, batch_valid_length,
                context_lens_tensor, q_seq_lens, actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen, attn_mask=attention_mask,
                key_cache=key_cache, value_cache=value_cache)

            core_attn_out = core_attn_out.reshape(bs, seq_len, self.num_attention_heads_per_partition, -1)
            core_attn_out = mint.matmul(mint.transpose(core_attn_out, -3, -2),
                                        mint.transpose(out_absorb, -2, -1))
            core_attn_out = mint.transpose(core_attn_out, -3, -2)
            core_attn_out = core_attn_out.reshape(bs, seq_len,
                                                  self.num_attention_heads_per_partition * self.config.v_head_dim)

            # ==================================
            # apply output projection. [b, s, h]
            # ==================================
            output = self.linear_proj(core_attn_out)

        return output


class MLASelfAttention(MultiLatentAttention):
    """MLA Self-attention layer class

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
            self,
            config: MLATransformerConfig,
            submodules: MLASelfAttentionSubmodules,
            layer_number: int,
            attn_mask_type=None,
            cp_comm_type: str = None
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type
        )

        if self.config.q_lora_rank is None:
            self.linear_q_proj = build_module(
                submodules.linear_q_proj,
                self.config.hidden_size,
                self.config.num_attention_heads * self.q_head_dim,
                config=self.config,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                transpose_b=True,
                compute_dtype=self.config.compute_dtype
            )

            self.linear_kv_down_proj = build_module(
                submodules.linear_kv_down_proj,
                self.config.hidden_size,
                self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim,
                config=self.config,
                bias=False,
                skip_bias_add=False,
                gather_output=False,
                transpose_b=True,
                compute_dtype=self.config.compute_dtype,
                is_expert=False
            )

        else:
            self.linear_qkv_down_proj = build_module(
                submodules.linear_qkv_down_proj,
                self.config.hidden_size,
                self.config.q_lora_rank + self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim,
                config=self.config,
                bias=False,
                transpose_b=True,
                compute_dtype=self.config.compute_dtype
            )

            self.linear_q_up_proj = build_module(
                submodules.linear_q_up_proj,
                self.config.q_lora_rank,
                self.config.num_attention_heads * self.q_head_dim,
                config=self.config,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                transpose_b=True,
                compute_dtype=self.config.compute_dtype,
                is_expert=False
            )

        self.linear_kv_up_proj = build_module(
            submodules.linear_kv_up_proj,
            self.config.kv_lora_rank,
            self.num_attention_heads * (self.config.qk_head_dim + self.config.v_head_dim),
            config=self.config,
            bias=False,
            transpose_b=True,
            compute_dtype=self.config.compute_dtype,
            is_expert=False
        )

        if self.config.q_lora_rank is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.config.q_lora_rank,
                config=self.config,
                eps=self.config.layernorm_epsilon
            )

        self.kv_layernorm = build_module(
            submodules.kv_layernorm,
            hidden_size=self.config.kv_lora_rank,
            config=self.config,
            eps=self.config.layernorm_epsilon
        )

    def get_query_key_value_tensors(
            self,
            hidden_states,
            batch_valid_length=None,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            slot_mapping=None,
            key_cache=None
    ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """

        bs, seq_len, _ = hidden_states.shape

        # =========================================
        # QKV down projection and layernorm
        # =========================================
        if self.config.q_lora_rank is not None:
            qkv = self.linear_qkv_down_proj(hidden_states)
            q_compressed, kv_compressed, k_pos_emb = mint.split(qkv,
                                                                [self.config.q_lora_rank, self.config.kv_lora_rank,
                                                                 self.config.qk_pos_emb_head_dim],
                                                                dim=-1)

            if q_compressed.shape[-1] != self.config.q_lora_rank:
                q_compressed = self.gather_from_mp_region(q_compressed)

            # q_compressed: [num_tokens, q_lora_rank]
            q_compressed = self.q_layernorm(q_compressed)
            # q: [num_tokens, n * (qk_head_dim + qk_pos_emb_head_dim)]
            q = self.linear_q_up_proj(q_compressed)

        else:
            q_compressed = hidden_states
            # if linear_kv_down_proj is ColumnParallelLinear:
            #     kv_combined: [s, b, (kv_lora_rank + qk_pos_emb_head_dim) / TP]
            # elif linear_kv_down_proj is ReplicatedLinear:
            #     kv_combined: [s / TP, b, (kv_lora_rank + qk_pos_emb_head_dim)]
            kv_combined = self.linear_kv_down_proj(hidden_states)
            if kv_combined.shape[-1] != self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim:
                # kv_combined: [s, b, (kv_lora_rank + qk_pos_emb_head_dim)]
                kv_combined = self.gather_from_mp_region(kv_combined)
                # kv_compressed: [s, b, kv_lora_rank], k_pos_emb: [s, b, qk_pos_emb_head_dim]
                kv_compressed, k_pos_emb = mint.split(
                    kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
                )
            else:
                # kv_compressed: [s / TP, b, kv_lora_rank], k_pos_emb: [s / TP, b, qk_pos_emb_head_dim]
                kv_compressed, k_pos_emb = mint.split(
                    kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
                )
            # q: [num_tokens, n * (qk_head_dim + qk_pos_emb_head_dim)]
            q = self.linear_q_proj(q_compressed)

        # q: [num_tokens, n, q_head_dim]
        q = q.reshape(*q.shape[:-1], self.num_attention_heads_per_partition, self.q_head_dim)
        # q_no_pe: [num_tokens, n, qk_head_dim], q_pos_emb: [num_tokens, n, qk_pos_emb_head_dim]
        q_no_pe, q_pos_emb = mint.split(q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1)
        kv_compressed = self.kv_layernorm(kv_compressed)

        # k_pos_emb: [num_tokens, qk_pos_emb_head_dim] -> [num_tokens, 1, qk_pos_emb_head_dim]
        k_pos_emb = mint.unsqueeze(k_pos_emb, -2)
        if rotary_pos_cos is not None and rotary_pos_sin is not None:
            q_pos_emb, k_pos_emb = self.rotary_pos_emb(q_pos_emb, k_pos_emb,
                                                       rotary_pos_cos, rotary_pos_sin,
                                                       batch_valid_length)
        q_pos_emb = q_pos_emb.reshape(bs, seq_len,
                                      self.num_attention_heads_per_partition, self.config.qk_pos_emb_head_dim)
        k_pos_emb = k_pos_emb.reshape(bs, seq_len, 1, self.config.qk_pos_emb_head_dim)

        key_states_cache = mint.cat((kv_compressed,
                                     k_pos_emb.reshape(bs, seq_len, self.config.qk_pos_emb_head_dim)), dim=-1)
        key_out = self.core_attention.reshape_and_cache(key_states_cache, None, key_cache, None, slot_mapping)
        q_no_pe = self.depend(q_no_pe, key_out)

        if self.is_prefill:
            # kv: [num_tokens, n * (qk_head_dim + v_head_dim)]
            kv = self.linear_kv_up_proj(kv_compressed)

            # k_no_pe: [num_tokens, n, qk_head_dim * self.kv_num_heads_per_partition],
            # value: [num_tokens, n, v_head_dim * self.kv_num_heads_per_partition]
            k_no_pe, value = mint.split(kv, [self.config.qk_head_dim * self.kv_num_heads_per_partition,
                                             self.config.v_head_dim * self.kv_num_heads_per_partition], dim=-1)
            k_no_pe = k_no_pe.reshape(bs, seq_len, self.num_attention_heads_per_partition, self.config.qk_head_dim)
            value_states = value.reshape(bs, seq_len, self.num_attention_heads_per_partition, self.config.v_head_dim)
            # query_states: [num_tokens, n, (qk_head_dim + v_head_dim)]
            query_states = mint.cat((q_no_pe, q_pos_emb), dim=-1)

            # k_pos_emb: [num_tokens, n, (qk_head_dim + v_head_dim)]
            if k_pos_emb.ndim == 4:
                k_pos_emb = mint.tile(k_pos_emb, (1, 1, self.kv_num_heads_per_partition, 1))
            elif k_pos_emb.ndim == 3:
                k_pos_emb = mint.tile(k_pos_emb, (1, self.kv_num_heads_per_partition, 1))
            else:
                raise RuntimeError(
                    f"k_pos_emb's dim must be 3 or 4, but got {k_pos_emb.ndim}."
                )

            key_states = mint.cat((k_no_pe, k_pos_emb), dim=-1)
            value_states = mint.cat((value_states, k_pos_emb), dim=-1)

            query = query_states.reshape(bs, seq_len, -1)
            key = key_states.reshape(bs, seq_len, -1)
            value = value_states.reshape(bs, seq_len, -1)

            return query, key, value, None

        q_absorb, out_absorb = mint.split(self.linear_kv_up_proj.weight,
                                          [self.num_attention_heads_per_partition * self.config.qk_head_dim,
                                           self.num_attention_heads_per_partition * self.config.v_head_dim], -2)
        q_absorb = q_absorb.reshape(self.num_attention_heads_per_partition,
                                    self.config.qk_head_dim, self.config.kv_lora_rank)
        out_absorb = out_absorb.reshape(self.num_attention_heads_per_partition,
                                        self.config.v_head_dim, self.config.kv_lora_rank)

        q_no_pe = mint.transpose(mint.matmul(mint.transpose(q_no_pe, -3, -2), q_absorb), -3, -2)
        query_states = mint.cat((q_no_pe, q_pos_emb), dim=-1)
        query = query_states.reshape(bs, seq_len, -1)
        key = key_states_cache
        return query, key, key, out_absorb
