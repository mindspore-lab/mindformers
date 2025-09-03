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
    'MLASelfAttention',
    'FusedMLASelfAttention'
]

import math
from dataclasses import dataclass
from typing import Union, Optional
import numpy as np

from mindspore import mint, ops, Tensor, dtype, Parameter
from mindspore.ops import operations as P
from mindspore.common.initializer import Zero

from mindformers.parallel_core.inference.quantization import QuantizationConfig
from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.inference.utils import divide, get_tp_world_size
from mindformers.parallel_core.transformer_config import MLATransformerConfig
from mindformers.parallel_core.inference.tensor_parallel.mappings import gather_from_model_parallel_region
from mindformers.parallel_core.inference.transformer.attention import Attention
from mindformers.parallel_core.inference.base_models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from mindformers.parallel_core.inference.parallel_state import get_tensor_model_parallel_world_size
from mindformers.parallel_core.inference.base_models.common.embeddings.yarn_rotary_pos_embedding import (
    YaRNScalingRotaryEmbedding,
    _yarn_get_mscale
)
from mindformers.parallel_core.process_group_config import ModelCommProcessGroups, default_model_comm_pgs


@dataclass
class MLASelfAttentionSubmodules:
    """Submodules for the MLA self-attention layer."""

    input_layernorm: Union[ModuleSpec, type] = None
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
            cp_comm_type: str = None,
            delay_allreduce: bool = False,
            model_comm_pgs: Optional[ModelCommProcessGroups] = default_model_comm_pgs,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            cp_comm_type=cp_comm_type,
            model_comm_pgs=model_comm_pgs,
        )
        if attn_mask_type:
            raise NotImplementedError("For MultiLatentAttention, `attn_mask_type` is not supported for now.")
        if attention_type:
            raise NotImplementedError("For MultiLatentAttention, `attention_type` is not supported for now.")
        if cp_comm_type is not None:
            raise NotImplementedError("For MultiLatentAttention, `cp_comm_type` is not supported for now.")

        self.config = config
        self.input_layernorm = build_module(
            submodules.input_layernorm,
            config=config,
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon
        )
        self.use_flash_attention = self.config.use_flash_attention
        self.query_projection_size = self.config.v_head_dim * self.config.num_attention_heads
        self.q_head_dim = self.config.qk_head_dim + self.config.qk_pos_emb_head_dim
        # Overwrite the base class kv shape to support MLA inference
        self.key_hidden_size = self.q_head_dim
        self.val_hidden_size = self.config.v_head_dim
        self.num_attention_heads = self.config.num_attention_heads
        self.compute_dtype = self.config.compute_dtype
        mscale = _yarn_get_mscale(self.config.rotary_scaling_factor, self.config.mscale)
        self.softmax_scale = mscale * mscale / math.sqrt(self.q_head_dim)

        if self.config.rope_type == "rope":
            self.rotary_pos_emb = RotaryEmbedding(
                self.config.qk_pos_emb_head_dim,
                rotary_percent=self.config.rotary_percent,
                rotary_base=self.config.rotary_base,
                rotary_cos_format="rotate_half")
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
                rotary_cos_format="rotate_half")
        else:
            raise ValueError(
                f"Unsupported RoPE type: {self.config.rope_type}, supported types are  'rope' and 'yarn'"
            )

        self.tp_group_size = get_tp_world_size()
        self.num_attention_heads_per_partition = divide(self.num_attention_heads, self.tp_group_size)
        self.hidden_size_per_attention_head = getattr(config, 'kv_channels', divide(
            self.config.hidden_size, self.num_attention_heads
        ))
        self.kv_num_heads_per_partition = self.num_attention_heads_per_partition

        if self.config.use_flash_attention:
            self.core_attention = build_module(
                submodules.core_attention,
                head_num=self.num_attention_heads_per_partition,
                head_dim=self.q_head_dim,
                kv_head_num=self.kv_num_heads_per_partition,
                scale_value=self.softmax_scale,
                next_tokens=0,
                pa_kv_head_num=1,
                pa_mla_v_dim=512)
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
            delay_allreduce=delay_allreduce,
            is_expert=False,
            transpose_b=True,
            compute_dtype=self.compute_dtype,
            tp_group=self.tp,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_proj"
        )

        self.cast = P.Cast()
        self.depend = P.Depend()
        self.dim_slice_3d = P.Slice()

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

        query, _, key, _, value, out_absorb = self.get_query_key_value_tensors(
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
        if self.is_prefill:
            core_attn_out = self.core_attention(
                query, key, value, slot_mapping, block_tables, batch_valid_length,
                context_lens_tensor, q_seq_lens, actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen, attn_mask=attention_mask,
                key_cache=key_cache, value_cache=value_cache)

            core_attn_out = core_attn_out.reshape(-1, self.num_attention_heads_per_partition, self.q_head_dim)
            core_attn_out = self.dim_slice_3d(core_attn_out, (0, 0, 0),
                                              (-1, self.num_attention_heads_per_partition, self.config.v_head_dim))
        else:
            core_attn_out = self.core_attention(
                query, key, value, slot_mapping, block_tables, batch_valid_length,
                context_lens_tensor, q_seq_lens, actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen, attn_mask=attention_mask,
                key_cache=key_cache, value_cache=value_cache)

            core_attn_out = core_attn_out.reshape(-1, self.num_attention_heads_per_partition, self.config.kv_lora_rank)
            core_attn_out = mint.matmul(mint.transpose(core_attn_out, -3, -2),
                                        mint.transpose(out_absorb, -2, -1))
            core_attn_out = mint.transpose(core_attn_out, -3, -2)

        core_attn_out = core_attn_out.reshape(-1, self.num_attention_heads_per_partition * self.config.v_head_dim)
        # ==================================
        # apply output projection. [t, h]
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
            cp_comm_type: str = None,
            delay_allreduce: bool = False,
            model_comm_pgs: Optional[ModelCommProcessGroups] = default_model_comm_pgs,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
        ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            delay_allreduce=delay_allreduce,
            model_comm_pgs=model_comm_pgs,
            quant_config=quant_config,
            prefix=prefix
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
                compute_dtype=self.config.compute_dtype,
                tp_group=self.tp,
                quant_config=quant_config,
                prefix=f"{prefix}.linear_q_proj"
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
                is_expert=False,
                quant_config=quant_config,
                prefix=f"{prefix}.linear_kv_down_proj"
            )

        else:
            self.linear_qkv_down_proj = build_module(
                submodules.linear_qkv_down_proj,
                self.config.hidden_size,
                self.config.q_lora_rank + self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim,
                config=self.config,
                bias=False,
                transpose_b=True,
                compute_dtype=self.config.compute_dtype,
                quant_config=quant_config,
                prefix=f"{prefix}.linear_qkv_down_proj"
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
                is_expert=False,
                tp_group=self.tp,
                quant_config=quant_config,
                prefix=f"{prefix}.linear_q_up_proj"
            )

        self.linear_kv_up_proj = build_module(
            submodules.linear_kv_up_proj,
            self.config.kv_lora_rank,
            self.num_attention_heads * (self.config.qk_head_dim + self.config.v_head_dim),
            config=self.config,
            bias=False,
            transpose_b=True,
            compute_dtype=self.config.compute_dtype,
            is_expert=False,
            tp_group=self.tp,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_kv_up_proj"
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

    def compute_kv(
            self,
            hidden_states,
            batch_valid_length=None,
            rotary_pos_cos=None,
            rotary_pos_sin=None
        ):
        """
        compute q_pos_emb, kv_compressed, k_pos_emb
        """
        hidden_states = self.input_layernorm(hidden_states)
        if self.config.q_lora_rank is not None:
            qkv = self.linear_qkv_down_proj(hidden_states)
            kv_compressed, k_pos_emb, q_compressed = mint.split(qkv,
                                                                [self.config.kv_lora_rank,
                                                                 self.config.qk_pos_emb_head_dim,
                                                                 self.config.q_lora_rank],
                                                                dim=-1)

            if q_compressed.shape[-1] != self.config.q_lora_rank:
                q_compressed = gather_from_model_parallel_region(q_compressed, self.tp)

            # q_compressed: [num_tokens, q_lora_rank]
            q_compressed = self.q_layernorm(q_compressed)
            # q: [num_tokens, n * (qk_head_dim + qk_pos_emb_head_dim)]
            q = self.linear_q_up_proj(q_compressed)

        else:
            q_compressed = hidden_states
            kv_combined = self.linear_kv_down_proj(hidden_states)
            if kv_combined.shape[-1] != self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim:
                # kv_combined: [s, b, (kv_lora_rank + qk_pos_emb_head_dim)]
                kv_combined = gather_from_model_parallel_region(q_compressed, self.tp)
            kv_compressed, k_pos_emb = mint.split(
                kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
            )
            # q: [num_tokens, n * (qk_head_dim + qk_pos_emb_head_dim)]
            q = self.linear_q_proj(q_compressed)

        # q: [num_tokens, n, q_head_dim]
        q = q.reshape(*q.shape[:-1], self.num_attention_heads_per_partition, self.q_head_dim)
        # q_no_pe: [num_tokens, n, qk_head_dim], q_pos_emb: [num_tokens, n, qk_pos_emb_head_dim]
        q_no_pe, q_pos_emb = mint.split(q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1)
        # kv_compressed: [num_tokens, kv_lora_rank]
        kv_compressed = self.kv_layernorm(kv_compressed)

        # q_pos_emb: [num_tokens, n, qk_pos_emb_head_dim] -> [num_tokens, n * qk_pos_emb_head_dim]
        q_pos_emb = q_pos_emb.reshape(-1, self.num_attention_heads_per_partition * self.config.qk_pos_emb_head_dim)
        if rotary_pos_cos is not None and rotary_pos_sin is not None:
            q_pos_emb = q_pos_emb.contiguous()
            k_pos_emb = k_pos_emb.contiguous()
            q_pos_emb, k_pos_emb = self.rotary_pos_emb(q_pos_emb, k_pos_emb,
                                                       rotary_pos_cos, rotary_pos_sin,
                                                       batch_valid_length)
        # q_pos_emb: [num_tokens, n * qk_pos_emb_head_dim] -> [num_tokens, n, qk_pos_emb_head_dim]
        q_pos_emb = q_pos_emb.reshape(-1, self.num_attention_heads_per_partition, self.config.qk_pos_emb_head_dim)
        return kv_compressed, k_pos_emb, q_no_pe, q_pos_emb

    def split_kv(
            self,
            kv_compressed,
            k_pos_emb
        ):
        """
        split k_no_pe, k_pos_emb, value_states from compressed kv and k position embedding
        """
        # kv: [num_tokens, n * (qk_head_dim + v_head_dim)]
        kv = self.linear_kv_up_proj(kv_compressed)

        # k_no_pe: [num_tokens, qk_head_dim * self.kv_num_heads_per_partition],
        # value: [num_tokens, v_head_dim * self.kv_num_heads_per_partition]
        k_no_pe, value = mint.split(kv, [self.config.qk_head_dim * self.kv_num_heads_per_partition,
                                         self.config.v_head_dim * self.kv_num_heads_per_partition], dim=-1)
        k_no_pe = k_no_pe.reshape(-1, self.kv_num_heads_per_partition, self.config.qk_head_dim)

        # value_states: [num_tokens, n, v_head_dim]
        value_states = value.reshape(-1, self.kv_num_heads_per_partition, self.config.v_head_dim)
        # k_pos_emb: [num_tokens, qk_pos_emb_head_dim] -> [num_tokens, 1, qk_pos_emb_head_dim]
        k_pos_emb = k_pos_emb.reshape(-1, 1, self.config.qk_pos_emb_head_dim)
        # k_pos_emb: [num_tokens, n, (qk_head_dim + v_head_dim)]
        k_pos_emb = mint.tile(k_pos_emb, (1, self.kv_num_heads_per_partition, 1))
        return k_no_pe, k_pos_emb, value_states

    def get_query_key_value_tensors(
            self,
            hidden_states,
            batch_valid_length=None,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            slot_mapping=None,
            key_cache=None,
            value_cache=None
        ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # =========================================
        # QKV down projection and layernorm
        # =========================================
        kv_compressed, k_pos_emb, q_no_pe, q_pos_emb = \
            self.compute_kv(hidden_states, batch_valid_length, rotary_pos_cos, rotary_pos_sin)
        key_states_cache = mint.cat((kv_compressed, k_pos_emb), dim=-1)
        key_out = self.core_attention.reshape_and_cache(key_states_cache, None, key_cache, None, slot_mapping)
        q_no_pe = self.depend(q_no_pe, key_out)

        if self.is_prefill:
            k_no_pe, k_pos_emb, value_states = self.split_kv(kv_compressed, k_pos_emb)
            # query_states: [num_tokens, n, (qk_head_dim + v_head_dim)]
            query_states = mint.cat((q_no_pe, q_pos_emb), dim=-1)
            key_states = mint.cat((k_no_pe, k_pos_emb), dim=-1)
            value_states = mint.cat((value_states, k_pos_emb), dim=-1)

            query = query_states.reshape(-1, self.num_attention_heads_per_partition * self.q_head_dim)
            key = key_states.reshape(-1, self.num_attention_heads_per_partition * self.q_head_dim)
            value = value_states.reshape(-1, self.num_attention_heads_per_partition * self.q_head_dim)

            return query, None, key, None, value, None

        q_absorb, out_absorb = mint.split(self.linear_kv_up_proj.weight,
                                          [self.num_attention_heads_per_partition * self.config.qk_head_dim,
                                           self.num_attention_heads_per_partition * self.config.v_head_dim], -2)
        q_absorb = q_absorb.reshape(self.num_attention_heads_per_partition,
                                    self.config.qk_head_dim, self.config.kv_lora_rank)
        out_absorb = out_absorb.reshape(self.num_attention_heads_per_partition,
                                        self.config.v_head_dim, self.config.kv_lora_rank)

        q_no_pe = mint.transpose(mint.matmul(mint.transpose(q_no_pe, -3, -2), q_absorb), -3, -2)
        query_states = mint.cat((q_no_pe, q_pos_emb), dim=-1)
        query = query_states.reshape(-1,
                                     self.num_attention_heads_per_partition *
                                     (self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim))

        return query, None, None, None, None, out_absorb

class FusedMLASelfAttention(MLASelfAttention):
    """MLA Self-attention layer class use fused op

    Only used in quantization inference
    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """
    def __init__(
            self,
            config: MLATransformerConfig,
            submodules: MLASelfAttentionSubmodules,
            layer_number: int,
            attn_mask_type=None,
            cp_comm_type: str = None,
            delay_allreduce: bool = False,
            model_comm_pgs: Optional[ModelCommProcessGroups] = default_model_comm_pgs,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
        ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            delay_allreduce=delay_allreduce,
            model_comm_pgs=model_comm_pgs,
            quant_config=quant_config,
            prefix=prefix
        )
        self.mla_preprocess = ops.auto_generate.MlaPreprocess()
        self.is_modelslim = quant_config.is_modelslim
        self.fa3_quant = quant_config.fa3_quant
        self.fa3_quant_layer = quant_config.fa3_quant_layer
        self.is_fa3_quant_layer = (layer_number - 1) in self.fa3_quant_layer # layer_number start from 1
        self.input_layernorm_weight = None
        self.qkv_down_proj_input_scale = None
        self.q_layernorm_weight = None
        self.q_up_proj_input_scale = None
        self.qkv_down_beta = None
        self.q_up_beta = None
        self.qkv_down_proj_input_offset = None
        self.q_up_proj_input_offset = None
        self.use_ringmla = get_tensor_model_parallel_world_size() < 16
        try:
            # pylint: disable=W0611
            import ms_custom_op
        except ModuleNotFoundError:
            # environment need install ms_custom_ops package
            self.use_ringmla = False

        self.mla = ops.auto_generate.Mla()
        self.scale_value = 1 / math.sqrt(self.config.kv_lora_rank + self.config.qk_head_dim) \
                           if self.softmax_scale is None else self.softmax_scale
        self.ring_mla_mask = Tensor(np.triu(np.ones((512, 512), dtype=np.float16), 1), dtype.bfloat16)
        self.depend = P.Depend()
        self.quant = QuantV2()
        if self.is_fa3_quant_layer:
            self.qnope_scale = Parameter(Tensor(shape=(self.num_attention_heads_per_partition,), dtype=dtype.float32,
                                                init=Zero()),
                                         name="qnope_scale",
                                         requires_grad=False)
            self.quant_ctkv_scale = None
            self.ctkv_scale = Parameter(Tensor(shape=(1,), dtype=dtype.float32, init=Zero()), name="ctkv_scale",
                                        requires_grad=False)
            self.ctkv_offset = Tensor(shape=(1,), dtype=self.compute_dtype, init=Zero())
            set_weight_attrs(
                self.ctkv_scale,
                {
                    "weight_loader": self.weight_loader,
                },
            )
            set_weight_attrs(
                self.qnope_scale,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )

        else:
            self.ctkv_scale = Tensor([0], dtype=dtype.bfloat16)
            self.qnope_scale = Tensor([0], dtype=dtype.bfloat16)
        self.qk_descale = None
        self.pv_descale = None
        if not self.use_ringmla:
            # cache shape: [blockNum,blockSize,1,576]
            self.cache_mode = 0
        elif self.is_fa3_quant_layer:
            # fa3 quant layers
            # cache shape: [blockNum,blockSize,1,1,512]
            self.cache_mode = 2
        elif self.fa3_quant:
            # fa3 no quant layers, kvcache also need nz
            # cache shape: [blockNum, headNum*512/32,block_size, 32]
            self.cache_mode = 3
        else:
            # cache shape: [blockNum, headNum*512/16,block_size, 16]
            self.cache_mode = 1

    def process_weights_after_loading(self) -> None:
        """
        Process the weight after loading.
        This can be used for example, to transpose weights for computation.
        """
        if not self.is_modelslim:
            # Temporary, to be deleted upon completion of weight conversion.
            qkv_down_input_scale = self.linear_qkv_down_proj.input_scale.asnumpy().astype(np.float32)
            input_layernorm = self.input_layernorm.weight.asnumpy().astype(np.float32)
            input_layernorm_dtype = self.input_layernorm.weight.dtype
            input_layernorm = input_layernorm / qkv_down_input_scale
            self.input_layernorm_weight = Tensor(input_layernorm, dtype=input_layernorm_dtype)
            self.qkv_down_proj_input_scale = Tensor([1], dtype=dtype.bfloat16)

            q_up_input_scale = self.linear_q_up_proj.input_scale.asnumpy().astype(np.float32)
            q_layernorm = self.q_layernorm.weight.asnumpy().astype(np.float32)
            q_layernorm_dtype = self.q_layernorm.weight.dtype
            q_layernorm = q_layernorm / q_up_input_scale
            self.q_layernorm_weight = Tensor(q_layernorm, dtype=q_layernorm_dtype)
            self.q_up_proj_input_scale = Tensor([1], dtype=dtype.bfloat16)
            self.qkv_down_beta = Tensor(shape=(self.config.hidden_size,), dtype=dtype.bfloat16, init=Zero())
            self.q_up_beta = Tensor(shape=(self.config.q_lora_rank,), dtype=dtype.bfloat16, init=Zero())
        else:
            self.input_layernorm_weight = self.input_layernorm.weight
            self.qkv_down_proj_input_scale = self.linear_qkv_down_proj.input_scale
            self.q_layernorm_weight = self.q_layernorm.weight
            self.q_up_proj_input_scale = self.linear_q_up_proj.input_scale
            self.qkv_down_beta = self.linear_qkv_down_proj.beta
            self.q_up_beta = self.linear_q_up_proj.beta
        self.qkv_down_proj_input_offset = Tensor(self.linear_qkv_down_proj.input_offset.asnumpy(),
                                                 dtype=dtype.int8)
        self.q_up_proj_input_offset = Tensor(self.linear_q_up_proj.input_offset.asnumpy(),
                                             dtype=dtype.int8)
        if not self.is_fa3_quant_layer:
            return
        qnope_scale = self.qnope_scale.asnumpy()
        ctkv_scale = self.ctkv_scale.asnumpy()
        qk_descale = ctkv_scale.astype(np.float32) * qnope_scale.astype(np.float32)
        pv_descale = np.repeat(ctkv_scale, qnope_scale.shape[0])
        self.quant_ctkv_scale = Tensor(1.0/ctkv_scale, dtype=self.compute_dtype)
        self.ctkv_scale = Parameter(Tensor(ctkv_scale, dtype=self.compute_dtype), name=self.ctkv_scale.name)
        self.qnope_scale = Parameter(Tensor(1.0/qnope_scale, dtype=self.compute_dtype), name=self.qnope_scale.name)
        self.qk_descale = ms.from_numpy(qk_descale)
        self.pv_descale = ms.from_numpy(pv_descale)

    def get_query_key_value_tensors(
            self,
            hidden_states,
            batch_valid_length=None,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            slot_mapping=None,
            key_cache=None,
            value_cache=None
    ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        if self.is_prefill:
            kv_compressed, k_pos_emb, q_no_pe, q_pos_emb = \
                self.compute_kv(hidden_states, batch_valid_length, rotary_pos_cos, rotary_pos_sin)
            if self.use_ringmla:
                if self.is_fa3_quant_layer:
                    quant_kv_compressed = self.quant(kv_compressed, self.quant_ctkv_scale, self.ctkv_offset, False,
                                                     "ROUND", dtype.int8)
                    kv_out = ms_custom_ops.reshape_and_cache(quant_kv_compressed, k_pos_emb, key_cache, value_cache,
                                                             slot_mapping, 1)
                else:
                    kv_out = ms_custom_ops.reshape_and_cache(kv_compressed, k_pos_emb, key_cache, value_cache,
                                                             slot_mapping, 1)
                q_no_pe = self.depend(q_no_pe, kv_out)
            else:
                key_states_cache = mint.cat((kv_compressed, k_pos_emb), dim=-1)
                key_out = self.core_attention.reshape_and_cache(key_states_cache, key_cache=key_cache,
                                                                slot_mapping=slot_mapping)
                q_no_pe = self.depend(q_no_pe, key_out)

            k_no_pe, k_pos_emb, value_states = self.split_kv(kv_compressed, k_pos_emb)
            if self.use_ringmla:
                return q_no_pe, q_pos_emb, k_no_pe, k_pos_emb, value_states, None
            # query_states: [num_tokens, n, (qk_head_dim + v_head_dim)]
            query_states = mint.cat((q_no_pe, q_pos_emb), dim=-1)
            key_states = mint.cat((k_no_pe, k_pos_emb), dim=-1)
            value_states = mint.cat((value_states, k_pos_emb), dim=-1)

            query = query_states.reshape(-1, self.num_attention_heads_per_partition * self.q_head_dim)
            key = key_states.reshape(-1, self.num_attention_heads_per_partition * self.q_head_dim)
            value = value_states.reshape(-1, self.num_attention_heads_per_partition * self.q_head_dim)

            return query, None, key, None, value, None

        # only decode use fused op
        q_absorb, out_absorb = mint.split(self.linear_kv_up_proj.weight,
                                          [self.num_attention_heads_per_partition * self.config.qk_head_dim,
                                           self.num_attention_heads_per_partition * self.config.v_head_dim], -2)
        q_absorb = q_absorb.reshape(self.num_attention_heads_per_partition,
                                    self.config.qk_head_dim, self.config.kv_lora_rank)
        out_absorb = out_absorb.reshape(self.num_attention_heads_per_partition,
                                        self.config.v_head_dim, self.config.kv_lora_rank)

        states = self.mla_preprocess(
            hidden_states,
            self.input_layernorm_weight,
            self.qkv_down_beta,
            self.qkv_down_proj_input_scale,
            self.qkv_down_proj_input_offset,
            self.linear_qkv_down_proj.weight,
            self.linear_qkv_down_proj.quant_bias,
            self.q_layernorm_weight,
            self.q_up_beta,
            self.q_up_proj_input_scale,
            self.q_up_proj_input_offset,
            self.kv_layernorm.weight,
            rotary_pos_sin,
            rotary_pos_cos,
            rotary_pos_sin,
            rotary_pos_cos,
            key_cache,
            slot_mapping,
            self.linear_q_up_proj.weight,
            self.linear_q_up_proj.quant_bias,
            q_absorb,
            self.linear_qkv_down_proj.deq_scale,
            self.linear_q_up_proj.deq_scale,
            self.ctkv_scale,
            self.qnope_scale,
            key_cache if not self.use_ringmla else value_cache,
            self.cache_mode)
        if self.use_ringmla:
            return states[0], states[2], None, None, None, out_absorb
        query_states = states[0]
        query = query_states.reshape(-1,
                                     self.num_attention_heads_per_partition *
                                     (self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim))

        return query, None, None, None, None, out_absorb

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
        if not self.use_ringmla:
            return super().construct(
                hidden_states, attention_mask, key_value_states, rotary_pos_cos, rotary_pos_sin,
                position_ids, batch_valid_length, block_tables, slot_mapping, q_seq_lens,
                actual_seq_qlen, actual_seq_kvlen, context_lens_tensor, key_cache, value_cache
            )
        q_no_pe, q_pos_emb, k_no_pe, k_pos_emb, value, out_absorb = self.get_query_key_value_tensors(
            hidden_states, batch_valid_length, rotary_pos_cos,
            rotary_pos_sin, slot_mapping, key_cache, value_cache
        )
        # ==================================
        # core attention computation
        # ==================================
        if self.is_prefill:
            q_seq_len_cpu = ops.move_to(q_seq_lens, "CPU")
            o_prev = mint.zeros((hidden_states.shape[0], self.num_attention_heads_per_partition,
                                 self.config.v_head_dim), dtype=dtype.bfloat16)
            lse_prev = mint.zeros((self.num_attention_heads_per_partition, hidden_states.shape[0]),
                                  dtype=dtype.float32)
            #self.ring_mla_mask = self.tile_kv(self.ring_mla_mask, (q_seq_lens.shape[0], 1, 1))
            core_attn_out, lse_out = ms_custom_ops.ring_mla(
                query=q_no_pe, query_rope=q_pos_emb, key=k_no_pe, key_rope=k_pos_emb, value=value,
                mask=self.ring_mla_mask, alibi_coeff=None, deq_scale_qk=None, deq_offset_qk=None,
                deq_scale_pv=None, deq_offset_pv=None, quant_p=None, log_n=None, o_prev=o_prev,
                lse_prev=lse_prev, q_seq_lens=q_seq_len_cpu, context_lens=q_seq_len_cpu,
                head_num=self.num_attention_heads_per_partition, scale_value=self.scale_value,
                kv_head_num=self.num_attention_heads_per_partition,
                mask_type=1, #"MASK_TYPE_TRIU",
                calc_type=1, #"CALC_TYPE_FISRT_RING",
            )
            # compute_prefill_context
            if self.is_chunked:
                k_cache = mint.zeros(mint.cat((context_lens_tensor.sum().reshape(1),
                                               Tensor([1, self.config.kv_lora_rank]))), dtype=dtype.bfloat16)
                r_cache = mint.zeros(mint.cat((context_lens_tensor.sum().reshape(1),
                                               Tensor([1, self.config.qk_pos_emb_head_dim]))), dtype=dtype.bfloat16)
                # fa3_quant kvcache format is nz (shape is three dims)
                # paged_cache_load + nz now has error in the network
                # by the reshape ops, Identity ops is inserted for nz to nd
                # to temporarily avoid paged_cache_load errors
                if self.fa3_quant:
                    num_blocks, block_size, _ = key_cache.shape
                    key_cache = key_cache.reshape(num_blocks, block_size, 1, self.config.kv_lora_rank)
                    value_cache = value_cache.reshape(num_blocks, block_size, 1, self.config.qk_pos_emb_head_dim)
                # for fa3_quant_layer, need to dequant the key_cache
                if self.is_fa3_quant_layer:
                    key_cache = self.cast(key_cache, dtype.bfloat16)
                    key_cache = key_cache / self.quant_ctkv_scale
                load_out = ms_custom_ops.paged_cache_load(key_cache, value_cache, block_tables,
                                                          context_lens_tensor, k_cache, r_cache)
                k_cache = self.depend(k_cache, load_out)

                k_no_pe, k_pos_emb, value = self.split_kv(k_cache, r_cache)

                context_lens_cpu = ops.move_to(context_lens_tensor, "CPU")
                core_attn_out, _ = ms_custom_ops.ring_mla(
                    query=q_no_pe, query_rope=q_pos_emb, key=k_no_pe, key_rope=k_pos_emb, value=value,
                    mask=None, alibi_coeff=None, deq_scale_qk=None, deq_offset_qk=None, deq_scale_pv=None,
                    deq_offset_pv=None, quant_p=None, log_n=None, o_prev=core_attn_out,
                    lse_prev=lse_out, q_seq_lens=q_seq_len_cpu, context_lens=context_lens_cpu,
                    head_num=self.num_attention_heads_per_partition, scale_value=self.scale_value,
                    kv_head_num=self.num_attention_heads_per_partition,
                    mask_type=0, #"MASK_TYPE_TRIU",
                    calc_type=0, #"CALC_TYPE_FISRT_RING",
                    )
        else:
            core_attn_out, _ = self.mla(q_no_pe, q_pos_emb, key_cache, value_cache, block_tables, None,
                                        self.qk_descale, self.pv_descale,
                                        q_seq_lens, batch_valid_length, self.num_attention_heads_per_partition,
                                        self.scale_value)

            core_attn_out = core_attn_out.reshape(-1, self.num_attention_heads_per_partition, self.config.kv_lora_rank)
            core_attn_out = mint.matmul(mint.transpose(core_attn_out, -3, -2),
                                        mint.transpose(out_absorb, -2, -1))
            core_attn_out = mint.transpose(core_attn_out, -3, -2)

        core_attn_out = core_attn_out.reshape(-1, self.num_attention_heads_per_partition * self.config.v_head_dim)
        # ==================================
        # apply output projection. [t, h]
        # ==================================
        output = self.linear_proj(core_attn_out)

        return output

    def weight_loader(self, param, loaded_weight):
        """
        Load and partition weights for FusedMLASelfAttention.

        This method handles the loading of weights that have been partitioned along the output dimension
        according to tensor parallelism. Each rank loads its corresponding shard of the weight matrix.

        Args:
            param: The parameter tensor to load weights into.
            loaded_weight: The full weight tensor loaded from checkpoint.

        """
        tp_rank = self.tp.rank
        shard_dim = getattr(param, "output_dim", None)
        shard_size = self.num_attention_heads_per_partition
        start_idx = tp_rank * shard_size
        if shard_dim is not None:
            loaded_weight = split_loaded_weight(loaded_weight, shard_dim, start_idx, shard_size)
        else:
            loaded_weight = loaded_weight[:]
        loaded_weight = loaded_weight.squeeze(-1)
        if param.shape == loaded_weight.shape:
            param.set_data(ms.from_numpy(loaded_weight))
        else:
            raise ValueError(
                f"'{param.name}.shape' should be equal to 'loaded_weight.shape',"
                f" but got the shape of param is {param.shape} and the shape of weight is{loaded_weight.shape}")
