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
"""Multi-head Latent Attention (MLA) mechanism with low-rank compression."""
import math
from dataclasses import dataclass
from typing import Union
import mindspore as ms
from mindspore import nn, Tensor
from mindspore.mint import unsqueeze
from mindspore.ops import cast
from mindspore.common.initializer import initializer
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.transformer_config import MLATransformerConfig
from mindformers.parallel_core.training_graph.base_models.common.embeddings.rope_utils import ApplyRotaryPosEmb


@dataclass
class MLASelfAttentionSubmodules:
    """Submodules for the MLA self-attention layer."""
    linear_qkv: Union[ModuleSpec, type] = None
    linear_qb: Union[ModuleSpec, type] = None
    linear_kvb: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None


@dataclass
class MLASelfAttentionSubmodulesMegatron:
    """Submodules for the MLA self-attention layer."""

    linear_q_proj: Union[ModuleSpec, type] = None
    linear_q_down_proj: Union[ModuleSpec, type] = None
    linear_q_up_proj: Union[ModuleSpec, type] = None
    linear_kv_down_proj: Union[ModuleSpec, type] = None
    linear_kv_up_proj: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    kv_layernorm: Union[ModuleSpec, type] = None


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class MultiLatentAttention(nn.Cell):
    """Multi-head Latent Attention (MLA) with KV compression and rotary position encoding."""
    def __init__(
            self,
            config: MLATransformerConfig,
            submodules: Union[MLASelfAttentionSubmodules, MLASelfAttentionSubmodulesMegatron],
            layer_number: int,
            attention_type: str,
            attn_mask_type: str = None,
            cp_comm_type: str = None,
    ) -> None:
        super().__init__()
        if attn_mask_type:
            raise NotImplementedError("For Attention, 'attn_mask_type' is not supported for now.")
        if cp_comm_type:
            raise NotImplementedError("cp_comm_type is not supported for now.")
        self.config = config
        self.layer_number = layer_number
        self.layer_index = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type

        # model structure config
        self.use_flash_attention = self.config.use_flash_attention
        self.use_ring_attention = self.config.use_ring_attention
        self.use_eod_attn_mask_compression = self.config.use_eod_attn_mask_compression
        self.use_attn_mask_compression = self.config.use_attn_mask_compression
        self.num_attention_heads = self.config.num_attention_heads
        self.query_projection_size = self.config.v_head_dim * self.config.num_attention_heads
        self.qk_head_dim = self.config.qk_head_dim
        self.qk_pos_emb_head_dim = self.config.qk_pos_emb_head_dim
        self.q_head_dim = self.qk_head_dim + self.qk_pos_emb_head_dim
        self.kv_lora_rank = self.config.kv_lora_rank
        self.v_head_dim = self.config.v_head_dim
        self.input_layout = self.config.input_layout
        self.compute_dtype = self.config.compute_dtype

        zero_pad_length = self.q_head_dim - self.v_head_dim
        if zero_pad_length == 0:
            self.use_zero_pad = False
        elif zero_pad_length < 0:
            raise ValueError("qk_head_dim + qk_pos_emb_head_dim should not less than v_head_dim")
        else:
            self.use_zero_pad = True
            self.pad_zeros = initializer('zeros', shape=(1, 1, self.num_attention_heads, zero_pad_length),
                                         dtype=self.compute_dtype)

        mscale = yarn_get_mscale(self.config.rotary_scaling_factor, self.config.mscale)
        self.softmax_scale = mscale * mscale / math.sqrt(self.q_head_dim)

        self.core_attention = build_module(
            submodules.core_attention,
            config=self.config,
            layer_number=self.layer_index
        )

        # Output.
        self.linear_proj = build_module(
            submodules.linear_proj,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
        )

        self.shape = ms.ops.auto_generate.Shape()
        self.reshape = ms.ops.auto_generate.Reshape()
        self.bs_transpose = ms.ops.auto_generate.Transpose()
        self.tile = ms.ops.auto_generate.Tile()
        self.value_concat = ms.ops.auto_generate.Concat(-1)
        self.value_tnd_concat = ms.ops.auto_generate.Concat(-1)
        self.dim_slice = ms.ops.auto_generate.StridedSlice()
        self.dim_tnd_slice = ms.ops.auto_generate.StridedSlice()

        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.sharding_propagation()
        elif _get_parallel_mode() in (ParallelMode.SEMI_AUTO_PARALLEL,):
            self.shard()

    def sharding_propagation(self):
        pass

    def shard(self):
        dp = self.config.data_parallel_size
        tp = self.config.tensor_model_parallel_size

        self.bs_transpose.shard(((dp, 1, tp),))
        self.value_concat.shard(((1, dp, tp, 1), (1, dp, tp, 1)))
        self.value_tnd_concat.shard(((dp, tp, 1), (dp, tp, 1)))
        self.dim_slice.shard(((1, dp, tp, 1),))
        self.dim_tnd_slice.shard(((dp, tp, 1),))
        self.tile.shard(((1, 1, 1, 1),))

    def construct(self, x: Tensor, attention_mask=None, rotary_pos_emb=None, rotary_pos_cos=None,
                  rotary_pos_sin=None, prefix_keys_values=None, pad_zeros=None, actual_seq_len=None):
        """Forward process."""
        if rotary_pos_cos:
            raise NotImplementedError("rotary_pos_cos is not supported for now.")
        if rotary_pos_sin:
            raise NotImplementedError("rotary_pos_sin is not supported for now.")
        if prefix_keys_values:
            raise NotImplementedError("prefix_keys_values is not supported for now.")
        ori_dtype = x.dtype
        seq_len, bs, _ = self.shape(x)

        query, key, value = self.get_query_key_value_tensors(x, rotary_pos_emb=rotary_pos_emb)

        query = cast(query, self.compute_dtype)
        key = cast(key, self.compute_dtype)
        value = cast(value, self.compute_dtype)
        if self.use_flash_attention:
            if self.use_zero_pad:
                if self.input_layout == "TND":
                    pad_zeros = self.tile(self.pad_zeros, (bs, seq_len, 1, 1))
                    pad_zeros = self.reshape(pad_zeros, (bs * seq_len, -1, pad_zeros.shape[-1]))
                    value = self.value_tnd_concat((value, pad_zeros))
                else:
                    pad_zeros = self.tile(self.pad_zeros, (seq_len, bs, 1, 1))
                    value = self.value_concat((value, pad_zeros))
            if self.use_eod_attn_mask_compression:
                context_layer = self.core_attention(
                    query, key, value, attention_mask,
                    actual_seq_qlen=actual_seq_len, actual_seq_kvlen=actual_seq_len
                )
                if self.use_zero_pad:
                    context_layer = self.dim_tnd_slice(context_layer, (0, 0, 0),
                                                       (seq_len * bs, self.num_attention_heads, self.v_head_dim),
                                                       (1, 1, 1))
                attn_out = self.reshape(context_layer, (bs, seq_len, -1))
                attn_out = self.bs_transpose(attn_out, (1, 0, 2))
            else:
                context_layer = self.core_attention(
                    query, key, value, attention_mask,
                )
                if self.use_zero_pad:
                    context_layer = self.reshape(context_layer, (seq_len, bs, self.num_attention_heads, -1))
                    context_layer = self.dim_slice(context_layer, (0, 0, 0, 0),
                                                   (seq_len, bs, self.num_attention_heads, self.v_head_dim),
                                                   (1, 1, 1, 1))
                attn_out = self.reshape(context_layer, (seq_len, bs, -1))
        else:
            attn_out = self.core_attention(query, key, value, attention_mask)

        output = self.linear_proj(attn_out)[0]  # dp, mp -> dp, 1 / dp * mp, 1
        output = cast(output, ori_dtype)
        return output


class MLASelfAttention(MultiLatentAttention):
    """MLA Self-attention layer class, the implementation follows the same structure as Mindspeed A2

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
            self,
            config: MLATransformerConfig,
            submodules: MLASelfAttentionSubmodules,
            layer_number: int,
            attn_mask_type: str = None,
            cp_comm_type: str = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            attention_type="self",
        )
        self.use_tnd = config.input_layout == "TND"
        self.split = ms.ops.auto_generate.SplitWithSize().add_prim_attr("skip_redistribution", True)
        self.tile_kv = ms.ops.auto_generate.Tile()
        self.pe_concat = ms.ops.auto_generate.Concat(axis=3)
        self.pe_tnd_concat = ms.ops.auto_generate.Concat(axis=3)
        self.apply_rotary_emb = ApplyRotaryPosEmb(config)
        self.reshape = ms.ops.auto_generate.Reshape()
        self.tnd_transpose = ms.ops.auto_generate.Transpose()

        if self.config.q_lora_rank is None:
            self.q_rank = self.config.num_attention_heads * self.q_head_dim
            self.q_layernorm = None
        else:
            self.q_rank = self.config.q_lora_rank
            if submodules.q_layernorm is not None:
                self.q_layernorm = build_module(
                    submodules.q_layernorm,
                    dim=self.config.q_lora_rank,
                    config=self.config,
                    eps=self.config.layernorm_epsilon,
                )
            else:
                self.q_layernorm = None

            self.linear_qb = build_module(
                submodules.linear_qb,
                self.config.q_lora_rank,
                self.config.num_attention_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                bias=self.config.add_bias_linear or self.config.add_qkv_bias,
                skip_bias_add=False,
            )

        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.config.hidden_size,
            self.q_rank + self.kv_lora_rank + self.qk_pos_emb_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
        )

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                dim=self.kv_lora_rank,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_layernorm = None

        self.linear_kvb = build_module(
            submodules.linear_kvb,
            self.kv_lora_rank,
            self.config.num_attention_heads * (self.q_head_dim - self.qk_pos_emb_head_dim + self.v_head_dim),
            config=self.config,
            init_method=self.config.init_method,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
        )

        self.linear_proj = build_module(
            submodules.linear_proj,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
        )

        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.sharding_propagation()
        elif _get_parallel_mode() in (ParallelMode.SEMI_AUTO_PARALLEL,):
            self.shard_self_attn()

    def sharding_propagation(self):
        dp = self.config.data_parallel_size
        tp = self.config.tensor_model_parallel_size

        self.tnd_transpose.shard(((1, dp, tp, 1),))
        self.tile_kv.shard(((1, dp, tp, 1),))

    def shard_self_attn(self):
        dp = self.config.data_parallel_size
        tp = self.config.tensor_model_parallel_size

        self.tile_kv.shard(((1, dp, tp, 1),))
        self.pe_concat.shard(((1, dp, tp, 1), (1, dp, tp, 1)))

    def get_query_key_value_tensors(self,
                                    hidden_states,
                                    rotary_pos_emb=None
                                    # position_ids is used to generate rotary_pos_emb in Megatron, While rotary_pos_emb
                                    # is input in MindFormers.
                                    # key_value_states in Megatron is only used for CrossAttention.
                                    # packed_seq_params in Megatron is replaced by
                                    # config.use_eod_attn_mask_compression and actual_seq_len in MindFormers.
                                    # inference_params in Megatron will be deprecated in the future.
                                    ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        query, key, value = self._get_query_key_value_tensors(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb
        )

        if self.use_tnd:
            query = self.sbnd2tnd(query)
            key = self.sbnd2tnd(key)
            value = self.sbnd2tnd(value)


        return query, key, value

    def sbnd2tnd(self, x):
        seq_len, bs, *_ = x.shape
        x = self.tnd_transpose(x, (1, 0, 2, 3))
        x = self.reshape(x, (bs * seq_len, self.num_attention_heads, -1))
        return x

    def _get_query_key_value_tensors(self,
                                     hidden_states,
                                     rotary_pos_emb=None):
        """get_query_key_value_tensors"""

        tmp = self.shape(hidden_states)
        seq_len, bs = tmp[0], tmp[1]

        qkv_combo = self.linear_qkv(hidden_states)[0]

        q_a, compressed_kv, k_pe = self.split(
            qkv_combo,
            [
                self.q_rank,
                self.kv_lora_rank,
                self.qk_pos_emb_head_dim,
            ],
            dim=-1,
        )

        if self.q_layernorm is not None:
            q_a = self.q_layernorm(q_a)
            q = self.linear_qb(q_a)[0]
            q = self.reshape(q, (seq_len, bs, self.num_attention_heads, -1))

            q_nope, q_pe = self.split(
                q, [self.qk_head_dim, self.qk_pos_emb_head_dim], dim=-1
            )

        else:
            q = self.reshape(q_a, (seq_len, bs, self.num_attention_heads, -1))
            q_nope, q_pe = self.split(
                q, [self.qk_head_dim, self.qk_pos_emb_head_dim], dim=-1
            )

        k_pe = self.reshape(k_pe, (seq_len, bs, 1, self.qk_pos_emb_head_dim))
        compressed_kv_norm = self.k_layernorm(compressed_kv)

        kv = self.linear_kvb(compressed_kv_norm)[0]
        kv = self.reshape(kv, (
            seq_len,
            bs,
            self.num_attention_heads,
            self.qk_head_dim + self.v_head_dim,
        ))

        k_nope, value = self.split(kv, [self.qk_head_dim, self.v_head_dim], dim=-1)

        if rotary_pos_emb is not None:
            q_pe = self.apply_rotary_emb(
                q_pe,
                rotary_pos_emb,
                rotary_interleaved=self.config.rotary_interleaved,
                multi_latent_attention=self.config.multi_latent_attention
            )
            k_pe = self.apply_rotary_emb(
                k_pe,
                rotary_pos_emb,
                rotary_interleaved=self.config.rotary_interleaved,
                multi_latent_attention=self.config.multi_latent_attention,
                input_is_parallel=True
            )

        query = self.pe_concat([q_nope, q_pe])
        k_pe = self.tile_kv(k_pe, (1, 1, self.num_attention_heads, 1))
        key = self.pe_concat([k_nope, k_pe])

        return query, key, value


class MLASelfAttentionMegatron(MultiLatentAttention):
    """MLA Self-attention layer class
    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
            self,
            config: MLATransformerConfig,
            submodules: MLASelfAttentionSubmodulesMegatron,
            layer_number: int,
            attn_mask_type: str = None,
            cp_comm_type: str = None,
    ) -> None:
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            attention_type="self",
        )

        self.use_tnd = config.input_layout == "TND"
        self.tnd_transpose = ms.ops.auto_generate.Transpose()
        self.split = ms.ops.auto_generate.SplitWithSize().add_prim_attr("skip_redistribution", True)
        self.tile_kv = ms.ops.auto_generate.Tile()
        self.pe_concat = ms.ops.auto_generate.Concat(axis=3)
        self.pe_tnd_concat = ms.ops.auto_generate.Concat(axis=3)
        self.apply_rotary_emb = ApplyRotaryPosEmb(config)
        self.reshape = ms.ops.auto_generate.Reshape()
        self.tnd_transpose = ms.ops.auto_generate.Transpose()

        if self.config.q_lora_rank is None:
            # Not projecting query
            self.linear_q_proj = build_module(
                submodules.linear_q_proj,
                self.config.hidden_size,
                self.config.num_attention_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
            )

        else:
            self.linear_q_down_proj = build_module(
                submodules.linear_q_down_proj,
                self.config.hidden_size,
                self.config.q_lora_rank,
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
            )

            self.linear_q_up_proj = build_module(
                submodules.linear_q_up_proj,
                self.config.q_lora_rank,
                self.config.num_attention_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
            )

        self.linear_kv_down_proj = build_module(
            submodules.linear_kv_down_proj,
            self.config.hidden_size,
            self.kv_lora_rank + self.qk_pos_emb_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
        )

        self.linear_kv_up_proj = build_module(
            submodules.linear_kv_up_proj,
            self.config.kv_lora_rank,
            self.config.num_attention_heads * (self.config.qk_head_dim + self.config.v_head_dim),
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
        )

        if self.config.q_lora_rank is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                dim=self.config.q_lora_rank,
                config=self.config,
                param_init_type=self.config.layernorm_compute_dtype,
                eps=self.config.layernorm_epsilon,
            )

        self.kv_layernorm = build_module(
            submodules.kv_layernorm,
            dim=self.config.kv_lora_rank,
            config=self.config,
            param_init_type=self.config.layernorm_compute_dtype,
            eps=self.config.layernorm_epsilon,
        )

        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.sharding_propagation()
        elif _get_parallel_mode() in (ParallelMode.SEMI_AUTO_PARALLEL,):
            self.shard_self_attn()

    def sharding_propagation_self_attn(self):
        dp = self.config.data_parallel_size
        tp = self.config.tensor_model_parallel_size

        self.tnd_transpose.shard(((1, dp, tp, 1),))

    def shard_self_attn(self):
        dp = self.config.data_parallel_size
        tp = self.config.tensor_model_parallel_size

        self.tile_kv.shard(((1, dp, tp, 1),))
        self.pe_concat.shard(((1, dp, tp, 1), (1, dp, tp, 1)))

    def sbnd2tnd(self, x):
        seq_len, bs, *_ = x.shape
        x = self.tnd_transpose(x, (1, 0, 2, 3))
        x = self.reshape(x, (bs * seq_len, self.num_attention_heads, -1))
        return x

    def qkv_up_proj_and_rope_apply(self, q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb):
        """
        Apply proj and rope on `query`, `key` and `value`.
        """
        if self.config.q_lora_rank is not None:
            q, _ = self.linear_q_up_proj(q_compressed)
        else:
            # hidden_states:[s, b, 2048], q: [s, b, n * 192]
            q, _ = self.linear_q_proj(q_compressed)

        q_len, bs, _ = self.shape(q)

        q = self.reshape(q, (q_len, bs, self.num_attention_heads, self.q_head_dim))
        kv, _ = self.linear_kv_up_proj(kv_compressed)

        kv = self.reshape(kv, (q_len, bs, self.num_attention_heads, self.config.qk_head_dim + self.config.v_head_dim))

        k_pos_emb = unsqueeze(k_pos_emb, 2)

        q_no_pe, q_pos_emb = self.split(
            q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1
        )

        k_no_pe, value = self.split(
            kv, [self.config.qk_head_dim, self.config.v_head_dim], dim=-1
        )

        if rotary_pos_emb is not None:
            q_pos_emb = self.apply_rotary_emb(
                q_pos_emb,
                rotary_pos_emb,
                rotary_interleaved=self.config.rotary_interleaved,
                multi_latent_attention=self.config.multi_latent_attention
            )
            k_pos_emb = self.apply_rotary_emb(
                k_pos_emb,
                rotary_pos_emb,
                rotary_interleaved=self.config.rotary_interleaved,
                multi_latent_attention=self.config.multi_latent_attention,
                input_is_parallel=True
            )

        query = self.pe_concat([q_no_pe, q_pos_emb])

        k_pos_emb = self.tile_kv(k_pos_emb, (1, 1, self.num_attention_heads, 1))
        key = self.pe_concat([k_no_pe, k_pos_emb])

        return query, key, value

    def get_query_key_value_tensors(self, hidden_states,
                                    rotary_pos_emb=None
                                    # position_ids is used to generate rotary_pos_emb in Megatron, While rotary_pos_emb
                                    # is input in MindFormers.
                                    # key_value_states in Megatron is only used for CrossAttention.
                                    # packed_seq_params in Megatron is replaced by
                                    # config.use_eod_attn_mask_compression and actual_seq_len in MindFormers.
                                    # inference_params in Megatron will be deprecated in the future.
                                    ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        query, key, value = self._get_query_key_value_tensors(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb
        )
        if self.use_tnd:
            query = self.sbnd2tnd(query)
            key = self.sbnd2tnd(key)
            value = self.sbnd2tnd(value)
        return query, key, value

    def _get_query_key_value_tensors(self, hidden_states,
                                     rotary_pos_emb=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """

        if hidden_states.ndim != 3:
            raise ValueError(f"hidden_states should be 3D, [s, b, n*h], got {hidden_states.ndim}D")

        # =========================================
        # QKV down projection and layernorm
        # =========================================
        if self.config.q_lora_rank is not None:
            q_compressed, _ = self.linear_q_down_proj(hidden_states)
            q_compressed = self.q_layernorm(q_compressed)
        else:
            q_compressed = hidden_states

        kv_combined, _ = self.linear_kv_down_proj(hidden_states)

        kv_compressed, k_pos_emb = self.split(
            kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
        )

        kv_compressed = self.kv_layernorm(kv_compressed)

        # =========================================
        # QKV up projection and RoPE apply
        # =========================================

        query, key, value = self.qkv_up_proj_and_rope_apply(
            q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb
        )

        return query, key, value
