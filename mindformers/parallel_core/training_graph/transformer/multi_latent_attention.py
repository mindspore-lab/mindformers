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
from dataclasses import dataclass
from typing import Union
import math

from mindspore import nn, Tensor
from mindspore.ops.auto_generate import Cast, Shape, Reshape, Transpose, Tile, Concat, SplitWithSize, ExpandDims
from mindspore.ops import functional as F
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.transformer_config import MLATransformerConfig
from mindformers.parallel_core.training_graph.base_models.common.embeddings.rope_utils import ApplyRotaryPosEmb
from mindformers.parallel_core.training_graph.base_models.common.embeddings.yarn_rotary_pos_embedding import (
    _yarn_get_mscale)
from mindformers.parallel_core.training_graph.device_matrix import layout


@dataclass
class MLASelfAttentionSubmodulesConcatenated:
    """
    Submodules for the MLA self-attention layer.

    Differences from MLASelfAttentionSubmodules (Megatron-style):
        linear_qkv: concat(linear_q_down_proj, linear_kv_down_proj) if q_lora_rank is not None else
            concat(linear_q_proj, linear_kv_down_proj)
        linear_qb: linear_q_up_proj
        linear_kvb: linear_kv_up_proj
    """
    linear_qkv: Union[ModuleSpec, type] = None
    linear_qb: Union[ModuleSpec, type] = None
    linear_kvb: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None


@dataclass
class MLASelfAttentionSubmodules:
    """Submodules for the MLA self-attention layer with Megatron structure."""
    linear_q_proj: Union[ModuleSpec, type] = None
    linear_q_down_proj: Union[ModuleSpec, type] = None
    linear_q_up_proj: Union[ModuleSpec, type] = None
    linear_kv_down_proj: Union[ModuleSpec, type] = None
    linear_kv_up_proj: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    kv_layernorm: Union[ModuleSpec, type] = None


class MultiLatentAttention(nn.Cell):
    """Multi-head Latent Attention (MLA) with KV compression and rotary position encoding."""

    def __init__(
            self,
            config: MLATransformerConfig,
            submodules: Union[MLASelfAttentionSubmodulesConcatenated, MLASelfAttentionSubmodules],
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
        self.use_seq_parallel = self.config.sequence_parallel
        self.seq_length = self.config.seq_length
        self.dp = 1 if self.config.data_parallel_size is None else self.config.data_parallel_size
        self.tp = 1 if self.config.tensor_model_parallel_size is None else self.config.tensor_model_parallel_size
        self.cp = 1 if self.config.context_parallel_size is None else self.config.context_parallel_size
        self.num_attention_heads = self.config.num_attention_heads
        self.query_projection_size = self.config.v_head_dim * self.config.num_attention_heads
        self.qk_head_dim = self.config.qk_head_dim
        self.qk_pos_emb_head_dim = self.config.qk_pos_emb_head_dim
        self.q_head_dim = self.qk_head_dim + self.qk_pos_emb_head_dim
        self.kv_lora_rank = self.config.kv_lora_rank
        self.v_head_dim = self.config.v_head_dim
        self.input_layout = self.config.input_layout
        self.compute_dtype = self.config.compute_dtype

        # Define ulysses context parallel related parameters
        self.cp_ds = self.config.hierarchical_context_parallel_sizes
        self.cp_co = self.cp // self.cp_ds

        if self.num_attention_heads % (self.tp * self.cp_ds) != 0:
            raise ValueError("For 'ParallelAttention', the class variable 'num_heads' must be a multiple of "
                             "'tensor_parallel * ulysses_cp_num', but got num_heads is {}, tensor_parallel is {}, "
                             "ulysses_cp_num is {}."
                             .format(self.num_attention_heads, self.tp, self.cp_ds))

        zero_pad_length = self.q_head_dim - self.v_head_dim
        if zero_pad_length < 0:
            raise ValueError("qk_head_dim + qk_pos_emb_head_dim should not less than v_head_dim")

        mscale = _yarn_get_mscale(self.config.rotary_scaling_factor, self.config.mscale)
        self.softmax_scale = mscale * mscale / math.sqrt(self.q_head_dim)

        self.core_attention = build_module(
            submodules.core_attention,
            config=self.config,
            layer_number=self.layer_index,
            softmax_scale=self.softmax_scale,
        )

        # Output.
        self.linear_proj = build_module(
            submodules.linear_proj,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=True
        )

        if self.cp_ds > 1:
            self._ulysses_initial()
        self.shape = Shape()
        self.reshape = Reshape()
        self.bs_transpose = Transpose()
        self.tnd_transpose = Transpose()
        self.cast = Cast()

        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.sharding_propagation()
        elif _get_parallel_mode() in (ParallelMode.SEMI_AUTO_PARALLEL,):
            self.shard()

    def _ulysses_initial(self):
        """Initialize ulysses related operations."""
        self.transpose_ulysses = Transpose()
        self.transpose_a2a = Transpose()
        self.transpose_ulysses_merger_a2a = Transpose()
        self.transpose_ulysses_merger = Transpose()

        dp = self.dp
        tp = self.tp
        cp = self.cp

        self.linear_proj.matmul.shard(in_strategy=((dp * cp, tp), (tp, 1)), out_strategy=((dp * cp * tp, 1),))
        self.transpose_ulysses.shard(((cp, dp, tp, 1, 1, 1),))
        self.transpose_a2a.shard(((self.cp_co, dp, self.cp_ds, tp, 1, 1),))
        self.transpose_ulysses_merger_a2a.shard(((self.cp_co, dp, self.cp_ds, tp, 1, 1),))
        self.transpose_ulysses_merger.shard(((cp, dp, 1, tp, 1, 1),))

    def sharding_propagation(self):
        """Set parallel strategy."""

    def shard(self):
        """Set parallel strategy."""
        dp = self.dp
        tp = self.tp
        cp = self.cp

        self.bs_transpose.shard(((dp, cp, tp),))
        self.tnd_transpose.shard(((cp, dp, tp, 1),))

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

        # with ulysses context parallel, insert all to all before FA
        if self.cp > 1:
            if self.cp_ds > 1:
                # For query & key & value, transpose from [S, B, N, D] back to [B, S, N, D]
                query = self._ulysses_qkv_a2a(query)
                key = self._ulysses_qkv_a2a(key)
                value = self._ulysses_qkv_a2a(value)
            else:
                # Merge heads for query and key
                query = self.reshape(query, (seq_len, bs, -1))
                key = self.reshape(key, (seq_len, bs, -1))
                value = self.reshape(value, (seq_len, bs, -1))

        if self.input_layout == "TND":
            query = self.sbh2tnd(query)
            key = self.sbh2tnd(key)
            value = self.sbh2tnd(value)

        query = self.cast(query, self.compute_dtype)
        key = self.cast(key, self.compute_dtype)
        value = self.cast(value, self.compute_dtype)
        if self.use_flash_attention:
            if self.use_eod_attn_mask_compression:
                context_layer = self.core_attention(
                    query, key, value, attention_mask,
                    actual_seq_qlen=actual_seq_len, actual_seq_kvlen=actual_seq_len
                )
                attn_out = self.reshape(context_layer, (bs, seq_len, -1))
                attn_out = self.bs_transpose(attn_out, (1, 0, 2))
            else:
                context_layer = self.core_attention(
                    query, key, value, attention_mask,
                )
                if self.cp > 1 and self.cp_ds > 1:
                    context_layer = self._ulysses_context_layer_a2a(context_layer)
                attn_out = self.reshape(context_layer, (seq_len, bs, -1))
        else:
            attn_out = self.core_attention(query, key, value, attention_mask)

        output = self.linear_proj(attn_out)[0]  # dp, mp -> dp, 1 / dp * mp, 1
        output = self.cast(output, ori_dtype)
        return output

    def sbh2tnd(self, x):
        """
        Convert a input tensor from SBH/SBND layout to TND layout.

        Inputs:
            x: input tensor

        Output:
            x_merge: the TND output tensor
        """
        seq_len, bs = x.shape[:2]
        x = self.reshape(x, (seq_len, bs, self.num_attention_heads, -1))
        x = self.tnd_transpose(x, (1, 0, 2, 3))
        x = self.reshape(x, (bs * seq_len, self.num_attention_heads, -1))
        return x

    def _ulysses_qkv_a2a(self, qkv):
        """Given a qkv tensor with shape of (seq_len, bs, n_head, head_dim),
        insert all to all in right place using transpose with specific shard strategy.
        refers to <https://arxiv.org/abs/2309.14509>

        Args:
            qkv (Tensor): qkv after rotary embedding and before attention, with shape of (B, S, N, D)

        Returns:
            Tensor: qkv tensor after all to all commu.
        """
        seq_len, bs, _, _ = F.shape(qkv)
        new_shape = (seq_len, bs, self.tp, self.cp_ds, -1, self.q_head_dim)
        # [seq_len, bs, n_head, head_dim] -> [seq_len, bs, n_head/cp_ds, cp_ds, head_dim]
        qkv = self.reshape(qkv, new_shape)
        # [seq_len, bs, n_head/cp_ds, cp_ds, head_dim] -> [seq_len, bs, cp_ds, n_head/cp_ds, head_dim]
        qkv = self.transpose_ulysses(qkv, (0, 1, 3, 2, 4, 5))
        # insert all-to-all (cp, dp, 1, tp, 1) -> (cp_co, dp, cp_ds, tp, 1)
        qkv = self.transpose_a2a(qkv, (0, 1, 2, 3, 4, 5))
        # reshape to SBH, here set -1 to H, for kv head could be different from q head
        qkv = F.reshape(qkv, (seq_len, bs, -1))
        return qkv

    def _ulysses_context_layer_a2a(self, context_layer):
        """Given the context_layer tensor after fa, with shape of (seq_len, bs, hidden_size),
        insert all to all in right place using transpose with specific shard strategy.
        refers to <https://arxiv.org/abs/2309.14509>

        Args:
            context_layer (Tensor): context layer after attention, with shape of (B, S, H)

        Returns:
            Tensor: context layer tensor after all to all communication.
        """
        seq_len, bs, _ = F.shape(context_layer)
        new_shape = (seq_len, bs, self.cp_ds, self.tp, -1, self.q_head_dim)
        context_layer = F.reshape(context_layer, new_shape)
        # insert all-to-all back (cp_co, dp, cp_ds, tp, 1) -> (cp, dp, 1, tp, 1)
        context_layer = self.transpose_ulysses_merger_a2a(context_layer, (0, 1, 2, 3, 4, 5))
        context_layer = self.transpose_ulysses_merger(context_layer, (0, 1, 3, 2, 4, 5))
        # reshape back to SBH
        context_layer = F.reshape(context_layer, (seq_len, bs, self.query_projection_size))
        return context_layer


class MLASelfAttentionConcatenated(MultiLatentAttention):
    """
    MLA Self-attention layer class, the implementation follows the same structure as Mindspeed A2.

    Differences fromMLASelfAttention (Megatron-style) lie mainly in the weights:
        linear_qkv: concat(linear_q_down_proj, linear_kv_down_proj) if q_lora_rank is not None else
            concat(linear_q_proj, linear_kv_down_proj)
        linear_qb: linear_q_up_proj
        linear_kvb: linear_kv_up_proj

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
            self,
            config: MLATransformerConfig,
            submodules: MLASelfAttentionSubmodulesConcatenated,
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
        self.use_seq_parallel = config.sequence_parallel
        self.split = SplitWithSize()
        self.split_3d = SplitWithSize()
        self.tile_kv = Tile()
        self.pe_concat = Concat(axis=3)
        self.pe_tnd_concat = Concat(axis=3)
        self.apply_rotary_emb = ApplyRotaryPosEmb(config)
        self.apply_rotary_emb2 = ApplyRotaryPosEmb(config, for_k_pos_emb=True)
        self.reshape = Reshape()

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
                    eps=self.config.layernorm_epsilon
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
                skip_bias_add=False
            )

        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.config.hidden_size,
            self.q_rank + self.kv_lora_rank + self.qk_pos_emb_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False
        )

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                dim=self.kv_lora_rank,
                config=self.config,
                eps=self.config.layernorm_epsilon
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
            skip_bias_add=False
        )

        self.linear_proj = build_module(
            submodules.linear_proj,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=True
        )

        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.sharding_propagation()
        elif _get_parallel_mode() in (ParallelMode.SEMI_AUTO_PARALLEL,):
            self.shard_self_attn()

    def sharding_propagation(self):
        dp = self.config.data_parallel_size
        tp = self.config.tensor_model_parallel_size

        self.tile_kv.shard(((1, dp, tp, 1),))

    def shard_self_attn(self):
        """sharding for MLASelfAttentionConcatenated with semi_auto_parallel"""
        dp = self.config.data_parallel_size
        tp = self.config.tensor_model_parallel_size
        cp = self.config.context_parallel_size

        self.tile_kv.shard((layout("cp", "dp", "None", "None"),))
        self.pe_concat.shard(((cp, dp, tp, 1), (cp, dp, tp, 1)))
        self.split.shard((layout("cp", "dp", "tp", "None"),))
        self.split_3d.shard((layout(("cp", "tp"), "dp", "None"),))

        if self.q_layernorm is not None:
            self.q_layernorm.shard(self.config, in_strategy=(layout(("cp", "tp"), "dp", "None"), layout("None",)))
        if self.k_layernorm is not None:
            self.k_layernorm.shard(self.config, in_strategy=(layout(("cp", "tp"), "dp", "None"), layout("None",)))

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
        seq_len, bs, _ = self.shape(hidden_states)

        qkv_combo = self.linear_qkv(hidden_states)[0]

        q_a, compressed_kv, k_pe = self.split_3d(
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
            k_pe = self.apply_rotary_emb2(
                k_pe,
                rotary_pos_emb,
                rotary_interleaved=self.config.rotary_interleaved,
                multi_latent_attention=self.config.multi_latent_attention
            )

        query = self.pe_concat([q_nope, q_pe])
        k_pe = self.tile_kv(k_pe, (1, 1, self.num_attention_heads, 1))
        key = self.pe_concat([k_nope, k_pe])

        return query, key, value


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
        self.split = SplitWithSize()
        self.split_3d = SplitWithSize()
        self.tile_kv = Tile()
        self.pe_concat = Concat(axis=3)
        self.pe_tnd_concat = Concat(axis=3)
        self.apply_rotary_emb = ApplyRotaryPosEmb(config)
        self.apply_rotary_emb2 = ApplyRotaryPosEmb(config, for_k_pos_emb=True)
        self.reshape = Reshape()
        self.expand_dims = ExpandDims()

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
                eps=self.config.layernorm_epsilon
            )

        self.kv_layernorm = build_module(
            submodules.kv_layernorm,
            dim=self.config.kv_lora_rank,
            config=self.config,
            eps=self.config.layernorm_epsilon
        )

        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.sharding_propagation()
        elif _get_parallel_mode() in (ParallelMode.SEMI_AUTO_PARALLEL,):
            self.shard_self_attn()

    def shard_self_attn(self):
        """sharding for MLASelfAttention with semi_auto_parallel"""
        dp = self.config.data_parallel_size
        tp = self.config.tensor_model_parallel_size
        cp = self.config.context_parallel_size

        self.tile_kv.shard((layout("cp", "dp", "None", "None"),))
        self.pe_concat.shard(((cp, dp, tp, 1), (cp, dp, tp, 1)))
        self.split.shard((layout("cp", "dp", "tp", "None"),))
        self.split_3d.shard((layout(("cp", "tp"), "dp", "None"),))
        self.expand_dims.shard((layout("cp", "dp", "None"),))

        if hasattr(self, "q_layernorm") and self.q_layernorm is not None:
            self.q_layernorm.shard(self.config, in_strategy=(layout(("cp", "tp"), "dp", "None"), layout("None",)))
        if self.kv_layernorm is not None:
            self.kv_layernorm.shard(self.config, in_strategy=(layout(("cp", "tp"), "dp", "None"), layout("None",)))

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

        k_pos_emb = self.expand_dims(k_pos_emb, 2)

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
            k_pos_emb = self.apply_rotary_emb2(
                k_pos_emb,
                rotary_pos_emb,
                rotary_interleaved=self.config.rotary_interleaved,
                multi_latent_attention=self.config.multi_latent_attention
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

        kv_compressed, k_pos_emb = self.split_3d(
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
