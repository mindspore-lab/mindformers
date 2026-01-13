# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# This file is derived from Megatron-LM and adapted for MindSpore.
# Modifications:
#     - Adapted to MindSpore framework: replaced torch with mindspore, nn.Module with nn.Cell.
#     - Used mindspore.mint and mindspore.ops for tensor operations.
#     - Integrated with mindformers.parallel_core for module specification and building.
#     - Added support for TND input layout.
#     - Utilized MindFormers' Rotary Embedding implementation.
"""
Multi-head Latent Attention (MLA) mechanism with KV compression and rotary position encoding.

This module implements the Multi-head Latent Attention mechanism with low-rank compression
for KV projections and rotary position encoding support.
"""
from dataclasses import dataclass
from typing import Union
import math

from mindspore import nn, Tensor, mint, ops

from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.transformer_config import MLATransformerConfig
from mindformers.pynative.base_models.common.embeddings.rope_utils import ApplyRotaryPosEmb
from mindformers.pynative.base_models.common.embeddings.yarn_rotary_pos_embedding import _yarn_get_mscale


@dataclass
class MLASelfAttentionSubmodules:
    """
    Dataclass for MLA self-attention layer submodules.

    This dataclass defines the submodules required for building the MLA self-attention layer.
    
    Attributes:
        linear_qkv: Linear layer for combined query, key, and value projections.
            If q_lora_rank is not None, it concatenates linear_q_down_proj and linear_kv_down_proj;
            otherwise, it concatenates linear_q_proj and linear_kv_down_proj.
        linear_qb: Linear layer for query up projection.
        linear_kvb: Linear layer for key-value up projection.
        core_attention: Core attention mechanism implementation.
        linear_proj: Linear layer for final attention output projection.
        q_layernorm: Layer normalization for query projections (optional).
        k_layernorm: Layer normalization for key projections (optional).
    """
    linear_qkv: Union[ModuleSpec, type] = None
    linear_qb: Union[ModuleSpec, type] = None
    linear_kvb: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None


class MultiLatentAttention(nn.Cell):
    """
    Multi-head Latent Attention (MLA) with KV compression and rotary position encoding.
    
    Base class for Multi-head Latent Attention mechanism that implements KV compression
    and supports rotary position encoding. This class provides the core functionality
    for both self-attention and cross-attention variants.
    
    Args:
        config: Configuration object with MLA parameters.
        submodules: Submodules configuration for building the attention layer.
        layer_number: Layer index in the transformer stack.
        attention_type: Type of attention ("self" or "cross").
    """

    def __init__(
            self,
            config: MLATransformerConfig,
            submodules: Union[MLASelfAttentionSubmodules],
            layer_number: int,
            attention_type: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.layer_index = max(1, layer_number)
        self.attention_type = attention_type

        # model structure config
        self.use_flash_attention = self.config.use_flash_attention
        self.use_ring_attention = self.config.use_ring_attention
        self.use_eod_attn_mask_compression = self.config.use_eod_attn_mask_compression
        self.use_attn_mask_compression = self.config.use_attn_mask_compression
        self.seq_length = self.config.seq_length
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

        self.linear_proj = build_module(
            submodules.linear_proj,
            input_size=self.query_projection_size,
            output_size=self.config.hidden_size,
            params_dtype=config.params_dtype,
            compute_dtype=config.compute_dtype,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=True
        )

        self.shape = ops.shape
        self.reshape = mint.reshape
        self.transpose = mint.transpose
        self.cast = ops.cast

    def construct(self, x: Tensor, attention_mask=None, rotary_pos_emb=None, pad_zeros=None, actual_seq_len=None):
        """
        Forward pass of the Multi-head Latent Attention mechanism.
        
        Args:
            x: Input tensor with shape (seq_length, batch_size, hidden_size).
            attention_mask: Attention mask tensor (optional).
            rotary_pos_emb: Rotary position embedding tensor (optional).
            pad_zeros: Padding zeros tensor (not used).
            actual_seq_len: Actual sequence length for EOD mask compression (optional).
        
        Returns:
            Tensor: Output tensor with shape (seq_length, batch_size, hidden_size).
        """
        ori_dtype = x.dtype
        seq_len, bs, _ = self.shape(x)

        query, key, value = self.get_query_key_value_tensors(x, rotary_pos_emb=rotary_pos_emb)

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
                attn_out = self.transpose(attn_out, 0, 1)
            else:
                context_layer = self.core_attention(
                    query, key, value, attention_mask,
                )
                attn_out = self.reshape(context_layer, (seq_len, bs, -1))
        else:
            attn_out = self.core_attention(query, key, value, attention_mask)

        output = self.linear_proj(attn_out)[0]
        output = self.cast(output, ori_dtype)
        return output

    def sbh2tnd(self, x):
        """
        Convert a tensor from SBH/SBND layout to TND layout.
        
        Args:
            x: Input tensor with SBH/SBND layout.
        
        Returns:
            Tensor: Output tensor with TND layout.
        """
        seq_len, bs = x.shape[:2]
        x = self.reshape(x, (seq_len, bs, self.num_attention_heads, -1))
        x = self.transpose(x, 0, 1)
        x = self.reshape(x, (bs * seq_len, self.num_attention_heads, -1))
        return x


class MLASelfAttention(MultiLatentAttention):
    """
    MLA Self-attention layer implementation.
    
    This class implements the MLA self-attention layer following the same structure as Mindspeed A2.
    It inherits from MultiLatentAttention and provides self-attention specific functionality.
    
    Args:
        config: Configuration object with MLA parameters.
        submodules: Submodules configuration for building the self-attention layer.
        layer_number: Layer index in the transformer stack.

    Inputs:
        x: Input tensor with shape [seq_length, batch_size, hidden_size].
    
    Outputs:
        Tensor: Output tensor with shape [seq_length, batch_size, hidden_size].
    """

    def __init__(
            self,
            config: MLATransformerConfig,
            submodules: MLASelfAttentionSubmodules,
            layer_number: int,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attention_type="self",
        )
        self.use_tnd = config.input_layout == "TND"
        self.split = mint.split
        self.tile_kv = mint.tile
        self.cat = mint.cat
        self.apply_rotary_emb = ApplyRotaryPosEmb(config)
        self.apply_rotary_emb2 = ApplyRotaryPosEmb(config, for_k_pos_emb=True)
        self.reshape = mint.reshape

        if self.config.q_lora_rank is None:
            self.q_rank = self.config.num_attention_heads * self.q_head_dim
            self.q_layernorm = None
        else:
            self.q_rank = self.config.q_lora_rank
            if submodules.q_layernorm is not None:
                self.q_layernorm = build_module(
                    submodules.q_layernorm,
                    dim=self.config.q_lora_rank,
                    eps=self.config.layernorm_epsilon,
                    params_dtype=config.params_dtype,
                    compute_dtype=config.layernorm_compute_dtype
                )
            else:
                self.q_layernorm = None

            self.linear_qb = build_module(
                submodules.linear_qb,
                input_size=self.config.q_lora_rank,
                output_size=self.config.num_attention_heads * self.q_head_dim,
                params_dtype=config.params_dtype,
                compute_dtype=config.compute_dtype,
                init_method=self.config.init_method,
                bias=self.config.add_bias_linear or self.config.add_qkv_bias,
                skip_bias_add=False
            )

        self.linear_qkv = build_module(
            submodules.linear_qkv,
            input_size=self.config.hidden_size,
            output_size=self.q_rank + self.kv_lora_rank + self.qk_pos_emb_head_dim,
            params_dtype=config.params_dtype,
            compute_dtype=config.compute_dtype,
            init_method=self.config.init_method,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False
        )

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                dim=self.kv_lora_rank,
                eps=self.config.layernorm_epsilon,
                params_dtype=config.params_dtype,
                compute_dtype=config.layernorm_compute_dtype
            )
        else:
            self.k_layernorm = None

        self.linear_kvb = build_module(
            submodules.linear_kvb,
            input_size=self.kv_lora_rank,
            output_size=self.config.num_attention_heads * (
                    self.q_head_dim - self.qk_pos_emb_head_dim + self.v_head_dim),
            params_dtype=config.params_dtype,
            compute_dtype=config.compute_dtype,
            init_method=self.config.init_method,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False
        )

        self.linear_proj = build_module(
            submodules.linear_proj,
            input_size=self.query_projection_size,
            output_size=self.config.hidden_size,
            params_dtype=config.params_dtype,
            compute_dtype=config.compute_dtype,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=True
        )

    def get_query_key_value_tensors(self,
                                    hidden_states,
                                    rotary_pos_emb=None
                                    ):
        """
        Derive query, key, and value tensors from hidden states.
        
        This method generates query, key, and value tensors from the input hidden states
        using the configured projection layers and applies rotary position embeddings.
        
        Args:
            hidden_states: Input hidden states tensor with shape [seq_length, batch_size, hidden_size].
            rotary_pos_emb: Rotary position embedding tensor (optional).
        
        Returns:
            tuple: A tuple containing query, key, and value tensors.
                - query: Query tensor with shape [seq_length, batch_size, num_heads, head_dim].
                - key: Key tensor with shape [seq_length, batch_size, num_heads, head_dim].
                - value: Value tensor with shape [seq_length, batch_size, num_heads, v_head_dim].
        """
        seq_len, bs, _ = self.shape(hidden_states)

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
            k_pe = self.apply_rotary_emb2(
                k_pe,
                rotary_pos_emb,
                rotary_interleaved=self.config.rotary_interleaved,
                multi_latent_attention=self.config.multi_latent_attention
            )

        query = self.cat([q_nope, q_pe], 3)
        k_pe = self.tile_kv(k_pe, (1, 1, self.num_attention_heads, 1))
        key = self.cat([k_nope, k_pe], 3)

        return query, key, value
