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
"""GPT LayerSpec."""
from typing import Optional, Union

from mindformers.parallel_core.training_graph.base_models.gpt.moe_module_specs import get_moe_module_spec
from mindformers.parallel_core.training_graph.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear, \
    LinearNoTP
from mindformers.parallel_core.training_graph.transformer.attention import SelfAttention, SelfAttentionSubmodules, \
    SelfAttentionMegatron
from mindformers.parallel_core.training_graph.transformer.flash_attention import FlashAttention
from mindformers.parallel_core.training_graph.transformer.identity_op import IdentityOp
from mindformers.parallel_core.training_graph.transformer.mlp import MLP, MLPSubmodules
from mindformers.parallel_core.training_graph.transformer.multi_latent_attention import MLASelfAttention, \
    MLASelfAttentionSubmodules, MLASelfAttentionConcatenated, MLASelfAttentionSubmodulesConcatenated
from mindformers.parallel_core.training_graph.transformer.multi_token_prediction import \
    MultiTokenPredictionBlockSubmodules, get_mtp_layer_spec
from mindformers.parallel_core.training_graph.transformer.norm import get_norm_cls
from mindformers.parallel_core.training_graph.transformer.transformer_block import TransformerBlockSubmodules
from mindformers.parallel_core.training_graph.transformer.transformer_layer import TransformerLayer, \
    TransformerLayerSubmodules
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.utils.spec_utils import ModuleSpec


def get_mlp_module_spec(
        num_experts: Optional[int] = None,
        moe_grouped_gemm: Optional[bool] = True,
) -> ModuleSpec:
    """Helper function to get module spec for MLP/MoE"""
    if num_experts is None:
        return ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=ColumnParallelLinear,
                linear_fc2=RowParallelLinear,
            ),
        )

    return get_moe_module_spec(
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
    )


def get_gpt_layer_local_spec(
        num_experts: Optional[int] = None,
        moe_grouped_gemm: Optional[bool] = False,
        qk_layernorm: Optional[bool] = False,
        multi_latent_attention: Optional[bool] = False,
        qk_l2_norm: Optional[bool] = False,
        use_contiguous_weight_layout: Optional[bool] = True,
        mla_qkv_concat: Optional[bool] = True,
        fused_norm: Optional[bool] = True,
) -> ModuleSpec:
    """Use this spec for an implementation using only modules in Megatron-Core.


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        multi_latent_attention (bool, optional): To use MultiLatentAttention. Defaults to False.
        qk_l2_norm (bool, optional): To use l2 norm for queries/keys. Defaults to False.
        use_contiguous_weight_layout(bool, optional): Determines the weight arrangement in SelfAttention's QKV linear
            projection. Only affects SelfAttention layers. Uses contiguous layout: [Q_weights, K_weights, V_weights]
            when True. Uses interleaved head layout: [Q_head0, K_head0, V_head0, Q_head1, ...] when False.
            Defaults to True.
        mla_qkv_concat(bool, optional): If True, Multi Latent Attention computes q_compressed, k, kv_compressed in
            a single linear transformation; if False (default), computes them separately. Defaults to True.
        fused_norm (bool): Whether to use fused-normalization. Defaults to True.

    Returns:
        ModuleSpec: Module specification with Megatron-Core modules
    """
    if qk_l2_norm:
        raise NotImplementedError("L2Norm has not been implemented for GPT yet.")

    mlp = get_mlp_module_spec(
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm
    )

    if multi_latent_attention:
        if qk_l2_norm:
            raise ValueError("qk_l2_norm is not supported with MLA.")
        if mla_qkv_concat:
            self_attention = ModuleSpec(
                module=MLASelfAttentionConcatenated,
                submodules=MLASelfAttentionSubmodulesConcatenated(
                    linear_qkv=LinearNoTP,
                    linear_qb=ColumnParallelLinear,
                    linear_kvb=ColumnParallelLinear,
                    core_attention=FlashAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=get_norm_cls(fused_norm) if qk_layernorm else IdentityOp,
                    k_layernorm=get_norm_cls(fused_norm) if qk_layernorm else IdentityOp,
                ),
            )
        else:
            self_attention = ModuleSpec(
                module=MLASelfAttention,
                submodules=MLASelfAttentionSubmodules(
                    linear_q_down_proj=LinearNoTP,
                    linear_q_up_proj=ColumnParallelLinear,
                    linear_kv_down_proj=LinearNoTP,
                    linear_kv_up_proj=ColumnParallelLinear,
                    core_attention=FlashAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=get_norm_cls(fused_norm) if qk_layernorm else IdentityOp,
                    kv_layernorm=get_norm_cls(fused_norm) if qk_layernorm else IdentityOp,
                ),
            )
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=get_norm_cls(fused_norm),
                self_attention=self_attention,
                pre_mlp_layernorm=get_norm_cls(fused_norm),
                mlp=mlp,
            ),
        )

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=get_norm_cls(fused_norm),
            self_attention=ModuleSpec(
                module=SelfAttention if use_contiguous_weight_layout else SelfAttentionMegatron,
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=FlashAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=get_norm_cls(fused_norm) if qk_layernorm else IdentityOp,
                    k_layernorm=get_norm_cls(fused_norm) if qk_layernorm else IdentityOp,
                ),
            ),
            pre_mlp_layernorm=get_norm_cls(fused_norm),
            mlp=mlp
        )
    )


def get_gpt_decoder_block_spec(
        config: TransformerConfig,
        qk_l2_norm: Optional[bool] = False,
) -> TransformerBlockSubmodules:
    """GPT block spec."""

    # Layer specs.
    dense_layer_spec = get_gpt_layer_local_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=config.qk_layernorm,
        multi_latent_attention=config.multi_latent_attention,
        qk_l2_norm=qk_l2_norm,
        use_contiguous_weight_layout=config.use_contiguous_weight_layout,
        mla_qkv_concat=config.mla_qkv_concat,
        fused_norm=config.fused_norm
    )

    moe_layer_spec = get_gpt_layer_local_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        multi_latent_attention=config.multi_latent_attention,
        qk_l2_norm=qk_l2_norm,
        use_contiguous_weight_layout=config.use_contiguous_weight_layout,
        mla_qkv_concat=config.mla_qkv_concat,
        fused_norm=config.fused_norm
    )

    # Parse config.moe_layer_freq to determine the pattern of expert/dense layers.
    # 0 stands for dense layers, 1 stands for expert layers.
    # For integer N: Creates a pattern with one expert layer every N layers.
    # For string pattern: Evaluates the str directly (e.g. "[1,0,1]" for alternating expert/dense).
    if config.first_k_dense_replace and config.moe_layer_freq > 1:
        raise ValueError("Configuration conflict: 'first_k_dense_replace' cannot be"
                         " used together with 'moe_layer_freq > 1'.")
    if config.first_k_dense_replace:
        moe_layer_pattern = [0] * config.first_k_dense_replace + \
                            [1] * (config.num_layers - config.first_k_dense_replace)
    elif isinstance(config.moe_layer_freq, int):
        moe_layer_pattern = [1 if (i % config.moe_layer_freq == 0) else 0 for i in range(config.num_layers)]
    elif isinstance(config.moe_layer_freq, list):
        moe_layer_pattern = config.moe_layer_freq
        if len(moe_layer_pattern) != config.num_layers:
            raise ValueError(f"Invalid length of moe_layer_pattern: {len(moe_layer_pattern)}, "
                             f"expected {config.num_layers}, "
                             f"current moe layer pattern: {config.moe_layer_freq}")
    else:
        raise ValueError(
            f"Invalid moe_layer_freq: {type(config.moe_layer_freq)}, {config.moe_layer_freq}"
        )

    # Create the layer specs for the model.
    layer_specs = []
    for layer_number in range(config.num_layers):
        if moe_layer_pattern[layer_number] == 1:
            layer_specs.append(moe_layer_spec)
        elif moe_layer_pattern[layer_number] == 0:
            layer_specs.append(dense_layer_spec)
        else:
            raise ValueError(f"Invalid layer pattern: {moe_layer_pattern}")

    # Block spec.
    block_spec = TransformerBlockSubmodules(
        layer_specs=layer_specs, layer_norm=get_norm_cls(config.fused_norm))

    return block_spec


def get_gpt_mtp_block_spec(
        config: TransformerConfig,
        spec: Union[TransformerBlockSubmodules, ModuleSpec],
) -> MultiTokenPredictionBlockSubmodules:
    """GPT Multi-Token Prediction (MTP) block spec."""
    num_layers_to_build = config.mtp_num_layers if config.mtp_num_layers else 0
    if num_layers_to_build == 0:
        return None

    if isinstance(spec, TransformerBlockSubmodules):
        # get the spec for the last layer of decoder block
        transformer_layer_spec = spec.layer_specs[-1]
    elif isinstance(spec, ModuleSpec) and spec.module == TransformerLayer:
        transformer_layer_spec = spec
    else:
        raise ValueError(f"Invalid spec: {spec}")

    mtp_layer_spec = get_mtp_layer_spec(
        transformer_layer_spec=transformer_layer_spec, fused_norm=config.fused_norm)
    mtp_num_layers = config.mtp_num_layers if config.mtp_num_layers else 0
    mtp_layer_specs = [mtp_layer_spec] * mtp_num_layers

    return MultiTokenPredictionBlockSubmodules(layer_specs=mtp_layer_specs)
