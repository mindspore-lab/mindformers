# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Modification points:
# 1. Replace all interfaces with MindSpore TransFormers'.
# 2. Modify some input parameters for MindSpore TransFormers.
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
from typing import Optional

from mindformers.pynative.layers.linear import Linear
from mindformers.pynative.transformers.attention import SelfAttentionSubmodules, SelfAttention
from mindformers.pynative.layers.flash_attention import FlashAttention
from mindformers.pynative.layers.identity_op import IdentityOp
from mindformers.pynative.transformers.mlp import MLP, MLPSubmodules
from mindformers.pynative.layers.layer_norm import get_norm_cls
from mindformers.pynative.transformers.transformer_block import TransformerBlockSubmodules
from mindformers.pynative.transformers.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.utils.spec_utils import ModuleSpec


def get_mlp_module_spec(
        num_experts: Optional[int] = None,
) -> ModuleSpec:
    """Helper function to get module spec for MLP/MoE"""
    mlp = MLP
    return ModuleSpec(
        module=mlp,
        submodules=MLPSubmodules(
            linear_fc1=Linear,
            linear_fc2=Linear,
        ),
    )


def get_gpt_layer_local_spec(
        num_experts: Optional[int] = None,
        qk_layernorm: Optional[bool] = False,
        fused_norm: Optional[bool] = True,
        normalization: Optional[str] = "RMSNorm",
) -> ModuleSpec:
    """Use this spec for an implementation using only modules in Megatron-Core.


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        multi_latent_attention (bool, optional): To use MultiLatentAttention. Defaults to False.
        fused_norm (bool): Whether to use fused-normalization. Defaults to True.
        normalization (str): The type of the norm. Defaults to RMSNorm.
    Returns:
        ModuleSpec: Module specification with Megatron-Core modules
    """

    mlp = get_mlp_module_spec(
        num_experts=num_experts,
    )

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=get_norm_cls(normalization, fused_norm),
            self_attention=ModuleSpec(
                module=SelfAttention,
                submodules=SelfAttentionSubmodules(
                    linear_qkv=Linear,
                    core_attention=FlashAttention,
                    linear_proj=Linear,
                    q_layernorm=get_norm_cls(normalization, fused_norm) if qk_layernorm else IdentityOp,
                    k_layernorm=get_norm_cls(normalization, fused_norm) if qk_layernorm else IdentityOp,
                ),
            ),
            pre_mlp_layernorm=get_norm_cls(normalization, fused_norm),
            mlp=mlp
        )
    )


def get_gpt_decoder_block_spec(
        config: TransformerConfig,
) -> TransformerBlockSubmodules:
    """GPT block spec."""

    # Layer specs.
    dense_layer_spec = get_gpt_layer_local_spec(
        num_experts=None,
        qk_layernorm=config.qk_layernorm,
        fused_norm=config.fused_norm,
    )

    moe_layer_spec = None

    # Parse config.moe_layer_freq to determine the pattern of expert/dense layers.
    # 0 stands for dense layers, 1 stands for expert layers.
    # For integer N: Creates a pattern with one expert layer every N layers.
    # For string pattern: Evaluates the str directly (e.g. "[1,0,1]" for alternating expert/dense).
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
