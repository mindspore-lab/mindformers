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
"""gpt spec utils"""
from typing import Optional

from mindformers.parallel_core.utils.spec_utils import ModuleSpec
from mindformers.parallel_core.inference.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)
from mindformers.parallel_core.inference.transformer.flash_attention import FlashAttention
from mindformers.parallel_core.inference.transformer.dot_product_attention import DotProductAttention
from mindformers.parallel_core.inference.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
    FusedMLASelfAttention,
)
from mindformers.parallel_core.inference.transformer.attention import (
    SelfAttention,
    SelfAttentionSubmodules,
)
from mindformers.parallel_core.inference.transformer.mlp import MLP, MLPSubmodules
from mindformers.parallel_core.inference.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    MergedColumnParallelLinear
)
from mindformers.parallel_core.inference.transformer.identity_op import IdentityOp
from mindformers.parallel_core.inference.transformer.norm import get_norm_cls
from mindformers.parallel_core.inference.base_models.gpt.moe_module_spec import get_moe_module_spec
from mindformers.parallel_core.inference.transformer.transformer_block import TransformerBlockSubmodules
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.inference.utils import get_num_layers_and_offset


def get_gpt_layer_local_spec(
        num_experts: Optional[int] = None,
        moe_grouped_gemm: Optional[bool] = True,
        qk_layernorm: Optional[bool] = False,
        gated_linear_unit: Optional[bool] = True,
        multi_latent_attention: Optional[bool] = False,
        normalization: Optional[str] = None,
        qk_l2_norm: Optional[bool] = False,
        use_flash_attention: Optional[bool] = True,
        sandwich_norm: Optional[bool] = False,
        use_alltoall: Optional[bool] = False,
        use_fused_mla: Optional[bool] = False,
) -> ModuleSpec:
    """Use this spec for an implementation using only modules in Mcore Inference.


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to True.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        gated_linear_unit (bool, optional): Whether to include a gated linear unit. Defaults to True.
        multi_latent_attention (bool, optional): To use multi latent attention. Defaults to False.
        normalization (str, optional): The type of normalization. Defaults to None.
        qk_l2_norm (bool, optional): To use l2 norm for queries/keys. Defaults to False.
        use_flash_attention (bool, optional): To use flash attention. Defaults to True.
        sandwich_norm (bool, optional): To use sandwich norm in the transformer layer. Defaults to False.
        use_alltoall (bool, optional): To use alltoall communication. Defaults to False.

    Returns:
        ModuleSpec: Module specification with MCore modules

    """

    if qk_l2_norm:
        raise NotImplementedError("`qk_l2_norm` is not currently supported.")

    mlp = get_mlp_module_spec(
        num_experts=num_experts,
        gated_linear_unit=gated_linear_unit,
        moe_grouped_gemm=moe_grouped_gemm,
        use_alltoall=use_alltoall,
    )

    if multi_latent_attention:
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=get_norm_cls(normalization) if not use_fused_mla else IdentityOp,
                self_attention=ModuleSpec(
                    module=MLASelfAttention if not use_fused_mla else FusedMLASelfAttention,
                    submodules=MLASelfAttentionSubmodules(
                        input_layernorm=get_norm_cls(normalization) if use_fused_mla else IdentityOp,
                        linear_q_proj=ColumnParallelLinear,
                        linear_qkv_down_proj=ReplicatedLinear,
                        linear_q_up_proj=ColumnParallelLinear,
                        linear_kv_down_proj=ReplicatedLinear,
                        linear_kv_up_proj=ColumnParallelLinear,
                        core_attention=FlashAttention if use_flash_attention else DotProductAttention,
                        linear_proj=RowParallelLinear,
                        q_layernorm=get_norm_cls(normalization) if qk_layernorm else IdentityOp,
                        kv_layernorm=get_norm_cls(normalization) if qk_layernorm else IdentityOp,
                    ),
                ),
                pre_mlp_layernorm=get_norm_cls(normalization),
                mlp=mlp,
            )
        )

    self_attn = ModuleSpec(
        module=SelfAttention,
        submodules=SelfAttentionSubmodules(
            core_attention=FlashAttention if use_flash_attention else DotProductAttention,
            linear_proj=RowParallelLinear,
            linear_qkv=QKVParallelLinear,
            q_layernorm=get_norm_cls(normalization) if qk_layernorm else IdentityOp,
            k_layernorm=get_norm_cls(normalization) if qk_layernorm else IdentityOp,
        )
    )
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=get_norm_cls(normalization),
            self_attention=self_attn,
            post_self_attn_layernorm=get_norm_cls(normalization) if sandwich_norm else IdentityOp,
            pre_mlp_layernorm=get_norm_cls(normalization),
            mlp=mlp,
            post_mlp_layernorm=get_norm_cls(normalization) if sandwich_norm else IdentityOp,
        )
    )


def get_gpt_decoder_block_spec(
        config: TransformerConfig,
        normalization: Optional[str] = None,
        qk_l2_norm: Optional[bool] = False,
) -> TransformerLayerSubmodules:
    """GPT block spec."""
    # layer specs.
    dense_layer_spec = get_gpt_layer_local_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=config.qk_layernorm,
        gated_linear_unit=config.gated_linear_unit,
        multi_latent_attention=config.multi_latent_attention,
        normalization=normalization,
        use_flash_attention=config.use_flash_attention,
        qk_l2_norm=qk_l2_norm,
        use_alltoall=config.use_alltoall,
        use_fused_mla=config.use_fused_mla,
    )

    moe_layer_spec = get_gpt_layer_local_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=True,
        qk_layernorm=config.qk_layernorm,
        gated_linear_unit=config.gated_linear_unit,
        multi_latent_attention=config.multi_latent_attention,
        normalization=normalization,
        use_flash_attention=config.use_flash_attention,
        qk_l2_norm=qk_l2_norm,
        use_alltoall=config.use_alltoall,
        use_fused_mla=config.use_fused_mla,
    )

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
            raise ValueError(
                f"Invalid length of moe_layer_pattern: {len(moe_layer_pattern)}, "
                f"expected {config.num_layers}, "
                f"current moe layer pattern: {config.moe_layer_freq}"
            )

    else:
        raise ValueError(
            f"Invalid moe_layer_freq: {type(config.moe_layer_freq)}, {config.moe_layer_freq}"
        )

    # Create the layer specs for the model.
    layer_specs = []
    num_layers, offset = get_num_layers_and_offset(config)
    for layer_number in range(num_layers):
        if moe_layer_pattern[layer_number + offset] == 1:
            layer_specs.append(moe_layer_spec)
        elif moe_layer_pattern[layer_number + offset] == 0:
            layer_specs.append(dense_layer_spec)
        else:
            raise ValueError(f"Invalid layer pattern: {moe_layer_pattern}")

    # Block spec.
    block_spec = TransformerBlockSubmodules(layer_specs=layer_specs, layer_norm=get_norm_cls(config.normalization))

    return block_spec


def get_mlp_module_spec(
        num_experts: Optional[int] = None,
        gated_linear_unit: Optional[bool] = True,
        moe_grouped_gemm: Optional[bool] = True,
        use_alltoall: Optional[bool] = False,
) -> ModuleSpec:
    """Helper function to get module spec for MLP/MoE."""
    if num_experts is None:
        # Dense MLP w or w/o modules.
        return ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=MergedColumnParallelLinear if gated_linear_unit else ColumnParallelLinear,
                linear_fc2=RowParallelLinear,
            ),
        )
    # Mixture of experts with modules.
    return get_moe_module_spec(
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        use_alltoall=use_alltoall,
    )
