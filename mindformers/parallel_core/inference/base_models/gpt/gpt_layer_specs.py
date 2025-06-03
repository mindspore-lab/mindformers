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
from mindformers.parallel_core.inference.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from mindformers.parallel_core.inference.transformer.flash_attention import FlashAttention
from mindformers.parallel_core.inference.transformer.dot_product_attention import DotProductAttention
from mindformers.parallel_core.inference.transformer.attention import (
    SelfAttention,
    SelfAttentionSubmodules,
)
from mindformers.parallel_core.inference.transformer.mlp import MLP, MLPSubmodules
from mindformers.parallel_core.inference.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    QKVParallelLinear,
)
from mindformers.parallel_core.inference.transformer.identity_op import IdentityOp
from mindformers.parallel_core.inference.transformer.norm import get_norm_cls


def get_gpt_layer_local_spec(
        num_experts: Optional[int] = None,
        moe_grouped_gemm: Optional[bool] = False,
        qk_layernorm: Optional[bool] = False,
        multi_latent_attention: Optional[bool] = False,
        normalization: Optional[str] = None,
        qk_l2_norm: Optional[bool] = False,
        use_flash_attention: Optional[bool] = True,
        sandwich_norm: Optional[bool] = False,
) -> ModuleSpec:
    """Use this spec for an implementation using only modules in Mcore Inference.


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        multi_latent_attention (bool, optional): To use multi latent attention. Defaults to False.
        normalization (str, optional): The type of normalization. Defaults to None.
        qk_l2_norm (bool, optional): To use l2 norm for queries/keys. Defaults to False.
        use_flash_attention (bool, optional): To use flash attention. Defaults to True.
        sandwich_norm (bool, optional): To use sandwich norm in the transformer layer. Defaults to False.

    Returns:
        ModuleSpec: Module specification with MCore modules

    """
    if num_experts or moe_grouped_gemm:
        raise NotImplementedError("moe spec is not currently supported.")
    if multi_latent_attention:
        raise NotImplementedError("`multi_latent_attention` is not currently supported.")
    if qk_l2_norm:
        raise NotImplementedError("`qk_l2_norm` is not currently supported.")

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
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=ColumnParallelLinear,
                    linear_fc2=RowParallelLinear
                )
            ),
            post_mlp_layernorm=get_norm_cls(normalization) if sandwich_norm else IdentityOp,
        )
    )
