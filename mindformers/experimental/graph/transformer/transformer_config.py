# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Config Class"""
from dataclasses import dataclass
import mindspore.common.dtype as mstype


@dataclass
class ModelParallelConfig:
    """
    Base configuration.

    The initialization function has an argument for each parameter.
    """
    data_parallel: int = 1
    """The split number of data parallel."""

    context_parallel: int = 1
    """The split number of context parallel."""

    tensor_parallel: int = 1
    """The split number of tensor parallel."""

    vocab_emb_dp: bool = True
    """Whether to split the vocabulary only along the dp dimension."""


@dataclass
class TransformerConfig(ModelParallelConfig):
    """
    Configuration object for mindformers transformers.

    The initialization function has an argument for each parameter, including those in ModelParallelConfig.
    """
    num_layers: int = 0
    """Number of hidden layers in the Transformer encoder."""

    hidden_size: int = 0
    """Dimensionality of the encoder layers and the pooler layer."""

    ffn_hidden_size: int = 0
    """Transformer Feed-Forward Network hidden size."""

    num_attention_heads: int = 0
    """Number of attention heads for each attention layer in the Transformer encoder."""

    hidden_dropout: float = 0.1
    """Dropout probability for transformer hidden state."""

    hidden_act: str = 'gelu'
    """Specifies the activation function used in hidden layers"""

    attention_dropout: float = 0.1
    """Post attention dropout probability."""

    layernorm_epsilon: float = 1e-5
    """Epsilon value for any LayerNorm operations."""

    add_bias_linear: bool = False
    """Include a bias term in all linear layers (QKV projections, after core attention, and two in
    MLP layer)."""

    add_qkv_bias: bool = False
    """Add a bias term only for QKV projections."""

    mlp_has_gate: bool = False
    """Indicates whether the MLP layer includes a gating mechanism."""

    gated_linear_unit: bool = False
    """Use a gated linear unit for the first linear layer in the MLP."""

    compute_dtype: mstype = mstype.float32
    """Linear layer compute dtype."""

    param_init_dtype: mstype = mstype.float32
    """parameter initial dtype."""

    softmax_compute_dtype: mstype = mstype.float32
    """softmax compute dtype."""

    fp16: bool = False
    """If true, train with fp16 mixed precision training."""

    bf16: bool = False
    """If true, train with bf16 mixed precision training."""

    mask_func_type: str = "attn_mask_fill"
    """use type of masking function"""

    group_query_attention: bool = False
    """use Group Query Attention (GQA) mechanism."""

    kv_num_heads: int = 0
    """Define multi group head attention heads number."""

    use_flash_attn: bool = False
    """Whether enable flash attention ops."""

    qkv_concat: bool = True
    """whether to concatenate query, key, and value tensors."""

    use_attn_mask_compression: bool = False
    """whether to use attention mask compression."""

    apply_residual_connection_post_layernorm: bool = True
    """whether to apply residual connections after layer normalization."""

    normalization: str = "RMSNorm"
    """use type of normalization"""

    init_method: callable = None
    """Method to initialize weights. Note that bias is always set to zero. Should be a function that
    takes a single Tensor and initializes it."""

    weight_init: str = "normal"
    """initialization method for weights."""

    apply_query_key_layer_scaling: bool = False
    """If true, scale Q * K^T by 1 / layer-number."""
