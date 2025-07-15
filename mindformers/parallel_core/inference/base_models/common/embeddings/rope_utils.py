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
"""
Rotary position embedding utils.
"""
import mindspore.common.dtype as mstype

from mindformers.parallel_core.transformer_config import TransformerConfig, MLATransformerConfig
from mindformers.parallel_core.inference.base_models.common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding,
    Llama3RotaryEmbedding
)
from mindformers.parallel_core.inference.base_models.common.embeddings.yarn_rotary_pos_embedding import \
    YaRNScalingRotaryEmbedding


def _get_default(**kwargs):
    """Instantiate a RotaryEmbedding object"""
    return RotaryEmbedding(
        kv_channels=kwargs["kv_channels"],
        rotary_percent=kwargs["rotary_percent"],
        rotary_base=kwargs["rotary_base"],
        rotary_dtype=kwargs["rotary_dtype"],
        seq_len_interpolation_factor=kwargs["seq_len_interpolation_factor"],
        rotary_cos_format=kwargs["rotary_cos_format"],
        max_position_embeddings=kwargs["original_max_position_embeddings"],
    )


def _get_llama3(**kwargs):
    """Instantiate a Llama3RotaryEmbedding object"""
    config: TransformerConfig = kwargs["config"]
    return Llama3RotaryEmbedding(
        kv_channels=kwargs["kv_channels"],
        rotary_percent=kwargs["rotary_percent"],
        rotary_base=kwargs["rotary_base"],
        rotary_dtype=kwargs["rotary_dtype"],
        seq_len_interpolation_factor=kwargs["seq_len_interpolation_factor"],
        rotary_cos_format=kwargs["rotary_cos_format"],
        scaling_factor=config.rotary_scaling_factor,
        max_position_embeddings=kwargs["max_position_embeddings"],
        low_freq_factor=getattr(config, "low_freq_factor", 1),
        high_freq_factor=getattr(config, "low_freq_factor", 8),
        orig_max_position=kwargs["original_max_position_embeddings"],
    )


def _get_yarn(**kwargs):
    """Instantiate a YaRNScalingRotaryEmbedding object"""
    config: MLATransformerConfig = kwargs["config"]
    return YaRNScalingRotaryEmbedding(
        kv_channels=kwargs["kv_channels"],
        rotary_percent=kwargs["rotary_percent"],
        rotary_base=kwargs["rotary_base"],
        rotary_dtype=kwargs["rotary_dtype"],
        seq_len_interpolation_factor=kwargs["seq_len_interpolation_factor"],
        rotary_cos_format=kwargs["rotary_cos_format"],
        scaling_factor=config.rotary_scaling_factor,
        original_max_position_embeddings=kwargs["original_max_position_embeddings"],
        beta_slow=config.beta_slow,
        beta_fast=config.beta_fast,
        mscale=config.mscale,
        mscale_all_dim=config.mscale_all_dim
    )


# Note: When adding a new RoPE class, add the mapping of function here
ROPE_FUNCTION = {
    'rope': _get_default,
    'llama': _get_llama3,
    'yarn': _get_yarn,
}


def get_rope(
        config: TransformerConfig,
        hidden_dim: int,
        rotary_percent: float,
        rotary_base: int,
        rotary_dtype: mstype,
        seq_len_interpolation_factor: float = None,
        position_embedding_type: str = 'rope',
        original_max_position_embeddings: int = 4096,
        rotary_cos_format: int = 2,
        **kwargs,
) -> RotaryEmbedding:
    """Obtain an instantiation object of RoPE class based on `position_embedding_type`"""
    if position_embedding_type not in ROPE_FUNCTION.keys():
        raise ValueError(f"RoPE type {position_embedding_type} is not supported")
    rotary_emb = ROPE_FUNCTION.get(position_embedding_type)(
        config=config,
        kv_channels=hidden_dim,
        rotary_percent=rotary_percent,
        rotary_base=rotary_base,
        rotary_dtype=rotary_dtype,
        seq_len_interpolation_factor=seq_len_interpolation_factor,
        original_max_position_embeddings=original_max_position_embeddings,
        rotary_cos_format=rotary_cos_format,
        **kwargs,
    )
    return rotary_emb
