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
"""Deepseek-V3 Model for training."""
from typing import Union

from mindformers.models.modeling_utils import PreTrainedModel, ModelMixin
from mindformers.parallel_core.training_graph.base_models.gpt.gpt_model import GPTModel
from mindformers.parallel_core.training_graph.tensor_parallel.layers import (
    LinearNoTP,
    ColumnParallelLinear,
    RowParallelLinear
)
from mindformers.parallel_core.training_graph.transformer.norm import FusedNorm
from mindformers.parallel_core.training_graph.transformer.transformer_block import TransformerBlockSubmodules
from mindformers.parallel_core.training_graph.transformer.flash_attention import FlashAttention
from mindformers.parallel_core.training_graph.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules
)
from mindformers.parallel_core.training_graph.transformer.mlp import MLP, MLPSubmodules
from mindformers.parallel_core.training_graph.transformer.moe.ffn import FFNGroupedGEMM
from mindformers.parallel_core.training_graph.transformer.moe.shared_experts import SharedExpertMLP
from mindformers.parallel_core.training_graph.transformer.moe.moe_layer import (
    MoELayer,
    MoESubmodules
)
from mindformers.parallel_core.training_graph.transformer.multi_token_prediction import (
    MultiTokenPredictionLayer,
    MultiTokenPredictionLayerSubmodules,
    MultiTokenPredictionBlock,
    MultiTokenPredictionBlockSubmodules
)
from mindformers.parallel_core.training_graph.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules
)
from mindformers.parallel_core.transformer_config import MLATransformerConfig
from mindformers.parallel_core.transformer_config_utils import convert_to_transformer_config
from mindformers.parallel_core.utils.model_mixin import TrainModelMixin
from mindformers.parallel_core.utils.spec_utils import ModuleSpec

from .deepseek3_config import DeepseekV3Config


# pylint: disable=W0613
def get_spec(config: MLATransformerConfig):
    """
    Get moe layer spec and dense layer spec of deepseek3.

    Args:
        config (MLATransformerConfig): Configuration object for the transformer model.


    Returns:
        tuple of moe layer spec and dense layer spec.
    """
    attention = FlashAttention
    spec = ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=FusedNorm,
            self_attention=ModuleSpec(
                module=MLASelfAttention,
                submodules=MLASelfAttentionSubmodules(
                    linear_qkv=LinearNoTP,
                    linear_qb=ColumnParallelLinear,
                    linear_kvb=ColumnParallelLinear,
                    core_attention=attention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=FusedNorm,
                    k_layernorm=FusedNorm
                ),
            ),
            pre_mlp_layernorm=FusedNorm,
            mlp=ModuleSpec(
                module=MoELayer,
                submodules=MoESubmodules(
                    experts=FFNGroupedGEMM,
                    shared_experts=ModuleSpec(
                        module=SharedExpertMLP,
                        submodules=MLPSubmodules(
                            linear_fc1=ColumnParallelLinear,
                            linear_fc2=RowParallelLinear
                        )
                    ),
                )
            ),
        )
    )

    dense_spec = ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=FusedNorm,
            self_attention=ModuleSpec(
                module=MLASelfAttention,
                submodules=MLASelfAttentionSubmodules(
                    linear_qkv=LinearNoTP,
                    linear_qb=ColumnParallelLinear,
                    linear_kvb=ColumnParallelLinear,
                    core_attention=attention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=FusedNorm,
                    k_layernorm=FusedNorm
                ),
            ),
            pre_mlp_layernorm=FusedNorm,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=ColumnParallelLinear,
                    linear_fc2=RowParallelLinear,
                )
            ),
        )
    )

    return spec, dense_spec


def get_mtp_spec(transformer_spec: Union[list[ModuleSpec], ModuleSpec], config: MLATransformerConfig):
    """
    Get Multi-Token-Prediction (MTP) block spec of deepseek3.

    Args:
        transformer_spec (Union[list[ModuleSpec], ModuleSpec]): transformer layer spec used in MTP layer, can be
            ModuleSpec, or list of ModuleSpec with length = `config.mtp_num_layers` .
        config (MLATransformerConfig): Configuration object for the transformer model.

    Returns:
        MTP block spec.
    """

    def get_mtp_layer(transformer_layer):
        return ModuleSpec(
            module=MultiTokenPredictionLayer,
            submodules=MultiTokenPredictionLayerSubmodules(
                enorm=FusedNorm,
                hnorm=FusedNorm,
                eh_proj=ColumnParallelLinear,
                transformer_layer=transformer_layer,
                layer_norm=FusedNorm
            )
        )

    if isinstance(transformer_spec, list):
        mtp_layer_spec = []
        for i in range(config.mtp_num_layers):
            mtp_layer_spec.append(get_mtp_layer(transformer_spec[i]))
    else:
        mtp_layer_spec = [get_mtp_layer(transformer_spec) for _ in range(config.mtp_num_layers)]

    mtp_block_spec = ModuleSpec(
        module=MultiTokenPredictionBlock,
        submodules=MultiTokenPredictionBlockSubmodules(
            layer_specs=mtp_layer_spec
        )
    )
    mtp_block = ModuleSpec(
        module=MultiTokenPredictionBlock,
        params={
            'spec': mtp_block_spec,
            'vocab_size': config.vocab_size
        }
    )
    return mtp_block


class DeepseekV3PreTrainedModel(PreTrainedModel, ModelMixin):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DeepseekV3Config
    base_model_prefix = "deepseekv3"


class TrainingDeepseekV3ForCausalLM(DeepseekV3PreTrainedModel, TrainModelMixin):
    """DeepseekV3 model for training"""

    def __init__(self, config: DeepseekV3Config):
        super().__init__(config, auto_prefix=False)
        self._init_gpt_model(config)

    def construct(self, *args, **kwargs):
        """DeepseekV3 construct for training"""
        return self.network(*args, **kwargs)

    def _init_gpt_model(self, config):
        """Initialize GPTModel"""
        transformer_config = convert_to_transformer_config(config, is_mla_model=True)
        spec, dense_spec = get_spec(transformer_config)
        moe_layer_freq = transformer_config.moe_layer_freq
        num_layers = transformer_config.num_layers
        spec_list = []
        for i in range(num_layers + transformer_config.mtp_num_layers):
            if i < moe_layer_freq:
                spec_list.append(dense_spec)
            else:
                spec_list.append(spec)
        transformer_block_list, mtp_block_list = spec_list[:num_layers], spec_list[num_layers:]
        transformer_block_spec = TransformerBlockSubmodules(
            layer_specs=transformer_block_list,
            layer_norm=FusedNorm,
        )
        mtp_block_spec = None
        if transformer_config.mtp_num_layers:
            mtp_block_spec = get_mtp_spec(mtp_block_list, transformer_config)

        self.network = GPTModel(
            transformer_config,
            transformer_block_spec,
            vocab_size=transformer_config.vocab_size,
            max_sequence_length=transformer_config.max_position_embeddings,
            position_embedding_type=transformer_config.position_embedding_type,
            rotary_percent=1.0,
            rotary_base=transformer_config.rotary_base,
            rope_scaling=False,
            mtp_block_spec=mtp_block_spec
        )
