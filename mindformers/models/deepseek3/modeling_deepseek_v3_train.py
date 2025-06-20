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
from mindformers.models.deepseek3.utils import DeepseekV3PreTrainedModel
from mindformers.parallel_core.training_graph.base_models.gpt.gpt_model import GPTModel
from mindformers.parallel_core.training_graph.base_models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec, \
    get_gpt_mtp_block_spec, get_gpt_layer_local_spec
from mindformers.parallel_core.transformer_config_utils import convert_to_transformer_config
from mindformers.parallel_core.utils.model_mixin import TrainModelMixin

from .configuration_deepseek_v3 import DeepseekV3Config


class TrainingDeepseekV3ForCausalLM(DeepseekV3PreTrainedModel, TrainModelMixin):
    """DeepseekV3 model for training"""

    def __init__(self, config: DeepseekV3Config):
        super().__init__(config, auto_prefix=False)
        transformer_config = convert_to_transformer_config(config, is_mla_model=True)
        if transformer_config.num_moe_experts:
            transformer_layer_spec = get_gpt_decoder_block_spec(transformer_config)
        else:
            transformer_layer_spec = get_gpt_layer_local_spec(
                qk_layernorm=transformer_config.qk_layernorm,
                multi_latent_attention=transformer_config.multi_latent_attention,
                use_contiguous_weight_layout=transformer_config.use_contiguous_weight_layout,
                mla_qkv_concat=transformer_config.mla_qkv_concat
            )
        mtp_block_spec = None
        if transformer_config.mtp_num_layers is not None:
            mtp_block_spec = get_gpt_mtp_block_spec(transformer_config, transformer_layer_spec)
        self.network = GPTModel(
            transformer_config,
            transformer_layer_spec,
            vocab_size=transformer_config.vocab_size,
            max_sequence_length=transformer_config.max_position_embeddings,
            position_embedding_type=transformer_config.position_embedding_type,
            rotary_percent=1.0,
            rotary_base=transformer_config.rotary_base,
            rope_scaling=False,
            mtp_block_spec=mtp_block_spec
        )

    def construct(self, *args, **kwargs):
        """DeepseekV3 construct for training"""
        return self.network(*args, **kwargs)

    @classmethod
    def can_generate(cls):
        return False
