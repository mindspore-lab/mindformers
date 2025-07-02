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
"""Qwen2 models' utils."""
from mindformers.tools.logger import logger
from mindformers.models.qwen2.configuration_qwen2 import Qwen2Config
from mindformers.models.modeling_utils import PreTrainedModel, ModelMixin


class Qwen2PreTrainedModel(PreTrainedModel, ModelMixin):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Qwen2Config
    base_model_prefix = "Qwen2"

    def convert_name(self, weight_name):
        r"""
        convert HuggingFace weight name to MindFormers weight name.

        Args:
            weight_name: huggingface weight names.

        Returns:
            weight_name: converted weight names.

        """
        origin_name = weight_name
        weight_name = weight_name.replace('model.embed_tokens.', 'embedding.word_embeddings.')
        weight_name = weight_name.replace('.self_attn.q_proj.', '.self_attention.linear_q.')
        weight_name = weight_name.replace('.self_attn.k_proj.', '.self_attention.linear_k.')
        weight_name = weight_name.replace('.self_attn.v_proj.', '.self_attention.linear_v.')
        weight_name = weight_name.replace('.self_attn.o_proj.', '.self_attention.linear_proj.')
        weight_name = weight_name.replace('.mlp.gate_proj.', '.mlp.gating.')
        weight_name = weight_name.replace('.mlp.down_proj.', '.mlp.linear_fc2.')
        weight_name = weight_name.replace('.mlp.up_proj.', '.mlp.linear_fc1.')
        weight_name = weight_name.replace('.post_attention_layernorm.', '.pre_mlp_layernorm.')
        weight_name = weight_name.replace('model.norm.', 'decoder.final_layernorm.')
        weight_name = weight_name.replace('lm_head.', 'output_layer.')
        weight_name = weight_name.replace('model.layers.', 'decoder.layers.')
        if weight_name == origin_name:
            logger.warning(f"weight name '{weight_name}' does not change after conversion. "
                           f"Please check if it is as expected.")
        return weight_name
