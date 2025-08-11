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
"""Glm4 models' utils."""
from mindformers.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig
from mindformers.models.modeling_utils import PreTrainedModel, ModelMixin


class Glm4MoePreTrainedModel(PreTrainedModel, ModelMixin):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Glm4MoeConfig
    base_model_prefix = "Glm4Moe"

    weight_mapping = [
        ('model.embed_tokens.', 'embedding.word_embeddings.'),
        ('.self_attn.q_proj.', '.self_attention.linear_q.'),
        ('.self_attn.k_proj.', '.self_attention.linear_k.'),
        ('.self_attn.v_proj.', '.self_attention.linear_v.'),
        ('.self_attn.o_proj.', '.self_attention.linear_proj.'),
        ('.mlp.gate_proj.', '.mlp.gating.'),
        ('.mlp.down_proj.', '.mlp.linear_fc2.'),
        ('.mlp.up_proj.', '.mlp.hidden.'),
        ('.gate.weight', '.router.weight.weight'),
        ('.gate.e_score_correction_bias', '.router.expert_bias'),
        ('.mlp.shared_experts.gate_proj.', '.mlp.shared_experts.gating.'),
        ('.mlp.shared_experts.up_proj.', '.mlp.shared_experts.hidden.'),
        ('.mlp.shared_experts.down_proj.', '.mlp.shared_experts.linear_fc2.'),
        ('.gate_proj.', '.gating.'),
        ('.down_proj.', '.linear_fc2.'),
        ('.up_proj.', '.hidden.'),
        ('.post_attention_layernorm.', '.pre_mlp_layernorm.'),
        ('model.norm.', 'decoder.final_layernorm.'),
        ('lm_head.', 'output_layer.'),
        ('model.layers.', 'decoder.layers.')
    ]
