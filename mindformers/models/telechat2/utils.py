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
"""Telechat2 models' utils."""
from mindformers.models.telechat2.configuration_telechat2 import Telechat2Config
from mindformers.models.modeling_utils import PreTrainedModel, ModelMixin


class Telechat2PreTrainedModel(PreTrainedModel, ModelMixin):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Telechat2Config
    base_model_prefix = "Telechat2"

    weight_mapping = {
        ('transformer.word_embeddings.', 'embedding.word_embeddings.'),
        ('.self_attention.query.', '.self_attention.linear_q.'),
        ('.self_attention.key.', '.self_attention.linear_k.'),
        ('.self_attention.value.', '.self_attention.linear_v.'),
        ('.self_attention.dense.', '.self_attention.linear_proj.'),
        ('.mlp.gate_proj.', '.mlp.gating.'),
        ('.mlp.down_proj.', '.mlp.linear_fc2.'),
        ('.mlp.up_proj.', '.mlp.hidden.'),
        ('.gate.weight', '.router.weight.weight'),
        ('.gate_proj.', '.gating.'),
        ('.down_proj.', '.linear_fc2.'),
        ('.up_proj.', '.hidden.'),
        ('.post_attention_layernorm.', '.pre_mlp_layernorm.'),
        ('transformer.ln_f.', 'decoder.final_layernorm.'),
        ('lm_head.', 'output_layer.'),
        ('transformer.h.', 'decoder.layers.')
    }
