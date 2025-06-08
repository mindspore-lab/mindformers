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
"""Qwen3 models' utils."""
import os
import json
from safetensors.numpy import load_file

from mindformers.tools.logger import logger
from mindformers.models.qwen3.configuration_qwen3 import Qwen3Config
from mindformers.models.modeling_utils import PreTrainedModel, ModelMixin


class Qwen3PreTrainedModel(PreTrainedModel, ModelMixin):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Qwen3Config
    base_model_prefix = "Qwen3"


    def convert_name(self, weight_name):
        r"""
        convert HuggingFace weight name to MindFormers weight name.

        Args:
            weight_name: huggingface weight names.

        Returns:
            weight_name: converted weight names.

        """
        origin_name = weight_name
        weight_name = weight_name.replace('embed_tokens.', 'embedding.word_embeddings.')
        weight_name = weight_name.replace('.self_attn.q_proj.', '.self_attention.linear_q.')
        weight_name = weight_name.replace('.self_attn.k_proj.', '.self_attention.linear_k.')
        weight_name = weight_name.replace('.self_attn.v_proj.', '.self_attention.linear_v.')
        weight_name = weight_name.replace('.self_attn.o_proj.', '.self_attention.linear_proj.')
        weight_name = weight_name.replace('.self_attn.q_norm.', '.self_attention.q_layernorm.')
        weight_name = weight_name.replace('.self_attn.k_norm.', '.self_attention.k_layernorm.')
        weight_name = weight_name.replace('.mlp.gate_proj.', '.mlp.gating.')
        weight_name = weight_name.replace('.mlp.down_proj.', '.mlp.linear_fc2.')
        weight_name = weight_name.replace('.mlp.up_proj.', '.mlp.linear_fc1.')
        weight_name = weight_name.replace('.post_attention_layernorm.', '.pre_mlp_layernorm.')
        weight_name = weight_name.replace('.norm.', '.decoder.final_norm.')
        weight_name = weight_name.replace('lm_head.', 'output_layer.')
        weight_name = weight_name.replace('.layers.', '.decoder.layers.')
        weight_name = weight_name.replace('model.', '')
        if weight_name == origin_name:
            logger.warning(f"weight name '{weight_name}' does not change after conversion. "
                           f"Please check if it is as expected.")
        return weight_name

    def check_key_mapping(self):
        r"""
        Store the weight keys of huggingface and mindformers.

        Returns:
            convert_key_map: store the weight keys of map.

        """
        convert_key_mapping = {
            "embed_tokens",
            "word_embeddings",
            "k_norm",
            "k_layernorm",
            "q_norm",
            "q_layernorm",
            "q_proj",
            "linear_q",
            "k_proj",
            "linear_k",
            "v_proj",
            "linear_v",
            "o_proj",
            "linear_proj",
            "gate_proj",
            "gating",
            "down_proj",
            "linear_fc2",
            "up_proj",
            "linear_fc1",
            "post_attention_layernorm",
            "pre_mlp_layernorm",
            "norm",
            "final_norm",
            "lm_head",
            "output_layer",
            "input_layernorm",
            "q_norm",
            "q_layernorm",
            "k_norm",
            "k_layernorm"
        }
        return convert_key_mapping

    def convert_hf_weight_to_mf(self, load_checkpoint):
        r"""
        Read and store weights.

        Args:
            load_checkpoint: the path of weights.

        Returns:
            non_layer_weights: Weights other than Transformer Layer
            layer_weights: Weights of Transformer Layer
        """
        sf_files = [f for f in os.listdir(load_checkpoint) if f.endswith(".safetensors")]
        non_layer_weights = {}
        layer_weights = {}
        if len(sf_files) > 1:
            json_files = [f for f in os.listdir(load_checkpoint)
                          if f.endswith('index.json') or f.endswith('param_name_map.json')]
            if not json_files:
                raise ValueError(f"No index.json or param_name_map.json found in {load_checkpoint}")
            param_json_path = os.path.join(load_checkpoint, json_files[0])
            with open(param_json_path, "r") as fp:
                data = json.load(fp)
            weight_map = data.get("weight_map", data)
        elif sf_files:
            weight_map = load_file(os.path.join(load_checkpoint, sf_files[0]))
        else:
            raise ValueError(f"No .safetensors files found in {load_checkpoint}")
        for key, value in weight_map.items():
            if key.startswith("model.layers") or key.startswith("model.decoder.layers"):
                layer_weights[key] = value
            else:
                non_layer_weights[key] = value
        return non_layer_weights, layer_weights
