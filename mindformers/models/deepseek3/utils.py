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
"""DeepSeek3 models' utils."""
import os
import json
from safetensors.numpy import load_file

import mindspore as ms

from mindformers.tools.logger import logger
from mindformers.models.modeling_utils import PreTrainedModel, ModelMixin
from mindformers.models.deepseek3.configuration_deepseek_v3 import DeepseekV3Config


class DeepseekV3PreTrainedModel(PreTrainedModel, ModelMixin):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DeepseekV3Config
    base_model_prefix = "Deepseekv3"

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
        weight_name = weight_name.replace('.self_attn.q_a_proj.', '.self_attention.linear_q_down_proj.')
        weight_name = weight_name.replace('.self_attn.q_a_layernorm.', '.self_attention.q_layernorm.')
        weight_name = weight_name.replace('.self_attn.q_b_proj.', '.self_attention.linear_q_up_proj.')
        weight_name = weight_name.replace('.self_attn.kv_a_proj_with_mqa.', '.self_attention.linear_kv_down_proj.')
        weight_name = weight_name.replace('.self_attn.kv_a_layernorm.', '.self_attention.kv_layernorm.')
        weight_name = weight_name.replace('.self_attn.kv_b_proj.', '.self_attention.linear_kv_up_proj.')
        weight_name = weight_name.replace('.self_attn.o_proj.', '.self_attention.linear_proj.')
        weight_name = weight_name.replace('.mlp.gate_proj.', '.mlp.gating.')
        weight_name = weight_name.replace('.mlp.down_proj.', '.mlp.linear_fc2.')
        weight_name = weight_name.replace('.mlp.up_proj.', '.mlp.linear_fc1.')
        weight_name = weight_name.replace('.gate.weight', '.router.weight.weight')
        weight_name = weight_name.replace('.gate.e_score_correction_bias', '.router.expert_bias')
        weight_name = weight_name.replace('.shared_experts.gate_proj.', '.shared_experts.gating.')
        weight_name = weight_name.replace('.shared_experts.up_proj.', '.shared_experts.linear_fc1.')
        weight_name = weight_name.replace('.shared_experts.down_proj.', '.shared_experts.linear_fc2.')
        weight_name = weight_name.replace('.gate_proj.', '.gating.')
        weight_name = weight_name.replace('.down_proj.', '.linear_fc2.')
        weight_name = weight_name.replace('.up_proj.', '.linear_fc1.')

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
            "q_a_proj",
            "q_a_layernorm",
            "q_b_proj",
            "linear_q_up_proj",
            "kv_a_proj_with_mqa",
            "linear_kv_down_proj",
            "kv_a_layernorm",
            "kv_layernorm",
            "kv_b_proj",
            "linear_kv_up_proj",
            "o_proj",
            "linear_proj",
            "gate_proj",
            "gating",
            "up_proj",
            "linear_fc1",
            "down_proj",
            "linear_fc2",
            "post_attention_layernorm",
            "pre_mlp_layernorm",
            "norm",
            "final_norm",
            "lm_head",
            "output_layer",
            "input_layernorm",
            "gate"
        }
        return convert_key_mapping

    def convert_hf_weight_to_mf(self, weights_path):
        r"""
        Read and store weights.

        Args:
            weights_path: the path of weights.

        Returns:
            non_layer_weights: Weights other than Transformer Layer
            layer_weights: Weights of Transformer Layer
        """
        sf_files = [f for f in os.listdir(weights_path) if f.endswith(".safetensors")]
        non_layer_weights = {}
        layer_weights = {}
        if len(sf_files) > 1:
            json_files = []
            for f in os.listdir(weights_path):
                if f.endswith('index.json'):
                    json_files.append(f)
                elif f.endswith('param_name_map.json'):
                    json_files.append(f)
            if not json_files:
                raise ValueError(f"No index.json or param_name_map.json found in {weights_path}")
            param_json_path = os.path.join(weights_path, json_files[0])
            with open(param_json_path, "r") as fp:
                data = json.load(fp)
            weight_map = data.get("weight_map", data)
        elif sf_files:
            weight_map = load_file(os.path.join(weights_path, sf_files[0]))
        else:
            raise ValueError(f"No .safetensors files found in {weights_path}")
        for key, value in weight_map.items():
            if key.startswith("model.layers") or key.startswith("model.decoder.layers"):
                layer_weights[key] = value
            else:
                non_layer_weights[key] = value
        return non_layer_weights, layer_weights

    def check_pipeline_stage(self):
        """check pipeline_stage and num_layers"""
        config = self.config
        parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        pp = config.parallel_config.pipeline_stage
        if parallel_mode in ["semi_auto_parallel"]:
            num_hidden_layers = config.num_hidden_layers
            num_nextn_predict_layers = config.num_nextn_predict_layers
            if num_hidden_layers and num_hidden_layers + num_nextn_predict_layers < pp:
                raise ValueError(
                    f"num_hidden_layers + num_nextn_predict_layers of model should be greater than or equal to "
                    f"pipeline_stage, but get num_hidden_layers ({num_hidden_layers})"
                    f" + num_nextn_predict_layers ({num_nextn_predict_layers}) < pp ({pp})"
                )
            pipeline_interleave_enabled = ms.get_auto_parallel_context("pipeline_interleave")
            pp_interleave_num = getattr(config, 'pp_interleave_num', 0) or 0
            if pipeline_interleave_enabled and pp_interleave_num * pp > num_hidden_layers + num_nextn_predict_layers:
                raise ValueError(
                    f"num_hidden_layers + num_nextn_predict_layers of model should be greater than "
                    f"`pp * pp_interleave_num`, but got num_hidden_layers + num_nextn_predict_layers : "
                    f"{num_hidden_layers} + {num_nextn_predict_layers} "
                    f"and pp * pp_interleave_num = {pp * pp_interleave_num}."
                )
