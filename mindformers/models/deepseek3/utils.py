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
        weight_name = weight_name.replace('model.embed_tokens.', 'embedding.word_embeddings.')
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
        weight_name = weight_name.replace('model.norm.', 'decoder.final_layernorm.')
        weight_name = weight_name.replace('lm_head.', 'output_layer.')
        weight_name = weight_name.replace('model.layers.', 'decoder.layers.')
        if weight_name == origin_name:
            logger.warning(f"weight name '{weight_name}' does not change after conversion. "
                           f"Please check if it is as expected.")
        return weight_name

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
