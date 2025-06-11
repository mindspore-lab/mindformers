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
"""Deepseek3 models' APIs."""
import gc
import os
from safetensors import safe_open
from tqdm import tqdm

import mindspore as ms

from mindspore.communication.management import get_rank

from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from mindformers.tools.logger import logger
from mindformers.models.deepseek3.utils import Deepseek3PreTrainedModel


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class InferenceDeepseek3ForCausalLM(Deepseek3PreTrainedModel):
    r"""
    Provide Deepseek3 model infer through network.

    Args:
        config (Deepseek3Config): The config of deepseek3 model.

    Returns:
        output: Tensor, the output of qwen3 deepseek3 layer

    """

    def load_weights(self, weights_path):
        r"""
        Load weights.

        Args:
            weights_path: The path of storing weights.

        """
        rank_id = get_rank()

        source_qkv_concat = False

        sf_files = [f for f in os.listdir(weights_path) if f.endswith(".safetensors")]
        keys = []
        if sf_files:
            with safe_open(os.path.join(weights_path, sf_files[0]), framework="np") as f:
                keys = f.keys()
        for key in keys:
            if key.split('.')[-2] not in self.check_key_mapping():
                raise ValueError(f'Please enter the correct weights of safetensors')
            if key.split('.')[-2] == 'linear_qkv':
                source_qkv_concat = True
                break

        non_layer_weights, layer_weights = (self.convert_hf_weight_to_mf(weights_path))

        mf_hf_map = {}
        for weight_name in list(non_layer_weights.keys()):
            value = non_layer_weights.pop(weight_name)
            new_name = self.convert_name(weight_name)
            non_layer_weights[new_name] = value
            mf_hf_map[new_name] = weight_name
        parameter_dict = self.model.load_weights(weights_path, non_layer_weights, mf_hf_map, source_qkv_concat)
        ms.load_param_into_net(self, parameter_dict)
        del parameter_dict
        del mf_hf_map
        gc.collect()
        logger.info('................weights loading complete except the transformer layers weights................')

        num_layers = self.config.num_hidden_layers
        enable_tqdm = rank_id == 0
        with tqdm(range(num_layers), desc="Weight loading", disable=not enable_tqdm) as pbar:
            for layer_id in pbar:
                layer_weight = {}
                mf_hf_map = {}
                prefix = f"model.layers.{layer_id}."
                train_prefix = f"decoder.layers.{layer_id}."
                for weight_name in list(layer_weights.keys()):
                    if weight_name.startswith(prefix) or weight_name.startswith(train_prefix):
                        value = layer_weights.pop(weight_name)
                        new_name = self.convert_name(weight_name)
                        layer_weight[new_name] = value
                        mf_hf_map[new_name] = weight_name
                parameter_dict = self.model.load_weights(
                    weights_path, layer_weight, mf_hf_map, source_qkv_concat, layer_id)
                ms.load_param_into_net(self.model.decoder.layers[layer_id], parameter_dict)
                del parameter_dict
                gc.collect()
                pbar.set_postfix({"current_layer": layer_id})
