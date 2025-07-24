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
"""ModelMixin for infer models."""
import os
import json
import re
from abc import abstractmethod
from safetensors import safe_open

from mindspore import Tensor, mutable
import mindspore.common.dtype as mstype

from mindformers.tools.logger import logger
from mindformers.models.modeling_utils import ModelMixin
from mindformers.parallel_core.inference.weights_utils import WeightsLoader


class InferModelMixin(ModelMixin):
    """
    A few utilities for `mindspore.nn.Cell`, to be used as a mixin.
    """

    @abstractmethod
    def convert_name(self, weight_name):
        pass

    def set_dynamic_inputs(self, **kwargs):
        """ dynamic shape"""
        dynamic_input_ids = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_positions = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_batch_valid_length = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_context_lens_tensor = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_q_seq_lens = Tensor(shape=[None], dtype=mstype.int32)

        dynamic_attention_mask = Tensor(shape=[None, None], dtype=self.compute_dtype)

        def get_input():
            cache_list = []
            for _ in range(self.config.num_hidden_layers):
                cache_list.append(Tensor(shape=[None, None, None, None], dtype=self.compute_dtype))
            return mutable(cache_list)

        key_cache = get_input()
        value_cache = get_input()

        self.set_inputs(dynamic_input_ids, dynamic_positions, dynamic_batch_valid_length,
                        dynamic_context_lens_tensor, dynamic_q_seq_lens, dynamic_block_tables,
                        dynamic_slot_mapping, dynamic_attention_mask, None, key_cache, value_cache)
        logger.info(f"Set dynamic input for {self.__class__.__name__}")

    def add_flags_custom_mcore(self, is_prefill):
        r"""
        Add flag to distinguish fa and pa.

        Args:
            is_prefill: flag to distinguish fa and pa.

        Returns:

        """
        self.add_flags(is_prefill=is_prefill)
        self.model.add_flags(is_prefill=is_prefill)
        self.model.decoder.add_flags(is_prefill=is_prefill)
        self.model.casual_mask.add_flags(is_prefill=is_prefill)
        for layer in self.model.decoder.layers:
            if self.config.use_flash_attention:
                layer.self_attention.core_attention.add_flags(is_prefill=is_prefill)

    def convert_net_name(self, mf_name, config):
        r"""
        Convert Mindformers weight name to network name.

        Args:
            mf_name (str): Mindformers weight name.
            config (object): Configuration object containing q_lora_rank.

        Returns:
            str: Converted network name.
        """
        net_name = mf_name
        replacements = [
            (r'\.self_attention\.linear_[qkv]\.', '.self_attention.linear_qkv.'),
            (r'\.mlp\.gating\.', '.mlp.linear_fc1.'),
            (r'\.experts\.\d+\.gating\.weight', '.experts.weight1'),
            (r'\.experts\.\d+\.linear_fc1\.weight', '.experts.weight1'),
            (r'\.experts\.\d+\.linear_fc2\.weight', '.experts.weight2'),
            (r'\.shared_experts\.gating\.', '.shared_experts.linear_fc1.')
        ]

        for pattern, replacement in replacements:
            net_name = re.sub(pattern, replacement, net_name)

        if hasattr(config, 'q_lora_rank') and config.q_lora_rank is not None:
            net_name = re.sub(r'\.self_attention\.linear_(q|kv)_down_proj\.',
                              '.self_attention.linear_qkv_down_proj.', net_name)
        else:
            net_name = net_name.replace('.self_attention.linear_q_down_proj.',
                                        '.self_attention.linear_q_proj.')

        return net_name

    def load_weights(self, weights_path):
        r"""
        Load weights.

        Args:
            weights_path: The path of weights.

        """
        weights_loader = WeightsLoader(weights_path)
        param_json_path = ""
        for file in os.listdir(weights_path):
            if file.endswith('index.json'):
                param_json_path = os.path.join(weights_path, file)
            elif file.endswith('param_name_map.json'):
                param_json_path = os.path.join(weights_path, file)

        weight_map = {}
        if os.path.exists(param_json_path):
            with open(param_json_path, "r") as fp:
                data = json.load(fp)
                weight_map = data.get("weight_map", data)
        else:
            # only one safetensors
            safetensors_count = sum(
                1 for file in os.listdir(weights_path)
                if file.endswith(".safetensors")
            )
            if safetensors_count != 1:
                raise ValueError(f"There should be only one `.safetensors` file {weights_path}, "
                                 f"but {safetensors_count} `.safetensors` files were unexpectedly found.")
            safetensor_file = "model.safetensors"
            with safe_open(f"{weights_path}/{safetensor_file}",
                           framework="np") as sf_file:
                all_keys = sf_file.keys()
                for key in all_keys:
                    weight_map[str(key).strip()] = safetensor_file

        for weight_name, weight_file in weight_map.items():
            if self.convert_name is not None:
                mf_name = self.convert_name(weight_name)
                net_name = self.convert_net_name(mf_name, self.config)
                weights_loader.mapping_dict.update({weight_name: (net_name, weight_file)})
                mf_name = mf_name.split('.')[-2]
                if mf_name not in weights_loader.mf_hf_mapping.keys():
                    weights_loader.mf_hf_mapping[mf_name] = weight_name.split('.')[-2]
        self.model.load_weights(weights_loader)
