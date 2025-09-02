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
"""quantization utils."""
import os
import json
import glob
from mindformers.parallel_core.inference.quantization import (get_quantization_config,
                                                              QuantizationConfig)
from mindformers.models.configuration_utils import PretrainedConfig


def get_quant_config(model_config: PretrainedConfig, weight_mapping: list) -> QuantizationConfig:
    """method to generate QuantizationConfig."""
    model_config = model_config.to_dict()
    quantization = model_config.get("quantization", None)
    if not quantization:
        return None
    quant_cls = get_quantization_config(quantization)
    quant_config = model_config.get("quantization_config", None)
    possible_config_filenames = quant_cls.get_config_filenames()

    # If possible_config_filenames are not found, use the quantization_config
    # in model_config.
    if not possible_config_filenames:
        if quant_config:
            return quant_cls(quant_config)
        return quant_cls()

    pretrained_model_dir = model_config.get("pretrained_model_dir")
    if not os.path.isdir(pretrained_model_dir):
        raise ValueError(
            f"Cannot find the quantization config file in {pretrained_model_dir}")
    # find all possible quant_config_files
    config_files = glob.glob(os.path.join(pretrained_model_dir, "*.json"))
    quant_config_files = []
    for f in config_files:
        if any(os.path.splitext(x)[0] in f for x in possible_config_filenames):
            quant_config_files.append(f)

    if not quant_config_files:
        raise ValueError(
            f"Cannot find the config file for {quantization}")
    if len(quant_config_files) > 1:
        raise ValueError(
            f"Found multiple config files for {quantization}: "
            f"{quant_config_files}")

    quant_config_file = quant_config_files[0]
    with open(quant_config_file, encoding='utf-8') as f:
        config = json.load(f)
    if quant_config:
        quant_config.update(config)
        config = quant_config
    config["weight_mapping"] = weight_mapping
    config["quantization"] = quantization
    return quant_cls.from_config(config)
