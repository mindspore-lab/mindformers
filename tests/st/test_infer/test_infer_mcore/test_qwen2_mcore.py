# Copyright 2022 Huawei Technologies Co., Ltd
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
"""test qwen2 load hf_config."""
import os
import json

from mindformers.tools.register import MindFormerConfig
from mindformers.core.context import build_context
from mindformers.experimental.models.qwen2.configuration_qwen2 import Qwen2Config


def test_qwen2_trainer_mcore():
    """
    Feature: qwen2 load hf
    Description: Test the qwen2 load hf_config
    Expectation: No exception
    """
    local_path = os.getcwd()
    json_file = os.path.join(local_path, "config.json")
    yaml_path = os.path.join(
        local_path,
        "predict_qwen2_0.5b_instruct_HF.yaml"
    )
    two_levels_up = os.path.abspath(os.path.join(os.getcwd(), "../../../.."))
    register_path = os.path.join(two_levels_up, "mindformers/experimental/models/qwen2")
    os.environ["REGISTER_PATH"] = register_path
    config = MindFormerConfig(yaml_path)
    build_context(config)
    model_config = Qwen2Config.from_pretrained(yaml_path)
    with open(json_file, "r", encoding="utf-8") as f:
        hf_dict = json.load(f)
    for k, v in hf_dict.items():
        if k == "model_type":
            v = "Qwen2"
        assert v == getattr(model_config, k)
