# Copyright 2024 Huawei Technologies Co., Ltd
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
""" test model configs """
import unittest
from mindformers.models.auto.configuration_auto import AutoConfig
from mindformers.models.glm2 import ChatGLM2ForConditionalGeneration
from mindformers.models.llama import LlamaForCausalLM


class TestGLM2Config(unittest.TestCase):
    """test glm2 config"""
    def test_init_model_for_conditional_generation_from_yaml(self):
        """test init model with config"""
        config = AutoConfig.from_pretrained("configs/glm2/run_glm2_6b.yaml")
        config.checkpoint_name_or_path = ''
        model = ChatGLM2ForConditionalGeneration(config)


class TestLlama2Config(unittest.TestCase):
    """test llama2 config"""
    def test_init_model_for_text_generation_from_yaml(self):
        """test init model with config"""
        config = AutoConfig.from_pretrained("configs/llama2/run_llama2_7b.yaml")
        config.checkpoint_name_or_path = ''
        model = LlamaForCausalLM(config)
