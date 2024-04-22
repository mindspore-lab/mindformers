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
"""test AutoConfig"""
import os
os.environ["OPENMIND_HUB_ENDPOINT"] = "https://openmind.test.osinfra.cn"

import tempfile
import unittest
from mindformers import LlamaConfig
from mindformers.models.auto import AutoConfig
from mindformers.models import GPT2Config

REPO_ID = "mindformersinfra/test_auto_config_ms"
NUM_LAYERS = 4

DYNAMIC_REPO_ID = "mindformersinfra/test_dynamic_config"
DYNAMIC_CLASS_NAME = "Baichuan2Config"

MODEL_TYPE = "gpt2"


class TestAutoConfig(unittest.TestCase):
    """test AutoConfig"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = self.temp_dir.name
        self.config_path = self.path + "/config.json"

    def test_download_from_repo(self):
        """Test init config by from_pretrained('usr_name/repo_name')"""
        config = AutoConfig.from_pretrained(REPO_ID)
        self.assertEqual(config.num_layers, NUM_LAYERS)
        self.assertTrue(isinstance(config, GPT2Config))

    def test_save_and_read_with_path(self):
        """Test save config by save_pretrained() and read it by from_pretrained()"""
        config = GPT2Config(num_layers=NUM_LAYERS)

        config.save_pretrained(self.path, save_json=True)

        config = AutoConfig.from_pretrained(self.path)
        self.assertEqual(config.num_layers, NUM_LAYERS)
        self.assertTrue(isinstance(config, GPT2Config))

    def test_save_and_read_with_json_file_path(self):
        """Test save config by save_pretrained() and read it by from_pretrained()"""
        config = GPT2Config(num_layers=NUM_LAYERS)

        config.save_pretrained(self.path, save_json=True)

        config = AutoConfig.from_pretrained(self.config_path)
        print(type(config))
        self.assertEqual(config.num_layers, NUM_LAYERS)
        self.assertTrue(isinstance(config, GPT2Config))

    def test_dynamic_config(self):
        """Test init dynamic config by from_pretrained('usr_name/repo_name')"""
        model = AutoConfig.from_pretrained(DYNAMIC_REPO_ID, trust_remote_code=True)
        self.assertEqual(model.__class__.__name__, DYNAMIC_CLASS_NAME)

    def test_load_with_model_name(self):
        """Test load config by from_pretrained() with model name"""
        config = AutoConfig.from_pretrained("llama_7b")
        self.assertTrue(isinstance(config, LlamaConfig))

    def test_load_local_yaml(self):
        """Test load config by from_pretrained() with local yaml file path"""
        config = AutoConfig.from_pretrained("configs/llama/run_llama_7b.yaml")
        self.assertTrue(isinstance(config, LlamaConfig))

    def test_for_model(self):
        """Test for_model"""
        config = AutoConfig.for_model(MODEL_TYPE)
        self.assertTrue(isinstance(config, GPT2Config))
