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
"""test configuration_utils"""
import os
os.environ["OPENMIND_HUB_ENDPOINT"] = "https://openmind.test.osinfra.cn"

import tempfile
import re
import unittest
from mindformers import LlamaConfig, MindFormerConfig, Blip2Config, ViTConfig
from mindformers.models.blip2.qformer_config import QFormerConfig
from mindformers.models.utils import CONFIG_NAME

RMS_NORM_EPS = 1.0e-6
BATCH_SIZE = 4
REPO_ID = "mindformersinfra/test_llama"

class TestLlamaConfig(unittest.TestCase):
    """test llama config"""
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = self.temp_dir.name

    def test_init_from_yaml(self):
        config = LlamaConfig.from_pretrained("configs/llama/run_llama_7b.yaml")
        self.assertTrue(isinstance(config, LlamaConfig))
        self.assertEqual(config.rms_norm_eps, RMS_NORM_EPS)

    def test_init_from_name(self):
        config = LlamaConfig.from_pretrained("llama_7b")
        self.assertTrue(isinstance(config, LlamaConfig))
        self.assertEqual(config.rms_norm_eps, RMS_NORM_EPS)
        self.assertTrue(os.path.exists("./checkpoint_download/llama/llama_7b.yaml"))

    def test_init_from_wrong_name(self):
        try:
            config = LlamaConfig.from_pretrained("xxxx")
        except ValueError as e:
            self.assertTrue(re.match(r"xxxx is not a supported model.*", str(e)))
        try:
            config = LlamaConfig.from_pretrained("mindspore/xxxx")
        except ValueError as e:
            self.assertTrue(re.match(r"mindspore/xxxx is not a supported model.*", str(e)))

    def test_init_from_remote(self):
        config = LlamaConfig.from_pretrained(REPO_ID)
        self.assertTrue(isinstance(config, LlamaConfig))
        self.assertEqual(config.rms_norm_eps, RMS_NORM_EPS)

    def test_init_from_remote_wrong(self):
        try:
            config = LlamaConfig.from_pretrained("repo/xxxx")
        except RuntimeError as e:
            error_message = r"Error occurred when executing function get_config_experimental_mode.*"
            self.assertTrue(re.match(error_message, str(e)))

    def test_init_from_dir_or_json(self):
        """test_init_from_dir_or_json"""
        config = LlamaConfig(batch_size=BATCH_SIZE)
        config.save_pretrained(self.path, save_json=True)

        new_config = LlamaConfig.from_pretrained(self.path)
        self.assertTrue(isinstance(new_config, LlamaConfig))
        self.assertEqual(new_config.batch_size, BATCH_SIZE)

        json_path = self.path + "/" + CONFIG_NAME
        new_config = LlamaConfig.from_pretrained(json_path)
        self.assertTrue(isinstance(new_config, LlamaConfig))
        self.assertEqual(new_config.batch_size, BATCH_SIZE)

    def test_save_pretrained(self):
        config = LlamaConfig(batch_size=BATCH_SIZE)
        config.save_pretrained(self.path, save_name="mindspore_model")
        yaml_path = self.path + "/" + "mindspore_model.yaml"
        self.assertTrue(os.path.exists(yaml_path))
        mf_config = MindFormerConfig(yaml_path)
        self.assertTrue(mf_config.model.model_config.batch_size, BATCH_SIZE)


class TestBlip2Config(unittest.TestCase):
    """test blip2 config"""
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = self.temp_dir.name

    def test_init_from_yaml(self):
        config = Blip2Config.from_pretrained("configs/blip2/run_blip2_stage2_vit_g_llama_7b.yaml")
        self.assertTrue(isinstance(config, Blip2Config))
        self.assertTrue(isinstance(config.vision_config, ViTConfig))
        self.assertTrue(isinstance(config.qformer_config, QFormerConfig))
        self.assertTrue(isinstance(config.text_config, LlamaConfig))

    def test_init_from_name(self):
        config = Blip2Config.from_pretrained("blip2_stage2_vit_g_llama_7b")
        self.assertTrue(isinstance(config, Blip2Config))
        self.assertTrue(isinstance(config.vision_config, ViTConfig))
        self.assertTrue(isinstance(config.qformer_config, QFormerConfig))
        self.assertTrue(isinstance(config.text_config, LlamaConfig))

    def test_save_pretrained(self):
        """test_save_pretrained"""
        config = Blip2Config.from_pretrained("blip2_stage2_vit_g_llama_7b")
        config.save_pretrained(self.path, save_name="mindspore_model")
        yaml_path = self.path + "/" + "mindspore_model.yaml"
        self.assertTrue(os.path.exists(yaml_path))
        mf_config = MindFormerConfig(yaml_path)
        self.assertEqual(mf_config.model.arch.type, "Blip2Llm")
        self.assertEqual(mf_config.model.model_config.type, "Blip2Config")
        self.assertEqual(mf_config.model.model_config.vision_config.type, "ViTConfig")
        self.assertEqual(mf_config.model.model_config.text_config.type, "LlamaConfig")
        self.assertEqual(mf_config.model.model_config.qformer_config.vocab_size, 44728)
