# Copyright 2023 Huawei Technologies Co., Ltd
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
"""test PretrainedConfig"""
import os
import tempfile
import unittest
from mindformers import TransformerOpParallelConfig, MoEConfig
from mindformers.models import GPT2Config

class TestPretrainedConfig(unittest.TestCase):
    """test PretrainedConfig"""
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = self.temp_dir.name
        self.config_path = self.path+"/config.json"

    def test_save_and_read(self):
        """test save and read config.json"""
        config1 = GPT2Config()

        config1.save_pretrained(self.path)

        self.assertEqual(os.path.isfile(self.config_path), True)

        config2 = GPT2Config.from_pretrained(self.config_path)
        config3 = GPT2Config.from_pretrained(self.path)
        config4 = GPT2Config.from_json_file(self.config_path)

        self.assertEqual(config2, config3)
        self.assertEqual(config2, config4)

    def test_save_and_read_2(self):
        """test save and read config.json with different config"""
        config1 = GPT2Config(
            parallel_config=TransformerOpParallelConfig(data_parallel=2, recompute={"recompute": True}),
            moe_config=MoEConfig(expert_num=2),
            batch_size=2
        )

        config1.save_pretrained(self.path)

        self.assertEqual(os.path.isfile(self.config_path), True)

        config2 = GPT2Config.from_pretrained(self.config_path)
        config3 = GPT2Config.from_pretrained(self.path)
        config4 = GPT2Config.from_json_file(self.config_path)

        self.assertEqual(config2, config3)
        self.assertEqual(config2, config4)
        self.assertEqual(config2.parallel_config,
                         TransformerOpParallelConfig(data_parallel=2, recompute={"recompute": True}))
        self.assertEqual(config2.moe_config, MoEConfig(expert_num=2))
