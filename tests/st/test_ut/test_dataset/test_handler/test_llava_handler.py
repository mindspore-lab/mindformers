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
"""test llava handler"""
import os
import unittest
import tempfile
import pytest
import mindspore
import mindformers
from mindformers import MindFormerConfig
from mindformers.dataset.dataloader.common_dataloader import CommonDataLoader
from mindformers.dataset.handler.llava_handler import LlavaInstructDataHandler

from tests.st.test_ut.test_dataset.get_test_data import get_llava_data


class TestLlavaHandler(unittest.TestCase):
    """A test class for testing llava handler"""

    @classmethod
    def setUpClass(cls):
        os.environ['USE_OM'] = "OFF"
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.path = cls.temp_dir.name
        get_llava_data(cls.path)
        handler_config = MindFormerConfig(
            **{'type': "LlavaInstructDataHandler", "image_dir": cls.path, "output_columns": ["conversations"]}
        )
        cls.handler = LlavaInstructDataHandler(handler_config)
        cls.dataset = CommonDataLoader(path=os.path.join(cls.path, "text"), shuffle=False,
                                       input_columns=["conversations"], handler=handler_config)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_default(self):
        """test default logic"""
        assert isinstance(self.dataset, mindspore.dataset.GeneratorDataset)
        assert isinstance(self.handler, mindformers.dataset.handler.LlavaInstructDataHandler)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_gen_prompt(self):
        """test gen_prompt function in base class"""
        res = self.handler.gen_prompt([{"from": "user", "value": "mock"}])
        assert res == "\n### Instruction:\nmock\n"
        res = self.handler.gen_prompt([{"from": "mock", "value": "mock"}])
        assert res == "\n### Response:\nmock\n"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_other(self):
        """test other logic"""
        assert self.handler.format_func("mock_input") is None
