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
"""test code alpaca handler"""
import os
import unittest
import tempfile
import pytest
import mindspore
import mindformers
from mindformers import MindFormerConfig
from mindformers.dataset.dataloader.common_dataloader import CommonDataLoader
from mindformers.dataset.handler.codealpaca_handler import CodeAlpacaInstructDataHandler

from tests.st.test_ut.test_dataset.get_test_data import get_alpaca_data
from tests.st.test_ut.test_tokenizers.get_vocab_model import get_sp_vocab_model


class TestLlavaHandler(unittest.TestCase):
    """A test class for testing alpaca handler"""

    @classmethod
    def setUpClass(cls):
        os.environ['USE_OM'] = "OFF"
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.path = cls.temp_dir.name
        get_alpaca_data(cls.path)
        get_sp_vocab_model("llama", cls.path)
        cls.handler_config = MindFormerConfig(
            **{
                'type': "CodeAlpacaInstructDataHandler",
                "tokenizer": {"type": "LlamaTokenizer", "vocab_file": os.path.join(cls.path, "llama_tokenizer.model")},
                "seq_length": 32,
                "prompt_key": "conversations",
                "output_columns": ["input_ids", "labels"]
            }
        )
        cls.handler = CodeAlpacaInstructDataHandler(cls.handler_config)
        cls.dataset = CommonDataLoader(path=cls.path, shuffle=False, split="train",
                                       input_columns=["input_ids", "labels"], handler=cls.handler_config)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_default(self):
        """test default logic"""
        assert isinstance(self.dataset, mindspore.dataset.GeneratorDataset)
        assert isinstance(self.handler, mindformers.dataset.handler.CodeAlpacaInstructDataHandler)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_other(self):
        """test other logic"""
        for item in self.dataset:
            assert len(item) == 2
            assert item[0].asnumpy().tolist() == [1, 134, 0, 142, 164, 134, 159, 7, 134, 159, 137, 111, 0, 134, 0, 143,
                                                  142, 161, 143, 159, 0, 46, 134, 17, 107, 140, 144, 159, 137, 144, 135,
                                                  134, 164]
            assert item[1].asnumpy().tolist() == [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                                                  -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                                                  -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]
