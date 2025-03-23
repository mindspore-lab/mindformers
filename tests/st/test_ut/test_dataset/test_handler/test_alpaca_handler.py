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
"""test alpaca handler"""
import os
import unittest
import tempfile
import pytest
import mindspore
import mindformers
from mindformers import MindFormerConfig
from mindformers.dataset.handler.base_handler import BaseTemplate
from mindformers.dataset.dataloader.common_dataloader import CommonDataLoader
from mindformers.dataset.handler.alpaca_handler import AlpacaInstructDataHandler

from tests.st.test_ut.test_dataset.get_test_data import get_alpaca_data
from tests.st.test_ut.test_tokenizers.get_vocab_model import get_sp_vocab_model


class MockAlpacaTemplate(BaseTemplate):
    """Alpaca Conv Template."""
    end_token = "\n"
    input_token = "### Input:"
    system = (
        "A mock template"
    )


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
                'type': "AlpacaInstructDataHandler",
                "tokenizer": {"type": "LlamaTokenizer", "vocab_file": os.path.join(cls.path, "llama_tokenizer.model")},
                "seq_length": 32,
                "prompt_key": "conversations",
                "output_columns": ["input_ids", "labels"]
            }
        )
        cls.handler = AlpacaInstructDataHandler(cls.handler_config)
        cls.dataset = CommonDataLoader(path=cls.path, shuffle=False, split="train",
                                       input_columns=["input_ids", "labels"], handler=[cls.handler_config])

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_default(self):
        assert isinstance(self.dataset, mindspore.dataset.GeneratorDataset)
        assert isinstance(self.handler, mindformers.dataset.handler.AlpacaInstructDataHandler)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_other(self):
        for item in self.dataset:
            assert len(item) == 2
            assert item[0].asnumpy().tolist() == [1, 111, 134, 141, 162, 159, 144, 134, 0, 136, 109, 136, 6, 134, 159,
                                                  134, 141, 164, 143, 139, 142, 164, 140, 134, 164, 106, 143, 134, 159,
                                                  137, 0, 134, 159]
            assert item[1].asnumpy().tolist() == [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                                                  -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                                                  -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]
