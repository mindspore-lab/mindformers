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
"""test build_config"""
import unittest
from mindformers import MindFormerConfig, LlamaTokenizer, Blip2Processor, \
    Blip2ImageProcessor, GPT2Processor, GPT2Tokenizer
from mindformers.models.build_processor import build_processor


class TestBuildProcessor(unittest.TestCase):
    """test func build_model_config()"""
    def test_blip2(self):
        config = MindFormerConfig("configs/blip2/run_blip2_stage2_vit_g_llama_7b.yaml")
        processor = build_processor(config.processor)
        self.assertTrue(isinstance(processor, Blip2Processor))
        self.assertTrue(isinstance(processor.image_processor, Blip2ImageProcessor))
        self.assertTrue(isinstance(processor.tokenizer, LlamaTokenizer))

    def test_gpt(self):
        config = MindFormerConfig("configs/gpt2/run_gpt2.yaml")
        processor = build_processor(config.processor)
        self.assertTrue(isinstance(processor, GPT2Processor))
        self.assertTrue(isinstance(processor.tokenizer, GPT2Tokenizer))
