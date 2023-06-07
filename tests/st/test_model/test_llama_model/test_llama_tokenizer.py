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
"""
Test Module for testing Tokenizer class

How to run this:
linux:  pytest ./tests/st/test_model/test_llama_model/test_llama_tokenizer.py
"""
import os
import shutil
import pytest
from mindformers.models.llama.llama_tokenizer import LlamaTokenizer

@pytest.mark.level0
@pytest.mark.platform_x86_ascend_train
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
class TestLlamaTokenizerMethod:
    """A test class for testing the Tokenizer"""
    def setup_method(self):
        self.output_path = os.path.join(os.path.dirname(__file__), 'test_tokenizer_output' + str(self))
        os.makedirs(self.output_path, exist_ok=True)

    def teardown_method(self):
        shutil.rmtree(self.output_path)

    def test_from_pretrained(self):
        """
        Feature: The LlamaTokenizer test using from python class
        Description: Using call forward process of the tokenizer without error
        Expectation: The returned ret is not equal to [[6, 7]].
        """
        tokenizer = LlamaTokenizer.from_pretrained(os.path.dirname(__file__))
        tokenizer.show_support_list()
        tokenizer("hello world")

    def test_llama__call__(self):
        """
        Feature: The LlamaTokenizer test call method
        Description: Using call forward process of the tokenizer without error
        Expectation: The returned ret is not equal to [[6, 7]].
        """
        tokenizer = LlamaTokenizer.from_pretrained(os.path.dirname(__file__))
        tokenizer("hello world", skip_special_tokens=True)
        tokenizer("hello world", skip_special_tokens=False)
        tokenizer("hello world")
        tokenizer("hello world", padding='max_length', max_length=10)
        tokenizer("hello world", add_special_tokens=False)
        tokenizer("hello world", return_tensors='ms')
        with pytest.raises(ValueError):
            tokenizer(["hello world", "hello world world"], return_tensors='ms')
        tokenizer(["hello world", " world"], max_length=5, padding='max_length', return_tensors='ms')
