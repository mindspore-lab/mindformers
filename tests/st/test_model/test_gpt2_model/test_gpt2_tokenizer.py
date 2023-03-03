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
linux:  pytest ./tests/st/test_model/test_gpt2_model/test_gpt2_tokenizer.py
"""
import json
import os
import shutil
import pytest
from mindformers.models.gpt2.gpt2_tokenizer import Gpt2Tokenizer

@pytest.mark.level0
@pytest.mark.platform_x86_ascend_train
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
class TestGptTokenizerMethod:
    """A test class for testing the Tokenizer"""
    def generate_fake_vocab(self):
        vocabs = {"Ġworld": 0, "hello": 1, "world": 2, "<|endoftext|>": 3}
        with open(os.path.join(self.output_path, 'vocab.json'), 'w') as fp:
            fp.write(json.dumps(vocabs))

        merges = ["#version: 0.2", "h e", "Ġ w", "o r", "o w", "Ġw or", "Ġ h", "l l", "l d", "e l",
                  "Ġw ould", "e ll", "Ġwor ld", "or ld", "ĠW orld", "he l", "l o",
                  "he ll", "Ġw o", "hell o", "r l"]
        with open(os.path.join(self.output_path, 'merges.txt'), 'w') as fp:
            fp.write("\n".join(merges))

    def setup_method(self):
        self.output_path = os.path.join(os.path.dirname(__file__), 'test_tokenizer_output' + str(self))
        self.gpt_path_saved = os.path.join(os.path.dirname(__file__), 'test_gpt_tokenizer_output' + str(self))
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.gpt_path_saved, exist_ok=True)
        self.generate_fake_vocab()

    def teardown_method(self):
        shutil.rmtree(self.output_path)
        shutil.rmtree(self.gpt_path_saved)

    def test_from_pretrained(self):
        """
        Feature: The Gpt2Tokenizer test using from python class
        Description: Using call forward process of the tokenizer without error
        Expectation: The returned ret is not equal to [[6, 7]].
        """
        tokenizer = Gpt2Tokenizer.from_pretrained(self.output_path)
        tokenizer.show_support_list()
        tokenizer("hello world")
        tokenizer = Gpt2Tokenizer.from_pretrained("gpt2")
        tokenizer("hello world")

    @pytest.mark.parametrize('skip_special_tokens', [True, False])
    def test_gpt_decode(self, skip_special_tokens):
        """
        Feature: The Gpt2Tokenizer test using auto_class
        Description: Using call forward process of the tokenizer without error
        Expectation: The returned ret is not equal to [[6, 7]].
        """
        tokenizer = Gpt2Tokenizer.from_pretrained(self.output_path)
        res = tokenizer("hello world")["input_ids"]
        tokenizer.decode(res, skip_special_tokens=skip_special_tokens)

    def test_gpt__call__(self):
        """
        Feature: The Gpt2Tokenizer test call method
        Description: Using call forward process of the tokenizer without error
        Expectation: The returned ret is not equal to [[6, 7]].
        """
        tokenizer = Gpt2Tokenizer.from_pretrained(self.output_path)
        tokenizer("hello world", skip_special_tokens=True)
        tokenizer("hello world", skip_special_tokens=False)

        tokenizer("hello world")
        tokenizer("hello world", padding='max_length', max_length=10)
        tokenizer("hello world", add_special_tokens=False)
        tokenizer("hello world", return_tensors='ms')
        with pytest.raises(ValueError):
            tokenizer(["hello world", "hello world world"], return_tensors='ms')
        tokenizer(["hello world", " world"], max_length=5, padding='max_length', return_tensors='ms')
