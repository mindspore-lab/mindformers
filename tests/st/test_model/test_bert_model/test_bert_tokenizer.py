# Copyright 2022 Huawei Technologies Co., Ltd
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
linux:  pytest ./tests/st/test_model/test_bert_model/test_bert_tokenizer.py
"""
import os
# import shutil
# import pytest

from mindformers import BertTokenizer, AutoTokenizer

# @pytest.mark.level0
# @pytest.mark.platform_x86_ascend_training
# @pytest.mark.platform_arm_ascend_training
# @pytest.mark.env_onecard
class TestBertTokenizerMethod:
    """A test class for testing the AutoTokenizer"""
    def generate_fake_vocab(self):
        vocabs = ["[PAD]", "[unused1]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "!"]
        with open(os.path.join(self.output_path, 'vocab.txt'), 'w') as fp:
            for item in vocabs:
                fp.write(item + '\n')

    def setup_method(self):
        self.output_path = os.path.join(os.path.dirname(__file__), 'test_tokenizer_output' + str(self))
        self.bert_path_saved = os.path.join(os.path.dirname(__file__), 'test_bert_tokenizer_output' + str(self))
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.bert_path_saved, exist_ok=True)
        self.generate_fake_vocab()

    # def teardown_method(self):
    #     shutil.rmtree(self.output_path)
    #     shutil.rmtree(self.bert_path_saved)

    def test_from_pretrained(self):
        """
        Feature: The BertTokenizer test using from python class
        Description: Using call forward process of the tokenizer without error
        Expectation: The returned ret is not equal to [[6, 7]].
        """
        tokenizer = BertTokenizer.from_pretrained(self.output_path)
        tokenizer.show_support_list()
        tokenizer("hello world")

    def test_auto_tokenizer(self):
        """
        Feature: The BertTokenizer test using auto_class
        Description: Using call forward process of the tokenizer without error
        Expectation: The returned ret is not equal to [[6, 7]].
        """
        tokenizer = BertTokenizer.from_pretrained(self.output_path)
        tokenizer.save_pretrained(self.bert_path_saved)
        tokenizer = AutoTokenizer.from_pretrained(self.bert_path_saved)
        tokenizer("hello world")

    # @pytest.mark.parametrize('skip_special_tokens', [True, False])
    def test_bert_decode(self, skip_special_tokens):
        """
        Feature: The BertTokenizer test using auto_class
        Description: Using call forward process of the tokenizer without error
        Expectation: The returned ret is not equal to [[6, 7]].
        """
        tokenizer = BertTokenizer.from_pretrained(self.output_path)
        res = tokenizer("hello world")["input_ids"]
        tokenizer.decode(res, skip_special_tokens=skip_special_tokens)

    def test_bert__call__(self):
        """
        Feature: The BertTokenizer test call method
        Description: Using call forward process of the tokenizer without error
        Expectation: The returned ret is not equal to [[6, 7]].
        """
        tokenizer = BertTokenizer.from_pretrained(self.output_path)
        tokenizer("hello world", skip_special_tokens=True)
        tokenizer("hello world", skip_special_tokens=False)

        tokenizer("hello world")
        tokenizer("hello world", padding='max_length', max_length=10)
        tokenizer("hello world", add_special_tokens=False)
        tokenizer("hello world", return_tensors='ms')

        # with pytest.raises(ValueError):
        #     tokenizer(["hello world", "today is a good day"], return_tensors='ms')

        tokenizer(["hello world", "today is a good day"], max_length=7, padding='max_length', return_tensors='ms')
