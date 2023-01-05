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
linux:  pytest ./tests/st/test_model/test_t5_model/test_t5_tokenizer.py
"""
import os
import shutil

import pytest

from mindformers import T5Tokenizer, AutoTokenizer, BertTokenizer

@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
class TestT5TokenizerMethod:
    """A test class for testing the BertTokenizer"""
    def setup_method(self):
        self.output_path = os.path.join(os.path.dirname(__file__), 'test_tokenizer_output')
        os.makedirs(self.output_path, exist_ok=True)

    def teardown_method(self):
        shutil.rmtree(self.output_path)

    def test_from_pretrained_tokenizer(self):
        """
        Feature: The T5Tokenizer test using from python class
        Description: Using call forward process of the tokenizer without error
        Expectation: The returned ret is not equal to [[6, 7]].
        """
        tokenizer = T5Tokenizer.from_pretrained(os.path.dirname(__file__))
        tokenizer.show_support_list()
        tokenizer("hello world")

    def test_auto_tokenizer(self):
        """
        Feature: The T5Tokenizer test using auto_class
        Description: Using call forward process of the tokenizer without error
        Expectation: The returned ret is not equal to [[6, 7]].
        """
        tokenizer = T5Tokenizer.from_pretrained(os.path.dirname(__file__))
        tokenizer.save_pretrained(self.output_path)
        tokenizer = AutoTokenizer.from_pretrained(self.output_path)
        tokenizer("hello world")

    @pytest.mark.parametrize('skip_special_tokens', [True, False])
    def test_t5_decode(self, skip_special_tokens):
        """
        Feature: The T5Tokenizer test using auto_class
        Description: Using call forward process of the tokenizer without error
        Expectation: The returned ret is not equal to [[6, 7]].
        """
        tokenizer = T5Tokenizer.from_pretrained(os.path.dirname(__file__))
        res = tokenizer("hello world")["input_ids"]
        tokenizer.decode(res, skip_special_tokens=skip_special_tokens)

    def test_wrong_tokneizer(self):
        """
        Feature: Check the wrong args for the bert tokenizer
        Description: Using call forward process of the tokenizer without error
        Expectation: No ValueError.
        """
        with pytest.raises(ValueError):
            BertTokenizer.from_pretrained("t5_small")

    def test_t5__call__(self):
        """
        Feature: The T5Tokenizer test call method
        Description: Using call forward process of the tokenizer without error
        Expectation: The returned ret is not equal to [[6, 7]].
        """
        tokenizer = T5Tokenizer.from_pretrained("t5_small")
        tokenizer("hello world", skip_special_tokens=True)
        tokenizer("hello world", skip_special_tokens=False)

        res = tokenizer("hello world")
        assert res == {'input_ids': [21820, 296, 1], 'attention_mask': [1, 1, 1]}

        res = tokenizer("hello world", padding='max_length', max_length=10)
        assert res == {'input_ids': [21820, 296, 1, 0, 0, 0, 0, 0, 0, 0],
                       'attention_mask': [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]}

        res = tokenizer("hello world", add_special_tokens=False)
        assert res == {'input_ids': [21820, 296], 'attention_mask': [1, 1]}

        res = tokenizer("hello world", return_tensors='ms')
        assert res['input_ids'].asnumpy().tolist() == [21820, 296, 1]
        assert res['attention_mask'].asnumpy().tolist() == [1, 1, 1]

        with pytest.raises(ValueError):
            tokenizer(["hello world", "today is a good day"], return_tensors='ms')

        tokenizer(["hello world", "today is a good day"], max_length=7, padding='max_length', return_tensors='ms')
