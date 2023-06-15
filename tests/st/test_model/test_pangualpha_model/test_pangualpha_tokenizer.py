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
linux:  pytest ./tests/st/test_model/test_pangualpha_model/test_pangualpha_tokenizer.py
"""
import os
import pytest
from mindformers.models.pangualpha import PanguAlphaTokenizer

@pytest.mark.level0
@pytest.mark.platform_x86_ascend_train
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
class TestPanguAlphaTokenizerMethod:
    """A test class for testing the Tokenizer"""
    def setup_method(self):
        self.vocab_path = os.path.join(os.path.dirname(__file__))

    def test_from_pretrained(self):
        """
        Feature: The PanguAlphaTokenizer test using from python class
        Description: Using call forward process of the tokenizer without error
        Expectation: The returned ret is not equal to [[6, 7]].
        """
        tokenizer = PanguAlphaTokenizer.from_pretrained(self.vocab_path)
        tokenizer.show_support_list()
        tokenizer("今天天气怎么样")
        tokenizer = PanguAlphaTokenizer.from_pretrained("pangualpha_2_6b")
        tokenizer("今天天气怎么样")

    @pytest.mark.parametrize('skip_special_tokens', [True, False])
    def test_pangualpha_decode(self, skip_special_tokens):
        """
        Feature: The PanguAlphaTokenizer test using auto_class
        Description: Using call forward process of the tokenizer without error
        Expectation: The returned ret is not equal to [[6, 7]].
        """
        tokenizer = PanguAlphaTokenizer.from_pretrained(self.vocab_path)
        res = tokenizer("今天天气怎么样")["input_ids"]
        tokenizer.decode(res, skip_special_tokens=skip_special_tokens)

    def test_pangualpha__call__(self):
        """
        Feature: The PanguAlphaTokenizer test call method
        Description: Using call forward process of the tokenizer without error
        Expectation: The returned ret is not equal to [[6, 7]].
        """
        tokenizer = PanguAlphaTokenizer.from_pretrained(self.vocab_path)
        tokenizer("今天天气怎么样", skip_special_tokens=True)
        tokenizer("今天天气怎么样", skip_special_tokens=False)

        tokenizer("今天天气怎么样")
        tokenizer("今天天气怎么样", padding='max_length', max_length=10)
        tokenizer("今天天气怎么样", add_special_tokens=False)
        tokenizer("今天天气怎么样", return_tensors='ms')
        with pytest.raises(ValueError):
            tokenizer(["今天天气怎么样", "今天天气怎么样，怎么样"], return_tensors='ms')
        tokenizer(["今天天气怎么样", " 怎么样"], max_length=10, padding='max_length', return_tensors='ms')
