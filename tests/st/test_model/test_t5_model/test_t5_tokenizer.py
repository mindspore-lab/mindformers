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

import pytest
from mindformers import T5Tokenizer


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
class TestT5TokenizerMethod:
    """A test class for testing the BertTokenizer"""
    def test_from_pretrained_tokenizer(self):
        """
        Feature: The T5Tokenizer test using from python class
        Description: Using call forward process of the tokenizer without error
        Expectation: The returned ret is not equal to [[6, 7]].
        """
        tokenizer = T5Tokenizer.from_pretrained(os.path.dirname(__file__))
        tokenizer.show_support_list()
        tokenizer("hello world")
