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
"""test text_transforms"""
import tempfile
import os
import pytest
import numpy as np
from mindformers import BloomTokenizer
from mindformers.dataset.transforms import (
    RandomChoiceTokenizerForward, TokenizerForward, TokenizeWithLabel, LabelPadding, CaptionTransform
)
from tests.st.test_ut.test_tokenizers.get_vocab_model import get_bbpe_vocab_model


string = "An increasing sequence: one, two, three."
b_string = b"An increasing sequence: one, two, three."
tokens = [113, 163, 116, 114, 191, 106, 123, 196, 13, 167, 4, 102, 126, 4, 199, 17, 5]
temp_dir = tempfile.TemporaryDirectory()
path = temp_dir.name
get_bbpe_vocab_model("bloom", path)
tokenizer_model_path = os.path.join(path, "bloom_tokenizer.json")
tokenizer = BloomTokenizer(vocab_file=tokenizer_model_path)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_random_choice_tokenizer_forward():
    """
    Feature: test transforms.RandomChoiceTokenizerForward
    Description: test RandomChoiceTokenizerForward function
    Expectation: success
    """
    random_choice_tokenizer_forward = RandomChoiceTokenizerForward(tokenizer=tokenizer, max_length=12)
    res = random_choice_tokenizer_forward(np.array([string]))
    assert res == tokens


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tokenizer_forward():
    """
    Feature: test transforms.TokenizerForward
    Description: test TokenizerForward function
    Expectation: success
    """
    tokenizer_forward = TokenizerForward(tokenizer=tokenizer, max_length=12)
    res = tokenizer_forward([string])
    assert res[0] == tokens
    tokenizer_forward = TokenizerForward(tokenizer=tokenizer, max_length=12)
    res = tokenizer_forward([b_string])
    assert res[0] == tokens


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tokenize_with_label():
    """
    Feature: test transforms.TokenizeWithLabel
    Description: test TokenizeWithLabel function
    Expectation: success
    """
    tokenize_with_label = TokenizeWithLabel(tokenizer=tokenizer, max_length=12)
    res = tokenize_with_label(string, tokens)
    res_byte = tokenize_with_label(b_string, tokens)
    assert len(res) == len(res_byte) == 4
    assert res[0].tolist() == res_byte[0].tolist() == tokens
    assert res[1].tolist() == res_byte[1].tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert res[2].tolist() == res_byte[2].tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert res[3] == res_byte[3] == tokens


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_label_padding():
    """
    Feature: test transforms.LabelPadding
    Description: test LabelPadding function
    Expectation: success
    """
    label_padding = LabelPadding(max_length=12)
    res = label_padding(tokens)
    assert res.tolist() == [0, 113, 163, 116, 114, 191, 106, 123, 196, 13, 167, 4]
    label_padding = LabelPadding(max_length=20)
    res = label_padding(tokens)
    assert res.tolist() == [0, 113, 163, 116, 114, 191, 106, 123, 196, 13, 167, 4, 102, 126, 4, 199, 17, 5, 0, 0]


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_caption_transform():
    """
    Feature: test transforms.CaptionTransform
    Description: test CaptionTransform function
    Expectation: success
    """
    caption_transform = CaptionTransform(tokenizer=tokenizer, max_length=12)
    res = caption_transform(np.array(string))
    assert res.tolist() == [15, 22, 163, 116, 114, 191, 106, 123, 196, 167, 4, 102]
    caption_transform = CaptionTransform(tokenizer=tokenizer, max_length=12)
    res = caption_transform(np.array([string]))
    assert res[0].tolist() == [15, 22, 163, 116, 114, 191, 106, 123, 196, 167, 4, 102]
    caption_transform = CaptionTransform(tokenizer=tokenizer, max_length=12, max_words=5)
    res = caption_transform(np.array([string]))
    assert res[0].tolist() == [15, 22, 163, 116, 114, 191, 106, 123, 196, 167, 4, 102]
