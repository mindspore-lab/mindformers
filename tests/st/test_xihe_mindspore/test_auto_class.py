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
Test module for testing the AutoClass used for mindformers.
How to run this:
pytest tests/st/test_auto_class.py
"""
import pytest

from mindformers.models import BaseModel, BaseConfig, BaseProcessor, BaseTokenizer
from mindformers import AutoModel, AutoConfig, AutoProcessor, AutoTokenizer


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_auto_model_for_xihe():
    """
    Feature: Build Model from xihe.mindspore platform
    Description: Test build function to instance API from xihe.mindspore keys.
    Expectation: ValueError
    """
    xihe_name_list = [
        'mindspore/clip_vit_b_32', 'mindspore/vit_base_p16', 'mindspore/swin_base_p4w7',
        'mindspore/tokcls_bert_base_chinese_cluener', 'mindspore/txtcls_bert_base_uncased_mnli'
    ]
    for xihe_name in xihe_name_list:
        model = AutoModel.from_pretrained(xihe_name)
        assert isinstance(model, BaseModel)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_auto_config_for_xihe():
    """
    Feature: Build Model Config from xihe.mindspore platform
    Description: Test build function to instance API from xihe.mindspore keys.
    Expectation: ValueError
    """
    xihe_name_list = [
        'mindspore/clip_vit_b_32', 'mindspore/vit_base_p16', 'mindspore/swin_base_p4w7',
        'mindspore/tokcls_bert_base_chinese_cluener', 'mindspore/txtcls_bert_base_uncased_mnli'
    ]
    for xihe_name in xihe_name_list:
        config = AutoConfig.from_pretrained(xihe_name)
        assert isinstance(config, BaseConfig)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_auto_processor_for_xihe():
    """
    Feature: Build Model Processor from xihe.mindspore platform
    Description: Test build function to instance API from xihe.mindspore keys.
    Expectation: ValueError
    """
    xihe_name_list = [
        'mindspore/clip_vit_b_32', 'mindspore/vit_base_p16', 'mindspore/swin_base_p4w7',
        'mindspore/tokcls_bert_base_chinese_cluener', 'mindspore/txtcls_bert_base_uncased_mnli'
    ]
    for xihe_name in xihe_name_list:
        processor = AutoProcessor.from_pretrained(xihe_name)
        assert isinstance(processor, BaseProcessor)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_auto_tokenizer_for_xihe():
    """
    Feature: Build Model Tokenizer from xihe.mindspore platform
    Description: Test build function to instance API from xihe.mindspore keys.
    Expectation: ValueError
    """
    xihe_name_list = [
        'mindspore/clip_vit_b_32', 'mindspore/tokcls_bert_base_chinese_cluener',
        'mindspore/txtcls_bert_base_uncased_mnli'
    ]
    for xihe_name in xihe_name_list:
        tokenizer = AutoTokenizer.from_pretrained(xihe_name)
        assert isinstance(tokenizer, BaseTokenizer)
