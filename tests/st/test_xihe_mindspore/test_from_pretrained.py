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
Test module for testing the from_pretrained attribution used for mindformers.
How to run this:
pytest tests/st/test_from_pretrained.py
"""
import pytest

from mindformers.models import BaseModel, BaseConfig, BaseProcessor, BaseTokenizer


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_clip_for_xihe():
    """
    Feature: Build Clip Model/Config/Processor/Tokenizer from xihe.mindspore platform
    Description: Test build function to instance API from xihe.mindspore keys.
    Expectation: ValueError
    """
    from mindformers.models import CLIPModel, CLIPConfig, CLIPProcessor, CLIPTokenizer
    xihe_clip_keys = 'mindspore/clip_vit_b_32'
    clip_model = CLIPModel.from_pretrained(xihe_clip_keys)
    assert isinstance(clip_model, BaseModel)

    clip_config = CLIPConfig.from_pretrained(xihe_clip_keys)
    assert isinstance(clip_config, BaseConfig)

    clip_processor = CLIPProcessor.from_pretrained(xihe_clip_keys)
    assert isinstance(clip_processor, BaseProcessor)

    clip_tokenizer = CLIPTokenizer.from_pretrained(xihe_clip_keys)
    assert isinstance(clip_tokenizer, BaseTokenizer)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cls_for_xihe():
    """
    Feature: Build ImageClassification Model/Config/Processor from xihe.mindspore platform
    Description: Test build function to instance API from xihe.mindspore keys.
    Expectation: ValueError
    """
    from mindformers.models import ViTForImageClassification, ViTProcessor, ViTConfig, \
        SwinForImageClassification, SwinProcessor, SwinConfig
    xihe_vit_keys = 'mindspore/vit_base_p16'
    vit_model = ViTForImageClassification.from_pretrained(xihe_vit_keys)
    assert isinstance(vit_model, BaseModel)

    vit_config = ViTConfig.from_pretrained(xihe_vit_keys)
    assert isinstance(vit_config, BaseConfig)

    vit_processor = ViTProcessor.from_pretrained(xihe_vit_keys)
    assert isinstance(vit_processor, BaseProcessor)

    xihe_swin_keys = 'mindspore/swin_base_p4w7'
    swin_model = SwinForImageClassification.from_pretrained(xihe_swin_keys)
    assert isinstance(swin_model, BaseModel)

    swin_config = SwinConfig.from_pretrained(xihe_swin_keys)
    assert isinstance(swin_config, BaseConfig)

    swin_processor = SwinProcessor.from_pretrained(xihe_swin_keys)
    assert isinstance(swin_processor, BaseProcessor)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tokcls_for_xihe():
    """
    Feature: Build TokenClassification Model/Config/Processor/Tokenizer from xihe.mindspore platform
    Description: Test build function to instance API from xihe.mindspore keys.
    Expectation: ValueError
    """
    from mindformers.models import BertForTokenClassification, BertProcessor, BertTokenizer, BertConfig
    xihe_tokcls_keys = 'mindspore/tokcls_bert_base_chinese_cluener'
    tokcls_model = BertForTokenClassification.from_pretrained(xihe_tokcls_keys)
    assert isinstance(tokcls_model, BaseModel)

    tokcls_config = BertConfig.from_pretrained(xihe_tokcls_keys)
    assert isinstance(tokcls_config, BaseConfig)

    tokcls_processor = BertProcessor.from_pretrained(xihe_tokcls_keys)
    assert isinstance(tokcls_processor, BaseProcessor)

    tokcls_tokenizer = BertTokenizer.from_pretrained(xihe_tokcls_keys)
    assert isinstance(tokcls_tokenizer, BaseTokenizer)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_txtcls_for_xihe():
    """
    Feature: Build TextClassification Model/Config/Processor/Tokenizer from xihe.mindspore platform
    Description: Test build function to instance API from xihe.mindspore keys.
    Expectation: ValueError
    """
    from mindformers.models import BertForMultipleChoice, BertProcessor, BertTokenizer, BertConfig
    xihe_tokcls_keys = 'mindspore/txtcls_bert_base_uncased_mnli'
    tokcls_model = BertForMultipleChoice.from_pretrained(xihe_tokcls_keys)
    assert isinstance(tokcls_model, BaseModel)

    tokcls_config = BertConfig.from_pretrained(xihe_tokcls_keys)
    assert isinstance(tokcls_config, BaseConfig)

    tokcls_processor = BertProcessor.from_pretrained(xihe_tokcls_keys)
    assert isinstance(tokcls_processor, BaseProcessor)

    tokcls_tokenizer = BertTokenizer.from_pretrained(xihe_tokcls_keys)
    assert isinstance(tokcls_tokenizer, BaseTokenizer)
