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
Test Module for testing functions of AutoConfig and BertConfig class

How to run this:
windows:  pytest .\\tests\\st\\test_model\\test_bert_model\\test_ner_config.py
linux:  pytest ./tests/st/test_model/test_bert_model/test_ner_config.py

Note:
    model name and config name should have the same prefix
Example:
    ClipModel and BertConfig have the same prefix, Clip
"""
import os
from mindformers import MindFormerBook, AutoConfig
from mindformers.models import BertConfig, BaseConfig
from mindformers.tools import logger


# the first method to load model config, AutoConfig
def test_config():
    """
    Feature: AutoConfig, BertConfig
    Description: Test to get config instance by AutoConfig.from_pretrained
    Expectation: TypeError, ValueError
    """
    ner_dense_config_path = os.path.join(MindFormerBook.get_project_path(),
                                         'configs', 'ner', 'model_config', 'ner_bert_base_chinese_dense.yaml')
    ner_dense_cluener_config_path = os.path.join(MindFormerBook.get_project_path(),
                                                 'configs', 'ner', 'model_config',
                                                 'ner_bert_base_chinese_dense_cluener.yaml')
    save_path = os.path.join(MindFormerBook.get_default_checkpoint_save_folder(), 'ner')


    AutoConfig.show_support_list()
    # fine-tuning part
    ner_dense_config_a = AutoConfig.from_pretrained('ner_bert_base_chinese_dense') # input a model name
    ner_dense_config_b = AutoConfig.from_pretrained(ner_dense_config_path) # input a path to .yaml file
    logger.info(ner_dense_config_a)
    logger.info(ner_dense_config_b)
    # evaluation and prediction test part
    ner_dense_cluener_config_a = AutoConfig.from_pretrained('ner_bert_base_chinese_dense_cluener') # input a model name
    ner_dense_cluener_config_b = AutoConfig.from_pretrained(ner_dense_cluener_config_path) # input a path to .yaml file
    logger.info(ner_dense_cluener_config_a)
    logger.info(ner_dense_cluener_config_b)

    assert isinstance(ner_dense_config_a, BaseConfig)
    assert isinstance(ner_dense_config_b, BaseConfig)
    assert isinstance(ner_dense_cluener_config_a, BaseConfig)
    assert isinstance(ner_dense_cluener_config_b, BaseConfig)

    BertConfig.show_support_list()
    support_list = BertConfig.get_support_list()
    logger.info(support_list)
    # fine-tuning part
    ner_dense_config_c = BertConfig.from_pretrained('ner_bert_base_chinese_dense')
    ner_dense_config_d = BertConfig.from_pretrained(ner_dense_config_path)
    ner_dense_config_c.save_pretrained()
    ner_dense_config_d.save_pretrained(save_path, "ner_bert_base_chinese_dense")
    # evaluation and prediction test part
    ner_dense_cluener_config_c = BertConfig.from_pretrained('ner_bert_base_chinese_dense_cluener')
    ner_dense_cluener_config_d = BertConfig.from_pretrained(ner_dense_cluener_config_path)
    ner_dense_cluener_config_c.save_pretrained()
    ner_dense_cluener_config_d.save_pretrained(save_path, "ner_bert_base_chinese_dense_cluener")

    assert isinstance(ner_dense_config_c, BaseConfig)
    assert isinstance(ner_dense_config_d, BaseConfig)
    assert isinstance(ner_dense_cluener_config_c, BaseConfig)
    assert isinstance(ner_dense_cluener_config_d, BaseConfig)


    assert isinstance(ner_dense_config_a, BertConfig)
    assert isinstance(ner_dense_config_a, type(ner_dense_config_b))
    assert isinstance(ner_dense_config_b, type(ner_dense_config_c))
    assert isinstance(ner_dense_cluener_config_a, BertConfig)
    assert isinstance(ner_dense_cluener_config_a, type(ner_dense_cluener_config_c))
    assert isinstance(ner_dense_cluener_config_b, type(ner_dense_cluener_config_c))
