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
windows:  pytest .\\tests\\st\\test_model\\test_tokcls_model\\test_tokcls_config.py
linux:  pytest ./tests/st/test_model/test_tokcls_model/test_tokcls_config.py

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
    tokcls_dense_config_path = os.path.join(MindFormerBook.get_project_path(),
                                            'configs', 'tokcls', 'run_tokcls_bert_base_chinese.yaml')
    tokcls_dense_cluener_config_path = os.path.join(MindFormerBook.get_project_path(),
                                                    'configs', 'tokcls', 'run_tokcls_bert_base_chinese_cluener.yaml')
    save_path = os.path.join(MindFormerBook.get_default_checkpoint_save_folder(), 'tokcls')


    AutoConfig.show_support_list()
    # fine-tuning part
    tokcls_dense_config_a = AutoConfig.from_pretrained('tokcls_bert_base_chinese') # input a model name
    tokcls_dense_config_b = AutoConfig.from_pretrained(tokcls_dense_config_path) # input a path to .yaml file
    logger.info(tokcls_dense_config_a)
    logger.info(tokcls_dense_config_b)
    # evaluation and prediction test part
    tokcls_dense_cluener_config_a = AutoConfig.from_pretrained('tokcls_bert_base_chinese_cluener') # input a model name
    tokcls_dense_cluener_config_b = AutoConfig.from_pretrained(tokcls_dense_cluener_config_path) # input a path to .yaml file
    logger.info(tokcls_dense_cluener_config_a)
    logger.info(tokcls_dense_cluener_config_b)

    assert isinstance(tokcls_dense_config_a, BaseConfig)
    assert isinstance(tokcls_dense_config_b, BaseConfig)
    assert isinstance(tokcls_dense_cluener_config_a, BaseConfig)
    assert isinstance(tokcls_dense_cluener_config_b, BaseConfig)

    BertConfig.show_support_list()
    support_list = BertConfig.get_support_list()
    logger.info(support_list)
    # fine-tuning part
    tokcls_dense_config_c = BertConfig.from_pretrained('tokcls_bert_base_chinese')
    tokcls_dense_config_d = BertConfig.from_pretrained(tokcls_dense_config_path)
    tokcls_dense_config_c.save_pretrained()
    tokcls_dense_config_d.save_pretrained(save_path, "tokcls_bert_base_chinese")
    # evaluation and prediction test part
    tokcls_dense_cluener_config_c = BertConfig.from_pretrained("tokcls_bert_base_chinese_cluener")
    tokcls_dense_cluener_config_d = BertConfig.from_pretrained(tokcls_dense_cluener_config_path)
    tokcls_dense_cluener_config_c.save_pretrained()
    tokcls_dense_cluener_config_d.save_pretrained(save_path, "tokcls_bert_base_chinese_cluener")

    assert isinstance(tokcls_dense_config_c, BaseConfig)
    assert isinstance(tokcls_dense_config_d, BaseConfig)
    assert isinstance(tokcls_dense_cluener_config_c, BaseConfig)
    assert isinstance(tokcls_dense_cluener_config_d, BaseConfig)


    assert isinstance(tokcls_dense_config_a, BertConfig)
    assert isinstance(tokcls_dense_config_a, type(tokcls_dense_config_b))
    assert isinstance(tokcls_dense_config_b, type(tokcls_dense_config_c))
    assert isinstance(tokcls_dense_cluener_config_a, BertConfig)
    assert isinstance(tokcls_dense_cluener_config_a, type(tokcls_dense_cluener_config_c))
    assert isinstance(tokcls_dense_cluener_config_b, type(tokcls_dense_cluener_config_c))
