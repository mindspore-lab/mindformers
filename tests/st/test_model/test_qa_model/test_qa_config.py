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
Test Module for testing functions of AutoConfig and BertConfig class

How to run this:
windows:  pytest .\\tests\\st\\test_model\\test_qa_model\\test_qa_config.py
linux:  pytest ./tests/st/test_model/test_qa_model/test_qa_config.py

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
    qa_config_path = os.path.join(MindFormerBook.get_project_path(),
                                  'configs', 'qa', 'run_qa_bert_base_uncased.yaml')
    qa_squad_config_path = os.path.join(MindFormerBook.get_project_path(),
                                        'configs', 'qa', 'run_qa_bert_base_uncased.yaml')
    save_path = os.path.join(MindFormerBook.get_default_checkpoint_save_folder(), 'qa')


    AutoConfig.show_support_list()
    # fine-tuning part
    qa_config_a = AutoConfig.from_pretrained('qa_bert_base_uncased') # input a model name
    qa_config_b = AutoConfig.from_pretrained(qa_config_path) # input a path to .yaml file
    logger.info(qa_config_a)
    logger.info(qa_config_b)
    # evaluation and prediction test part
    qa_squad_config_a = AutoConfig.from_pretrained('qa_bert_base_uncased_squad') # input a model name
    qa_squad_config_b = AutoConfig.from_pretrained(qa_squad_config_path) # input a path to .yaml file
    logger.info(qa_squad_config_a)
    logger.info(qa_squad_config_b)

    assert isinstance(qa_config_a, BaseConfig)
    assert isinstance(qa_config_b, BaseConfig)
    assert isinstance(qa_squad_config_a, BaseConfig)
    assert isinstance(qa_squad_config_b, BaseConfig)

    BertConfig.show_support_list()
    support_list = BertConfig.get_support_list()
    logger.info(support_list)
    # fine-tuning part
    qa_config_c = BertConfig.from_pretrained('qa_bert_base_uncased')
    qa_config_d = BertConfig.from_pretrained(qa_config_path)
    qa_config_c.save_pretrained()
    qa_config_d.save_pretrained(save_path, "qa_bert_base_uncased")
    # evaluation and prediction test part
    qa_squad_config_c = BertConfig.from_pretrained('qa_bert_base_uncased_squad')
    qa_squad_config_d = BertConfig.from_pretrained(qa_squad_config_path)
    qa_squad_config_c.save_pretrained()
    qa_squad_config_d.save_pretrained(save_path, "qa_bert_base_uncased_squad")

    assert isinstance(qa_config_c, BaseConfig)
    assert isinstance(qa_config_d, BaseConfig)
    assert isinstance(qa_squad_config_c, BaseConfig)
    assert isinstance(qa_squad_config_d, BaseConfig)


    assert isinstance(qa_config_a, BertConfig)
    assert isinstance(qa_config_a, type(qa_config_b))
    assert isinstance(qa_config_b, type(qa_config_c))
    assert isinstance(qa_squad_config_a, BertConfig)
    assert isinstance(qa_squad_config_a, type(qa_squad_config_c))
    assert isinstance(qa_squad_config_b, type(qa_squad_config_c))
