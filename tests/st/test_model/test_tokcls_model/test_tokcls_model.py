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
Test Module for testing functions of AutoModel and BertTokenClassification class

How to run this:
windows:  pytest .\\tests\\st\\test_model\\test_tokcls_model\\test_tokcls_model.py
linux:  pytest ./tests/st/test_model/test_tokcls_model/test_tokcls_model.py
"""
import os
import pytest
from mindformers import MindFormerBook, AutoModel, AutoConfig
from mindformers.models import BertTokenClassification, BaseModel
from mindformers.tools import logger


class TestModelMethod:
    """A test class for testing Model classes"""
    def setup_method(self):
        """get_input"""
        # fine-tuning
        self.tokcls_checkpoint_dir = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(),
                                                  'tokcls')
        self.tokcls_config_path = os.path.join(MindFormerBook.get_project_path(),
                                               'configs', 'tokcls', 'model_config',
                                               'tokcls_bert_base_chinese.yaml')
        self.tokcls_config = AutoConfig.from_pretrained('tokcls_bert_base_chinese')

        # evaluation and prediction
        self.tokcls_cluener_checkpoint_dir = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(),
                                                          'tokcls')
        self.tokcls_cluener_config_path = os.path.join(MindFormerBook.get_project_path(),
                                                       'configs', 'tokcls', 'model_config',
                                                       'tokcls_bert_base_chinese_cluener.yaml')
        self.tokcls_cluener_config = AutoConfig.from_pretrained('tokcls_bert_base_chinese_cluener')

        # save path
        self.save_directory = os.path.join(MindFormerBook.get_default_checkpoint_save_folder(), 'tokcls')
    # the first method to load model, AutoModel
    @pytest.mark.level0
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.env_onecard
    def test_auto_model(self):
        """
        Feature: AutoModel, from_pretrained, from_config
        Description: Test to get model instance by AutoModel.from_pretrained
                    and AutoModel.from_config
        Expectation: TypeError, ValueError, RuntimeError
        """
        AutoModel.show_support_list()
        support_list = AutoModel.get_support_list()
        logger.info(support_list)

        # fine-tuning part
        # input model name
        tokcls_model_a = AutoModel.from_pretrained('tokcls_bert_base_chinese')
        # input model directory
        tokcls_model_b = AutoModel.from_pretrained(self.tokcls_checkpoint_dir)
        # input yaml path
        tokcls_model_c = AutoModel.from_config(self.tokcls_config_path)
        # input config
        tokcls_model_d = AutoModel.from_config(self.tokcls_config)

        tokcls_model_a.save_pretrained(self.save_directory, save_name='tokcls_bert_base_chinese')

        assert isinstance(tokcls_model_a, BertTokenClassification)
        assert isinstance(tokcls_model_b, BertTokenClassification)
        assert isinstance(tokcls_model_c, BertTokenClassification)
        assert isinstance(tokcls_model_d, BertTokenClassification)

        assert isinstance(tokcls_model_a, BaseModel)
        assert isinstance(tokcls_model_b, BaseModel)
        assert isinstance(tokcls_model_c, BaseModel)
        assert isinstance(tokcls_model_d, BaseModel)

        # evaluation and prediction test part
        # input model name
        tokcls_cluener_model_a = AutoModel.from_pretrained('tokcls_bert_base_chinese_cluener')
        # input model directory
        tokcls_cluener_model_b = AutoModel.from_pretrained(self.tokcls_cluener_checkpoint_dir)
        # input yaml path
        tokcls_cluener_model_c = AutoModel.from_config(self.tokcls_cluener_config_path)
        # input config
        tokcls_cluener_model_d = AutoModel.from_config(self.tokcls_cluener_config)

        tokcls_cluener_model_a.save_pretrained(self.save_directory, save_name='tokcls_bert_base_chinese_cluener')

        assert isinstance(tokcls_cluener_model_a, BertTokenClassification)
        assert isinstance(tokcls_cluener_model_b, BertTokenClassification)
        assert isinstance(tokcls_cluener_model_c, BertTokenClassification)
        assert isinstance(tokcls_cluener_model_d, BertTokenClassification)

        assert isinstance(tokcls_cluener_model_a, BaseModel)
        assert isinstance(tokcls_cluener_model_b, BaseModel)
        assert isinstance(tokcls_cluener_model_c, BaseModel)
        assert isinstance(tokcls_cluener_model_d, BaseModel)

        BertTokenClassification.show_support_list()
        support_list = BertTokenClassification.get_support_list()
        logger.info(support_list)

        # fine-tuning part
        # input model name, load model and weights
        tokcls_model_e = BertTokenClassification.from_pretrained('tokcls_bert_base_chinese')
        # input model directory, load model and weights
        tokcls_model_f = BertTokenClassification.from_pretrained(self.tokcls_checkpoint_dir)
        # input config, load model weights
        tokcls_model_g = BertTokenClassification(self.tokcls_config)
        # input config, load model without weights
        self.tokcls_config.checkpoint_name_or_path = None
        tokcls_model_h = BertTokenClassification(self.tokcls_config)

        tokcls_model_e.save_pretrained(self.save_directory, save_name='tokcls_bert_base_chinese')

        assert isinstance(tokcls_model_e, BertTokenClassification)
        assert isinstance(tokcls_model_f, BertTokenClassification)
        assert isinstance(tokcls_model_g, BertTokenClassification)
        assert isinstance(tokcls_model_h, BertTokenClassification)

        assert isinstance(tokcls_model_e, BaseModel)
        assert isinstance(tokcls_model_f, BaseModel)
        assert isinstance(tokcls_model_g, BaseModel)
        assert isinstance(tokcls_model_h, BaseModel)

        # evaluation and prediction test part
        # input model name, load model and weights
        tokcls_cluener_model_e = BertTokenClassification.from_pretrained('tokcls_bert_base_chinese_cluener')
        # input model directory, load model and weights
        tokcls_cluener_model_f = BertTokenClassification.from_pretrained(self.tokcls_cluener_checkpoint_dir)
        # input config, load model weights
        tokcls_cluener_model_g = BertTokenClassification(self.tokcls_cluener_config)
        # input config, load model without weights
        self.tokcls_cluener_config.checkpoint_name_or_path = None
        tokcls_cluener_model_h = BertTokenClassification(self.tokcls_cluener_config)

        tokcls_cluener_model_e.save_pretrained(self.save_directory, save_name='tokcls_bert_base_chinese_cluener')

        assert isinstance(tokcls_cluener_model_e, BertTokenClassification)
        assert isinstance(tokcls_cluener_model_f, BertTokenClassification)
        assert isinstance(tokcls_cluener_model_g, BertTokenClassification)
        assert isinstance(tokcls_cluener_model_h, BertTokenClassification)

        assert isinstance(tokcls_cluener_model_e, BaseModel)
        assert isinstance(tokcls_cluener_model_f, BaseModel)
        assert isinstance(tokcls_cluener_model_g, BaseModel)
        assert isinstance(tokcls_cluener_model_h, BaseModel)
