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
Test Module for testing functions of AutoModel and BertSoftmaxForNer class

How to run this:
windows:  pytest .\\tests\\st\\test_model\\test_bert_model\\test_ner_model.py
linux:  pytest ./tests/st/test_model/test_bert_model/test_ner_model.py
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
        self.ner_dense_checkpoint_dir = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(),
                                                     'ner')
        self.ner_dense_config_path = os.path.join(MindFormerBook.get_project_path(),
                                                  'configs', 'ner', 'model_config',
                                                  'ner_bert_base_chinese_dense.yaml')
        self.ner_dense_config = AutoConfig.from_pretrained('ner_bert_base_chinese_dense')

        # evaluation and prediction
        self.ner_dense_cluener_checkpoint_dir = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(),
                                                             'ner')
        self.ner_dense_cluener_config_path = os.path.join(MindFormerBook.get_project_path(),
                                                          'configs', 'ner', 'model_config',
                                                          'ner_bert_base_chinese_dense_cluener.yaml')
        self.ner_dense_cluener_config = AutoConfig.from_pretrained('ner_bert_base_chinese_dense_cluener')

        # save path
        self.save_directory = os.path.join(MindFormerBook.get_default_checkpoint_save_folder(), 'ner')
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
        ner_dense_model_a = AutoModel.from_pretrained('ner_bert_base_chinese_dense')
        # input model directory
        ner_dense_model_b = AutoModel.from_pretrained(self.ner_dense_checkpoint_dir)
        # input yaml path
        ner_dense_model_c = AutoModel.from_config(self.ner_dense_config_path)
        # input config
        ner_dense_model_d = AutoModel.from_config(self.ner_dense_config)

        ner_dense_model_a.save_pretrained(self.save_directory, save_name='ner_bert_base_chinese_dense')

        assert isinstance(ner_dense_model_a, BertTokenClassification)
        assert isinstance(ner_dense_model_b, BertTokenClassification)
        assert isinstance(ner_dense_model_c, BertTokenClassification)
        assert isinstance(ner_dense_model_d, BertTokenClassification)

        assert isinstance(ner_dense_model_a, BaseModel)
        assert isinstance(ner_dense_model_b, BaseModel)
        assert isinstance(ner_dense_model_c, BaseModel)
        assert isinstance(ner_dense_model_d, BaseModel)

        # evaluation and prediction test part
        # input model name
        ner_dense_cluener_model_a = AutoModel.from_pretrained('ner_bert_base_chinese_dense_cluener')
        # input model directory
        ner_dense_cluener_model_b = AutoModel.from_pretrained(self.ner_dense_cluener_checkpoint_dir)
        # input yaml path
        ner_dense_cluener_model_c = AutoModel.from_config(self.ner_dense_cluener_config_path)
        # input config
        ner_dense_cluener_model_d = AutoModel.from_config(self.ner_dense_cluener_config)

        ner_dense_cluener_model_a.save_pretrained(self.save_directory, save_name='ner_bert_base_chinese_dense_cluener')

        assert isinstance(ner_dense_cluener_model_a, BertTokenClassification)
        assert isinstance(ner_dense_cluener_model_b, BertTokenClassification)
        assert isinstance(ner_dense_cluener_model_c, BertTokenClassification)
        assert isinstance(ner_dense_cluener_model_d, BertTokenClassification)

        assert isinstance(ner_dense_cluener_model_a, BaseModel)
        assert isinstance(ner_dense_cluener_model_b, BaseModel)
        assert isinstance(ner_dense_cluener_model_c, BaseModel)
        assert isinstance(ner_dense_cluener_model_d, BaseModel)

        BertTokenClassification.show_support_list()
        support_list = BertTokenClassification.get_support_list()
        logger.info(support_list)

        # fine-tuning part
        # input model name, load model and weights
        ner_dense_model_e = BertTokenClassification.from_pretrained('ner_bert_base_chinese_dense')
        # input model directory, load model and weights
        ner_dense_model_f = BertTokenClassification.from_pretrained(self.ner_dense_checkpoint_dir)
        # input config, load model weights
        ner_dense_model_g = BertTokenClassification(self.ner_dense_config)
        # input config, load model without weights
        self.ner_dense_config.checkpoint_name_or_path = None
        ner_dense_model_h = BertTokenClassification(self.ner_dense_config)

        ner_dense_model_e.save_pretrained(self.save_directory, save_name='ner_bert_base_chinese_dense')

        assert isinstance(ner_dense_model_e, BertTokenClassification)
        assert isinstance(ner_dense_model_f, BertTokenClassification)
        assert isinstance(ner_dense_model_g, BertTokenClassification)
        assert isinstance(ner_dense_model_h, BertTokenClassification)

        assert isinstance(ner_dense_model_e, BaseModel)
        assert isinstance(ner_dense_model_f, BaseModel)
        assert isinstance(ner_dense_model_g, BaseModel)
        assert isinstance(ner_dense_model_h, BaseModel)

        # evaluation and prediction test part
        # input model name, load model and weights
        ner_dense_cluener_model_e = BertTokenClassification.from_pretrained('ner_bert_base_chinese_dense_cluener')
        # input model directory, load model and weights
        ner_dense_cluener_model_f = BertTokenClassification.from_pretrained(self.ner_dense_cluener_checkpoint_dir)
        # input config, load model weights
        ner_dense_cluener_model_g = BertTokenClassification(self.ner_dense_cluener_config)
        # input config, load model without weights
        self.ner_dense_cluener_config.checkpoint_name_or_path = None
        ner_dense_cluener_model_h = BertTokenClassification(self.ner_dense_cluener_config)

        ner_dense_cluener_model_e.save_pretrained(self.save_directory, save_name='ner_bert_base_chinese_dense_cluener')

        assert isinstance(ner_dense_cluener_model_e, BertTokenClassification)
        assert isinstance(ner_dense_cluener_model_f, BertTokenClassification)
        assert isinstance(ner_dense_cluener_model_g, BertTokenClassification)
        assert isinstance(ner_dense_cluener_model_h, BertTokenClassification)

        assert isinstance(ner_dense_cluener_model_e, BaseModel)
        assert isinstance(ner_dense_cluener_model_f, BaseModel)
        assert isinstance(ner_dense_cluener_model_g, BaseModel)
        assert isinstance(ner_dense_cluener_model_h, BaseModel)
