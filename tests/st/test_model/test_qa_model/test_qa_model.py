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
Test Module for testing functions of AutoModel and BertForQuestionAnswering class

How to run this:
windows:  pytest .\\tests\\st\\test_model\\test_qa_model\\test_qa_model.py
linux:  pytest ./tests/st/test_model/test_qa_model/test_qa_model.py
"""
import os
# import pytest
from mindformers import MindFormerBook, AutoModel, AutoConfig
from mindformers.models import BertForQuestionAnswering, BaseModel
from mindformers.tools import logger


class TestModelMethod:
    """A test class for testing Model classes"""
    def setup_method(self):
        """get_input"""
        # fine-tuning
        self.qa_checkpoint_dir = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(),
                                              'qa')
        self.qa_config_path = os.path.join(MindFormerBook.get_project_path(),
                                           'configs', 'qa', 'run_qa_bert_base_uncased.yaml')
        self.qa_config = AutoConfig.from_pretrained('qa_bert_base_uncased')

        # evaluation and prediction
        self.qa_squad_checkpoint_dir = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(),
                                                    'qa')
        self.qa_squad_config_path = os.path.join(MindFormerBook.get_project_path(),
                                                 'configs', 'qa', 'run_qa_bert_base_uncased.yaml')
        self.qa_squad_config = AutoConfig.from_pretrained('qa_bert_base_uncased_squad')

        # save path
        self.save_directory = os.path.join(MindFormerBook.get_default_checkpoint_save_folder(), 'qa')
    # the first method to load model, AutoModel
    # @pytest.mark.level0
    # @pytest.mark.platform_x86_ascend_training
    # @pytest.mark.platform_arm_ascend_training
    # @pytest.mark.env_onecard
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
        qa_model_a = AutoModel.from_pretrained('qa_bert_base_uncased')
        # # input model directory
        # qa_model_b = AutoModel.from_pretrained(self.qa_checkpoint_dir)
        # # input yaml path
        # qa_model_c = AutoModel.from_config(self.qa_config_path)
        # # input config
        # qa_model_d = AutoModel.from_config(self.qa_config)

        qa_model_a.save_pretrained(self.save_directory, save_name='qa_bert_base_uncased')

        assert isinstance(qa_model_a, BertForQuestionAnswering)
        # assert isinstance(qa_model_b, BertForQuestionAnswering)
        # assert isinstance(qa_model_c, BertForQuestionAnswering)
        # assert isinstance(qa_model_d, BertForQuestionAnswering)

        assert isinstance(qa_model_a, BaseModel)
        # assert isinstance(qa_model_b, BaseModel)
        # assert isinstance(qa_model_c, BaseModel)
        # assert isinstance(qa_model_d, BaseModel)

        # evaluation and prediction test part
        # input model name
        qa_squad_model_a = AutoModel.from_pretrained('qa_bert_base_uncased_squad')
        # # input model directory
        # qa_squad_model_b = AutoModel.from_pretrained(self.qa_squad_checkpoint_dir)
        # # input yaml path
        # qa_squad_model_c = AutoModel.from_config(self.qa_squad_config_path)
        # # input config
        # qa_squad_model_d = AutoModel.from_config(self.qa_squad_config)

        qa_squad_model_a.save_pretrained(self.save_directory, save_name='qa_bert_base_uncased_squad')

        assert isinstance(qa_squad_model_a, BertForQuestionAnswering)
        # assert isinstance(qa_squad_model_b, BertForQuestionAnswering)
        # assert isinstance(qa_squad_model_c, BertForQuestionAnswering)
        # assert isinstance(qa_squad_model_d, BertForQuestionAnswering)

        # assert isinstance(qa_squad_model_a, BaseModel)
        # assert isinstance(qa_squad_model_b, BaseModel)
        # assert isinstance(qa_squad_model_c, BaseModel)
        # assert isinstance(qa_squad_model_d, BaseModel)

        # BertForQuestionAnswering.show_support_list()
        # support_list = BertForQuestionAnswering.get_support_list()
        # logger.info(support_list)

        # fine-tuning part
        # input model name, load model and weights
        # qa_model_e = BertForQuestionAnswering.from_pretrained('qa_bert_base_uncased')
        # # input model directory, load model and weights
        # qa_model_f = BertForQuestionAnswering.from_pretrained(self.qa_checkpoint_dir)
        # # input config, load model weights
        # qa_model_g = BertForQuestionAnswering(self.qa_config)
        # # input config, load model without weights
        # self.qa_config.checkpoint_name_or_path = None
        # qa_model_h = BertForQuestionAnswering(self.qa_config)

        # qa_model_e.save_pretrained(self.save_directory, save_name='qa_bert_base_uncased')

        # assert isinstance(qa_model_e, BertForQuestionAnswering)
        # assert isinstance(qa_model_f, BertForQuestionAnswering)
        # assert isinstance(qa_model_g, BertForQuestionAnswering)
        # assert isinstance(qa_model_h, BertForQuestionAnswering)

        # assert isinstance(qa_model_e, BaseModel)
        # assert isinstance(qa_model_f, BaseModel)
        # assert isinstance(qa_model_g, BaseModel)
        # assert isinstance(qa_model_h, BaseModel)

        # evaluation and prediction test part
        # input model name, load model and weights
        # qa_squad_model_e = BertForQuestionAnswering.from_pretrained('qa_bert_base_uncased_squad')
        # # input model directory, load model and weights
        # qa_squad_model_f = BertForQuestionAnswering.from_pretrained(self.qa_squad_checkpoint_dir)
        # # input config, load model weights
        # qa_squad_model_g = BertForQuestionAnswering(self.qa_squad_config)
        # # input config, load model without weights
        # self.qa_squad_config.checkpoint_name_or_path = None
        # qa_squad_model_h = BertForQuestionAnswering(self.qa_squad_config)

        # qa_squad_model_e.save_pretrained(self.save_directory, save_name='qa_bert_base_uncased_squad')

        # assert isinstance(qa_squad_model_e, BertForQuestionAnswering)
        # assert isinstance(qa_squad_model_f, BertForQuestionAnswering)
        # assert isinstance(qa_squad_model_g, BertForQuestionAnswering)
        # assert isinstance(qa_squad_model_h, BertForQuestionAnswering)
        #
        # assert isinstance(qa_squad_model_e, BaseModel)
        # assert isinstance(qa_squad_model_f, BaseModel)
        # assert isinstance(qa_squad_model_g, BaseModel)
        # assert isinstance(qa_squad_model_h, BaseModel)
