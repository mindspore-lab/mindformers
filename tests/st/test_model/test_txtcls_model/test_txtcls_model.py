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
Test Module for testing functions of AutoModel and BertForMultipleChoice class

How to run this:
windows:  pytest .\\tests\\st\\test_model\\test_txtcls_model\\test_txtcls_model.py
linux:  pytest ./tests/st/test_model/test_txtcls_model/test_txtcls_model.py
"""
import os
# import pytest
from mindformers import MindFormerBook, AutoModel, AutoConfig
from mindformers.models import BertForMultipleChoice
from mindformers.tools import logger


# @pytest.mark.level0
# @pytest.mark.platform_x86_cpu
# @pytest.mark.env_onecard
class TestModelMethod:
    """A test class for testing Model classes"""
    def setup_method(self):
        """get_input"""
        # fine-tuning
        self.txtcls_checkpoint_dir = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(),
                                                  'txtcls')
        self.txtcls_config_path = os.path.join(MindFormerBook.get_project_path(),
                                               'configs', 'txtcls', 'run_txtcls_bert_base_uncased.yaml')
        self.txtcls_config = AutoConfig.from_pretrained('txtcls_bert_base_uncased')

        # evaluation and prediction
        self.txtcls_mnli_checkpoint_dir = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(),
                                                       'txtcls')
        self.txtcls_mnli_config_path = os.path.join(MindFormerBook.get_project_path(),
                                                    'configs', 'txtcls', 'run_txtcls_bert_base_uncased_mnli.yaml')
        self.txtcls_mnli_config = AutoConfig.from_pretrained('txtcls_bert_base_uncased_mnli')

        # save path
        self.save_directory = os.path.join(MindFormerBook.get_default_checkpoint_save_folder(), 'txtcls')

    # the first method to load model, AutoModel
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
        txtcls_model_a = AutoModel.from_pretrained('txtcls_bert_base_uncased')
        # # input model directory
        # txtcls_model_b = AutoModel.from_pretrained(self.txtcls_checkpoint_dir)
        # # input yaml path
        # txtcls_model_c = AutoModel.from_config(self.txtcls_config_path)
        # # input config
        # txtcls_model_d = AutoModel.from_config(self.txtcls_config)

        txtcls_model_a.save_pretrained(self.save_directory, save_name='txtcls_bert_base_uncased')

        # assert isinstance(txtcls_model_a, BertForMultipleChoice)
        # assert isinstance(txtcls_model_b, BertForMultipleChoice)
        # assert isinstance(txtcls_model_c, BertForMultipleChoice)
        # assert isinstance(txtcls_model_d, BertForMultipleChoice)
        #
        # assert isinstance(txtcls_model_a, BaseModel)
        # assert isinstance(txtcls_model_b, BaseModel)
        # assert isinstance(txtcls_model_c, BaseModel)
        # assert isinstance(txtcls_model_d, BaseModel)

        # evaluation and prediction test part
        # input model name
        txtcls_mnli_model_a = AutoModel.from_pretrained('txtcls_bert_base_uncased_mnli')
        # # input model directory
        # txtcls_mnli_model_b = AutoModel.from_pretrained(self.txtcls_mnli_checkpoint_dir)
        # # input yaml path
        # txtcls_mnli_model_c = AutoModel.from_config(self.txtcls_mnli_config_path)
        # # input config
        # txtcls_mnli_model_d = AutoModel.from_config(self.txtcls_mnli_config)

        txtcls_mnli_model_a.save_pretrained(self.save_directory, save_name='txtcls_bert_base_uncased_mnli')

        assert isinstance(txtcls_mnli_model_a, BertForMultipleChoice)
        # assert isinstance(txtcls_mnli_model_b, BertForMultipleChoice)
        # assert isinstance(txtcls_mnli_model_c, BertForMultipleChoice)
        # assert isinstance(txtcls_mnli_model_d, BertForMultipleChoice)
        #
        # assert isinstance(txtcls_mnli_model_a, BaseModel)
        # assert isinstance(txtcls_mnli_model_b, BaseModel)
        # assert isinstance(txtcls_mnli_model_c, BaseModel)
        # assert isinstance(txtcls_mnli_model_d, BaseModel)

        # BertForMultipleChoice.show_support_list()
        # support_list = BertForMultipleChoice.get_support_list()
        # logger.info(support_list)

        # fine-tuning part
        # input model name, load model and weights
        # txtcls_model_e = BertForMultipleChoice.from_pretrained('txtcls_bert_base_uncased')
        # # input model directory, load model and weights
        # txtcls_model_f = BertForMultipleChoice.from_pretrained(self.txtcls_checkpoint_dir)
        # # input config, load model weights
        # txtcls_model_g = BertForMultipleChoice(self.txtcls_config)
        # # input config, load model without weights
        # self.txtcls_config.checkpoint_name_or_path = None
        # txtcls_model_h = BertForMultipleChoice(self.txtcls_config)

        # txtcls_model_e.save_pretrained(self.save_directory, save_name='txtcls_bert_base_uncased')

        # assert isinstance(txtcls_model_e, BertForMultipleChoice)
        # assert isinstance(txtcls_model_f, BertForMultipleChoice)
        # assert isinstance(txtcls_model_g, BertForMultipleChoice)
        # assert isinstance(txtcls_model_h, BertForMultipleChoice)
        #
        # assert isinstance(txtcls_model_e, BaseModel)
        # assert isinstance(txtcls_model_f, BaseModel)
        # assert isinstance(txtcls_model_g, BaseModel)
        # assert isinstance(txtcls_model_h, BaseModel)

        # evaluation and prediction test part
        # input model name, load model and weights
        # txtcls_mnli_model_e = BertForMultipleChoice.from_pretrained('txtcls_bert_base_uncased_mnli')
        # # input model directory, load model and weights
        # txtcls_mnli_model_f = BertForMultipleChoice.from_pretrained(self.txtcls_mnli_checkpoint_dir)
        # # input config, load model weights
        # txtcls_mnli_model_g = BertForMultipleChoice(self.txtcls_mnli_config)
        # # input config, load model without weights
        # self.txtcls_mnli_config.checkpoint_name_or_path = None
        # txtcls_mnli_model_h = BertForMultipleChoice(self.txtcls_mnli_config)
        #
        # txtcls_mnli_model_e.save_pretrained(self.save_directory, save_name='txtcls_bert_base_uncased_mnli')

        # assert isinstance(txtcls_mnli_model_e, BertForMultipleChoice)
        # assert isinstance(txtcls_mnli_model_f, BertForMultipleChoice)
        # assert isinstance(txtcls_mnli_model_g, BertForMultipleChoice)
        # assert isinstance(txtcls_mnli_model_h, BertForMultipleChoice)
        #
        # assert isinstance(txtcls_mnli_model_e, BaseModel)
        # assert isinstance(txtcls_mnli_model_f, BaseModel)
        # assert isinstance(txtcls_mnli_model_g, BaseModel)
        # assert isinstance(txtcls_mnli_model_h, BaseModel)
