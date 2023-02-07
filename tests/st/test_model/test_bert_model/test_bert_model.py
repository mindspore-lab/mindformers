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
Test Module for testing functions of AutoModel and ClipModel class

How to run this:
windows:  pytest .\\tests\\st\\test_model\\test_bert_model\\test_bert_model.py
linux:  pytest ./tests/st/test_model/test_bert_model/test_bert_model.py
"""
import os
import pytest
from mindformers import MindFormerBook, AutoModel
from mindformers.models import BaseModel
from mindformers.tools import logger


class TestModelMethod:
    """A test class for testing Model classes"""
    def setup_method(self):
        """get_input"""
        self.config_path = os.path.join(MindFormerBook.get_project_path(),
                                        'configs', 'bert', 'model_config', "bert_base_uncased.yaml")

    # the first method to load model, AutoModel
    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_auto_model(self):
        """
        Feature: AutoModel, from_pretrained, from_config
        Description: Test to get model instance by AutoModel.from_pretrained
                    and AutoModel.from_config
        Expectation: TypeError, ValueError, RuntimeError
        """
        AutoModel.show_support_list()
        self.config_path = os.path.join(MindFormerBook.get_project_path(),
                                        'configs', 'bert', 'model_config', "bert_base_uncased.yaml")
        support_list = AutoModel.get_support_list()
        logger.info(support_list)
        # input yaml path, load model without weights
        model = AutoModel.from_config(self.config_path)
        # assert isinstance(model, BertForPreTraining)
        assert isinstance(model, BaseModel)
