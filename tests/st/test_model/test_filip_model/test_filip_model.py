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
Test Module for testing functions of AutoModel and ClipModel class

How to run this:
windows:  pytest .\\tests\\st\\test_model\\test_filip_model\\test_filip_model.py
linux:  pytest ./tests/st/test_model/test_filip_model/test_filip_model.py
"""
import os
# import pytest
from mindformers import MindFormerBook, AutoModel
from mindformers.models import BaseModel, FilipModel
from mindformers.tools import logger


class TestModelMethod:
    """A test class for testing Model classes"""
    def setup_method(self):
        """get_input"""
        self.config_path = os.path.join(MindFormerBook.get_project_path(),
                                        'configs', 'filip', 'run_filip_vit_l_14.yaml')

    # the first method to load model, AutoModel
    # @pytest.mark.level0
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
        # input yaml path, load model without weights
        model = AutoModel.from_config(self.config_path)
        assert isinstance(model, FilipModel)
        assert isinstance(model, BaseModel)

    # @pytest.mark.level0
    # @pytest.mark.platform_arm_ascend_training
    # @pytest.mark.env_onecard
    def test_save_model(self):
        """
        Feature: save_pretrained method of FilipModel
        Description: Test to save checkpoint for FilipModel
        Expectation: ValueError, AttributeError
        """
        self.save_directory = os.path.join(MindFormerBook.get_project_path(),
                                           'checkpoint_save', 'filip')
        self.config_path = os.path.join(MindFormerBook.get_project_path(),
                                        'configs', 'filip', 'run_filip_vit_l_14.yaml')

        filip = AutoModel.from_config(self.config_path)
        filip.save_pretrained(self.save_directory, save_name='filip_test')
        new_filip = FilipModel.from_pretrained(self.save_directory)
        assert isinstance(new_filip, FilipModel)
        assert isinstance(new_filip, BaseModel)
