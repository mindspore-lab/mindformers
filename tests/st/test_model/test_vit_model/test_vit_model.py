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
Test Module for testing functions of AutoModel and ViTForImageClassification class

How to run this:
windows:  pytest .\\tests\\st\\test_model\\test_vit_model\\test_vit_model.py
linux:  pytest ./tests/st/test_model/test_vit_model/test_vit_model.py

Note:
    obs path for weights and yaml saving:
        XForme_for_mindspore/vit/vit_base_p16.yaml
        XForme_for_mindspore/vit/vit_base_p16.ckpt

    self.config is necessary for a model
    ViTForImageClassification amd ViTConfig start with the same prefix "Vit"
"""
import os
import time
# import pytest

from mindformers import MindFormerBook, AutoConfig, AutoModel
from mindformers.models import BaseModel, ViTForImageClassification
from mindformers.tools import logger


# @pytest.mark.level0
# @pytest.mark.platform_x86_ascend_training
# @pytest.mark.platform_arm_ascend_training
# @pytest.mark.env_onecard
class TestModelMethod:
    """A test class for testing Model classes"""
    def setup_method(self):
        """get_input"""
        self.config = AutoConfig.from_pretrained('vit_base_p16')

        self.save_directory = os.path.join(MindFormerBook.get_default_checkpoint_save_folder(),
                                           'vit')

    # the first method to load model, AutoModel

    def test_auto_model(self):
        """
        Feature: AutoModel, from_pretrained, from_config
        Description: Test to get model instance by AutoModel.from_pretrained
                    and AutoModel.from_config
        Expectation: TypeError, ValueError, RuntimeError
        """
        time.sleep(5)

        AutoModel.show_support_list()
        support_list = AutoModel.get_support_list()
        logger.info(support_list)

        # input model name, load model and weights
        model_a = AutoModel.from_pretrained('vit_base_p16')

        # input config, load model without weights
        self.config.checkpoint_name_or_path = None
        self.config.embed_dim = 192
        self.config.depths = 12
        self.config.num_heads = 3
        model_d = AutoModel.from_config(self.config)

        model_a.save_pretrained(self.save_directory, save_name='vit_base_p16')

        # all models are ViTForImageClassification class， and inherited from BaseModel
        assert isinstance(model_a, ViTForImageClassification)
        assert isinstance(model_d, ViTForImageClassification)

        assert isinstance(model_a, BaseModel)
        assert isinstance(model_d, BaseModel)
