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
Test Module for testing functions of AutoModel and SwinForImageClassification class

How to run this:
windows:  pytest .\\tests\\st\\test_model\\test_swin_model\\test_swin_model.py
linux:  pytest ./tests/st/test_model/test_swin_model/test_swin_model.py

Note:
    obs path for weights and yaml saving:
        XForme_for_mindspore/swin/swin_base_p4w7.yaml
        XForme_for_mindspore/swin/swin_base_p4w7.ckpt

    self.config is necessary for a model
    SwinForImageClassification amd SwinConfig start with the same prefix "Swin"
"""
import os
import time
# import pytest

from mindformers import MindFormerBook, AutoConfig, AutoModel
from mindformers.models import SwinForImageClassification, BaseModel
from mindformers.tools import logger


# @pytest.mark.level0
# @pytest.mark.platform_arm_ascend_training
# @pytest.mark.platform_x86_ascend_training
# @pytest.mark.env_onecard
class TestModelMethod:
    """A test class for testing Model classes"""
    def setup_method(self):
        """get_input"""
        self.config = AutoConfig.from_pretrained('swin_base_p4w7')

        self.save_directory = os.path.join(MindFormerBook.get_default_checkpoint_save_folder(),
                                           'swin')

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
        model_a = AutoModel.from_pretrained('swin_base_p4w7')

        # input config, load model without weights
        self.config.checkpoint_name_or_path = None
        self.batch_size = 32
        self.embed_dim = 24
        self.config.depths = [1, 1, 2, 1]
        self.config.num_heads = [1, 2, 3, 4]
        model_d = AutoModel.from_config(self.config)

        model_a.save_pretrained(self.save_directory, save_name='swin_base_p4w7')

        # all models are SwinForImageClassification class， and inherited from BaseModel
        assert isinstance(model_a, SwinForImageClassification)
        assert isinstance(model_d, SwinForImageClassification)

        assert isinstance(model_a, BaseModel)
        assert isinstance(model_d, BaseModel)
