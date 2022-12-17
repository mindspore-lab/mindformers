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
windows:  pytest .\\tests\\st\\test_model\\test_clip_model\\test_clip_model.py
linux:  pytest ./tests/st/test_model/test_clip_model/test_clip_model.py

Note:
    obs path for weights and yaml saving:
        XForme_for_mindspore/clip/clip_vit_b_32.yaml
        XForme_for_mindspore/clip/clip_vit_b_32.ckpt

    self.config is necessary for a model
    ClipModel amd ClipConfig start with the same prefix "Clip"
"""
import os
import pytest
from mindformers import MindFormerBook, AutoConfig, AutoModel
from mindformers.models import ClipModel, BaseModel
from mindformers.tools import logger


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
class TestModelMethod:
    """A test class for testing Model classes"""
    def setup_method(self):
        """get_input"""
        self.checkpoint_dir = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(),
                                           'clip')
        self.config_path = os.path.join(MindFormerBook.get_project_path(),
                                        'configs', 'clip', 'model_config', "clip_vit_b_32.yaml")
        self.config = AutoConfig.from_pretrained('clip_vit_b_32')

        self.checkpoint_path = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(),
                                            'clip', 'clip_vit_b_32.ckpt')
        self.save_directory = os.path.join(MindFormerBook.get_default_checkpoint_save_folder(),
                                           'clip')

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
        # input model name, load model and weights
        model_a = AutoModel.from_pretrained('clip_vit_b_32')
        # input model directory, load model and weights
        model_b = AutoModel.from_pretrained(self.checkpoint_dir)
        # input yaml path, load model without weights
        model_c = AutoModel.from_config(self.config_path)
        # input config, load model without weights
        model_d = AutoModel.from_config(self.config)

        model_a.save_pretrained(self.save_directory, save_name='clip_vit_b_32')

        ClipModel.show_support_list()
        support_list = ClipModel.get_support_list()
        logger.info(support_list)
        # input model name, load model and weights
        model_i = ClipModel.from_pretrained('clip_vit_b_32')
        # input model directory, load model and weights
        model_j = ClipModel.from_pretrained(self.checkpoint_dir)
        # input config, load model weights
        model_k = ClipModel(self.config)
        # input config, load model without weights
        self.config.checkpoint_name_or_path = None
        model_l = ClipModel(self.config)

        model_i.save_pretrained(self.save_directory, save_name='clip_vit_b_32')

        # all models are ClipModel class， and inherited from BaseModel
        assert isinstance(model_i, ClipModel)
        assert isinstance(model_j, ClipModel)
        assert isinstance(model_k, ClipModel)
        assert isinstance(model_l, ClipModel)

        assert isinstance(model_i, BaseModel)
        assert isinstance(model_j, BaseModel)
        assert isinstance(model_k, BaseModel)
        assert isinstance(model_l, BaseModel)

        # all models are ClipModel class， and inherited from BaseModel
        assert isinstance(model_a, ClipModel)
        assert isinstance(model_b, ClipModel)
        assert isinstance(model_c, ClipModel)
        assert isinstance(model_d, ClipModel)

        assert isinstance(model_a, BaseModel)
        assert isinstance(model_b, BaseModel)
        assert isinstance(model_c, BaseModel)
        assert isinstance(model_d, BaseModel)
