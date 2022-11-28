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

'''
Test Module for testing functions of AotoModel and ClipModel class

How to run this:
windows:  pytest .\\tests\\ut\\test_model.py
linux:  pytest ./tests/ut/test_model.py

Note:
    obs path for weights and yaml saving:
        XForme_for_mindspore/clip/clip_vit_b_32.yaml
        XForme_for_mindspore/clip/clip_vit_b_32.clip

    self.config is necessary for a model
    ClipModel amd ClipConfig start with the same prefix "Clip"
'''
import os

from mindformers import XFormerBook, AutoConfig, AutoModel
from mindformers.models import ClipModel, BaseModel


class TestModelMethod:
    '''A test class for testing Model classes'''
    def setup_method(self):
        '''get_input'''
        self.checkpoint_dir = os.path.join(XFormerBook.get_project_path(),
                                           'checkpoint_download', 'clip')
        self.config_path = os.path.join(XFormerBook.get_project_path(),
                                        'configs', 'clip', 'model_config', "clip_vit_b_32.yaml")
        self.config = AutoConfig.from_pretrained('clip_vit_b_32')
        self.checkpoint_path = os.path.join(XFormerBook.get_project_path(),
                                            'checkpoint_download', 'clip', 'clip_vit_b_32.ckpt')
        self.save_directory = os.path.join(XFormerBook.get_project_path(),
                                           'checkpoint_save', 'clip')

    # the first method to load model, AutoModel
    def test_auto_model(self):
        '''
        Feature: AutoModel, from_pretrained, from_config
        Description: Test to get model instance by AutoModel.from_pretrained
                    and AutoModel.from_config
        Exception: TypeError, ValueError, RuntimeError
        '''
        AutoModel.show_support_list()
        # input model name, load model and weights
        model_a = AutoModel.from_pretrained('clip_vit_b_32')
        # input model directory, load model and weights
        model_b = AutoModel.from_pretrained(self.checkpoint_dir)
        # input yaml path, load model without weights
        model_c = AutoModel.from_config(self.config_path)
        # input config, load model without weights
        model_d = AutoModel.from_config(self.config)
        # input yaml path and model name, load model and weights
        model_e = AutoModel.from_config(self.config_path, checkpoint_name_or_path='clip_vit_b_32')
        # input yaml path and weight path, load model and weights
        model_f = AutoModel.from_config(self.config_path, checkpoint_name_or_path=self.checkpoint_path)
        # input config and model name, load model and weights
        model_g = AutoModel.from_config(self.config, checkpoint_name_or_path='clip_vit_b_32')
        # input config and weight path, load model and weights
        model_h = AutoModel.from_config(self.config, checkpoint_name_or_path=self.checkpoint_path)

        # all models are ClipModel class， and inherited from BaseModel
        assert isinstance(model_a, ClipModel)
        assert isinstance(model_b, ClipModel)
        assert isinstance(model_c, ClipModel)
        assert isinstance(model_d, ClipModel)
        assert isinstance(model_e, ClipModel)
        assert isinstance(model_f, ClipModel)
        assert isinstance(model_g, ClipModel)
        assert isinstance(model_h, ClipModel)

        assert isinstance(model_a, BaseModel)
        assert isinstance(model_b, BaseModel)
        assert isinstance(model_c, BaseModel)
        assert isinstance(model_d, BaseModel)
        assert isinstance(model_e, BaseModel)
        assert isinstance(model_f, BaseModel)
        assert isinstance(model_g, BaseModel)
        assert isinstance(model_h, BaseModel)

    # the second method to load model, ClipModel
    def test_clip_model(self):
        '''
        Feature: ClipModel, from_pretrained, input config
        Description: Test to get model instance by ClipModel.from_pretrained
                    and input config
        Exception: TypeError, ValueError, RuntimeError
        '''
        ClipModel.show_support_list()
        # input model name, load model and weights
        model_i = ClipModel.from_pretrained('clip_vit_b_32')
        # input model directory, loda model and weights
        model_j = ClipModel.from_pretrained(self.checkpoint_dir)
        # input config, load model without weights
        model_k = ClipModel(self.config)
        # input config and model name, load model and weights
        model_l = ClipModel(self.config, checkpoint_name_or_path='clip_vit_b_32')
        # input config and weight path, load model and weights
        model_m = ClipModel(self.config, checkpoint_name_or_path=self.checkpoint_path)

        # all models are ClipModel class， and inherited from BaseModel
        assert isinstance(model_i, ClipModel)
        assert isinstance(model_j, ClipModel)
        assert isinstance(model_k, ClipModel)
        assert isinstance(model_l, ClipModel)
        assert isinstance(model_m, ClipModel)

        assert isinstance(model_i, BaseModel)
        assert isinstance(model_j, BaseModel)
        assert isinstance(model_k, BaseModel)
        assert isinstance(model_l, BaseModel)
        assert isinstance(model_m, BaseModel)

    def test_save_model(self):
        '''
        Feature: save_pretrained method of ClipModel
        Description: Test to save checkpoint for ClipModel
        Exception: ValueError, AttributeError
        '''
        model_a = AutoModel.from_pretrained('clip_vit_b_32')
        model_i = ClipModel.from_pretrained('clip_vit_b_32')

        model_a.save_pretrained(self.save_directory, save_name='clip_vit_b_32')
        model_i.save_pretrained(self.save_directory, save_name='clip_vit_b_32')
