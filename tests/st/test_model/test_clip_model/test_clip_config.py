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
Test Module for testing functions of AutoConfig and CLIPConfig class

How to run this:
windows:  pytest .\\tests\\st\\test_model\\test_clip_model\\test_clip_config.py
linux:  pytest ./tests/st/test_model/test_clip_model/test_clip_config.py
"""
import os
from mindformers import MindFormerBook, AutoConfig
from mindformers.models import CLIPConfig, CLIPVisionConfig, CLIPTextConfig, BaseConfig
from mindformers.tools import logger


# the first method to load model config, AutoConfig
def test_config():
    """
    Feature: AutoConfig, CLIPConfig
    Description: Test to get config instance by AutoConfig.from_pretrained
    Expectation: TypeError, ValueError
    """
    model_type = 'clip_vit_b_32'

    config_path = os.path.join(MindFormerBook.get_project_path(),
                               'configs', 'clip', 'run_clip_vit_b_32_pretrain_flickr8k.yaml')
    save_path = os.path.join(MindFormerBook.get_default_checkpoint_save_folder(),
                             'clip')

    AutoConfig.show_support_list()
    config_a = AutoConfig.from_pretrained(model_type)      # input a model name
    config_b = AutoConfig.from_pretrained(config_path)          # input a path to .yaml file

    logger.info(config_a)
    logger.info(config_b)

    CLIPConfig.show_support_list()
    support_list = CLIPConfig.get_support_list()
    logger.info(support_list)

    config_c = CLIPConfig.from_pretrained(model_type)
    config_d = CLIPConfig.from_pretrained(config_path)
    config_e = CLIPConfig(
        CLIPTextConfig(
            hidden_size=512,
            vocab_size=49408,
            max_position_embeddings=77,
            num_hidden_layers=12
        ),
        CLIPVisionConfig(
            hidden_size=768,
            image_size=224,
            patch_size=32,
            num_hidden_layers=12,
        ),
        projection_dim=512
    )

    config_c.save_pretrained()
    config_d.save_pretrained(save_path, model_type)
    config_e.save_pretrained()

    logger.info(config_c)
    assert isinstance(config_c, BaseConfig)
    assert isinstance(config_d, BaseConfig)

    assert isinstance(config_a, BaseConfig)
    assert isinstance(config_b, BaseConfig)

    assert isinstance(config_a, CLIPConfig)
    assert isinstance(config_a, type(config_b))
    assert isinstance(config_b, type(config_c))
