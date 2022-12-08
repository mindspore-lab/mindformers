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
Test Module for testing functions of AutoConfig and ClipConfig class

How to run this:
windows:  pytest .\\tests\\ut\\test_config.py
linux:  pytest ./tests/ut/test_config.py

Note:
    model name and config name should have the same prefix
Example:
    ClipModel and ClipConfig have the same prefix, Clip
"""
import os

from mindformers import MindFormerBook, AutoConfig
from mindformers.models import ClipConfig, ClipVisionConfig, ClipTextConfig, BaseConfig
from mindformers.tools import logger


# the first method to load model config, AutoConfig
def test_auto_config():
    """
    Feature: AutoConfig, from_pretrained
    Description: Test to get config instance by AutoConfig.from_pretrained
    Expectation: TypeError, ValueError
    """
    config_path = os.path.join(MindFormerBook.get_project_path(),
                               'configs', 'clip', 'model_config', "clip_vit_b_32.yaml")

    AutoConfig.show_support_list()
    config_a = AutoConfig.from_pretrained('clip_vit_b_32')      # input a model name
    config_b = AutoConfig.from_pretrained(config_path)          # input a path to .yaml file

    logger.info(config_a)
    logger.info(config_b)

    assert isinstance(config_a, BaseConfig)
    assert isinstance(config_b, BaseConfig)
    return config_a, config_b

# the second method to load model config, ClipConfig (Model's config class)
def test_clip_config():
    """
    Feature: ClipConfig
    Description: Test to get config instance by ClipConfig
    Expectation: None
    """
    config_path = os.path.join(MindFormerBook.get_project_path(),
                               'configs', 'clip', 'model_config', "clip_vit_b_32.yaml")
    save_path = os.path.join(MindFormerBook.get_default_checkpoint_save_folder(),
                             'clip')

    ClipConfig.show_support_list()
    support_list = ClipConfig.get_support_list()
    logger.info(support_list)

    config_c = ClipConfig.from_pretrained('clip_vit_b_32')
    config_d = ClipConfig.from_pretrained(config_path)
    config_e = ClipConfig(
        ClipTextConfig(
            hidden_size=512,
            vocab_size=49408,
            max_position_embeddings=77,
            num_hidden_layers=12
        ),
        ClipVisionConfig(
            hidden_size=768,
            image_size=224,
            patch_size=32,
            num_hidden_layers=12,
        ),
        projection_dim=512
    )

    config_c.save_pretrained()
    config_d.save_pretrained(save_path, "clip_vit_b_32")
    config_e.save_pretrained()

    logger.info(config_c)
    assert isinstance(config_c, BaseConfig)
    assert isinstance(config_d, BaseConfig)
    return config_c

# three configs are all ClipConfig class and inherited from BaseConfig
def test_return():
    """
    Feature: AutoConfig and ClipConfig return the same results
    Description: Test to get the same config instance by ClipConfig and AutoConfig
    Expectation: AssertionError
    """
    config_a, config_b = test_auto_config()
    config_c = test_clip_config()

    assert isinstance(config_a, ClipConfig)
    assert isinstance(config_a, type(config_b))
    assert isinstance(config_b, type(config_c))
