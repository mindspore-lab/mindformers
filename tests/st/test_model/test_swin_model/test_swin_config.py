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
Test Module for testing functions of AutoConfig and SwinConfig class

How to run this:
windows:  pytest .\\tests\\st\\test_model\\test_swin_model\\test_swin_config.py
linux:  pytest ./tests/st/test_model/test_swin_model/test_swin_config.py

Note:
    model name and config name should have the same prefix
Example:
    SwinForImageClassification and SwinConfig have the same prefix, Swin
"""
import os
import pytest

import mindspore.common.dtype as mstype

from mindformers.modules.transformer import TransformerOpParallelConfig, TransformerRecomputeConfig
from mindformers.modules.transformer.moe import default_moe_config

from mindformers import MindFormerBook, AutoConfig
from mindformers.models import SwinConfig, BaseConfig
from mindformers.tools import logger

default_recompute_config = TransformerRecomputeConfig()
default_parallel_config = TransformerOpParallelConfig(recompute=default_recompute_config)


# the first method to load model config, AutoConfig
@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_config():
    """
    Feature: AutoConfig, SwinConfig
    Description: Test to get config instance by AutoConfig.from_pretrained
    Expectation: TypeError, ValueError
    """
    config_path = os.path.join(MindFormerBook.get_project_path(),
                               'configs', 'swin', 'run_swin_base_p4w7_224_100ep.yaml')
    save_path = os.path.join(MindFormerBook.get_default_checkpoint_save_folder(),
                             'swin')

    AutoConfig.show_support_list()
    config_a = AutoConfig.from_pretrained('swin_base_p4w7')  # input a model name
    config_b = AutoConfig.from_pretrained(config_path)  # input a path to .yaml file

    logger.info(config_a)
    logger.info(config_b)

    SwinConfig.show_support_list()
    support_list = SwinConfig.get_support_list()
    logger.info(support_list)

    config_c = SwinConfig.from_pretrained('swin_base_p4w7')
    config_d = SwinConfig.from_pretrained(config_path)

    config_e = SwinConfig(
        image_size=224,
        patch_size=4,
        num_labels=1000,
        num_channels=3,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        shift_size=0,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        hidden_dropout_prob=0.,
        attention_probs_dropout_prob=0.,
        drop_path_rate=0.1,
        use_absolute_embeddings=False,
        patch_norm=True,
        patch_type="conv",
        hidden_act='gelu',
        weight_init='normal',
        loss_type="SoftTargetCrossEntropy",
        param_init_type=mstype.float32,
        moe_config=default_moe_config,
        parallel_config=default_parallel_config,
    )

    config_c.save_pretrained()
    config_d.save_pretrained(save_path, "swin_base_p4w7")
    config_e.save_pretrained()

    logger.info(config_c)
    assert isinstance(config_c, BaseConfig)
    assert isinstance(config_d, BaseConfig)

    assert isinstance(config_a, BaseConfig)
    assert isinstance(config_b, BaseConfig)

    assert isinstance(config_a, SwinConfig)
    assert isinstance(config_a, type(config_b))
    assert isinstance(config_b, type(config_c))
