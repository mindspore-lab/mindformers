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
Test Module for testing functions of AutoConfig and VitConfig class

How to run this:
windows:  pytest .\\tests\\st\\test_model\\test_vit_model\\test_vit_config.py
linux:  pytest ./tests/st/test_model/test_vit_model/test_vit_config.py

Note:
    model name and config name should have the same prefix
Example:
    VitModel and VitConfig have the same prefix, Vit
"""
import os

import pytest

from mindformers import MindFormerBook, AutoConfig
from mindformers.models import VitConfig, BaseConfig
from mindformers.tools import logger
from mindspore.nn.transformer.transformer import TransformerOpParallelConfig, TransformerRecomputeConfig
import mindspore.common.dtype as mstype
from mindspore.nn.transformer.moe import default_moe_config

default_recompute_config = TransformerRecomputeConfig()
default_parallel_config = TransformerOpParallelConfig(recompute=default_recompute_config)


# the first method to load model config, AutoConfig
@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_config():
    """
    Feature: AutoConfig, VitConfig
    Description: Test to get config instance by AutoConfig.from_pretrained
    Expectation: TypeError, ValueError
    """
    config_path = os.path.join(MindFormerBook.get_project_path(),
                               'configs', 'vit', 'model_config', "vit_base_p16.yaml")
    save_path = os.path.join(MindFormerBook.get_default_checkpoint_save_folder(),
                             'vit')

    AutoConfig.show_support_list()
    config_a = AutoConfig.from_pretrained('vit_base_p16')      # input a model name
    config_b = AutoConfig.from_pretrained(config_path)  # input a path to .yaml file

    logger.info(config_a)
    logger.info(config_b)

    VitConfig.show_support_list()
    support_list = VitConfig.get_support_list()
    logger.info(support_list)

    config_c = VitConfig.from_pretrained('vit_base_p16')
    config_d = VitConfig.from_pretrained(config_path)
    config_e = VitConfig(
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        drop_rate=0.,
        drop_path_rate=0.,
        use_abs_pos_emb=True,
        attention_dropout_rate=0.,
        use_mean_pooling=True,
        init_values=None,
        hidden_act='gelu',
        post_layernorm_residual=False,
        layernorm_compute_type=mstype.float32,
        softmax_compute_type=mstype.float32,
        param_init_type=mstype.float32,
        loss_type="SoftTargetCrossEntropy",
        parallel_config=default_parallel_config,
        moe_config=default_moe_config,
        batch_size=32,
        image_size=224,
        num_classes=1000
    )

    config_c.save_pretrained()
    config_d.save_pretrained(save_path, "vit_base_p16")
    config_e.save_pretrained()

    logger.info(config_c)
    assert isinstance(config_c, BaseConfig)
    assert isinstance(config_d, BaseConfig)

    assert isinstance(config_a, BaseConfig)
    assert isinstance(config_b, BaseConfig)

    assert isinstance(config_a, VitConfig)
    assert isinstance(config_a, type(config_b))
    assert isinstance(config_b, type(config_c))
