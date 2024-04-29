# Copyright 2024 Huawei Technologies Co., Ltd
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
Test module for testing verifying llama configs content.
How to run this:
pytest tests/st/test_model/test_llama_model/test_config.py
"""
import os
import sys
from glob import glob
from multiprocessing.pool import Pool
import pytest

import mindspore as ms

from mindformers import MindFormerConfig
from mindformers.tools import logger
from mindformers.models.llama.llama import LlamaForCausalLM
from mindformers.models.llama.llama_config import LlamaConfig

ms.set_context(mode=0, device_target='CPU')


def build_model(config):
    """build model from yaml"""
    model_config = MindFormerConfig(os.path.realpath(config))
    model_config = LlamaConfig(**model_config.model.model_config)
    model_config.num_layers = 2
    model_config.checkpoint_name_or_path = None
    try:
        _ = LlamaForCausalLM(model_config)
    except Exception as e:
        logger.error(e)
        logger.error(f"Create Model with {config} Failed.")
        raise AssertionError


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.env_onecard
def test_configs():
    """
    Feature: Model configs
    Description: Test configs by instantiating model
    Expectation: No exception
    """
    configs = list()
    for path in sys.path:
        if path.endswith('/site-packages'):
            config_path = os.path.join(path, 'configs/llama2/*.yaml')
            configs += glob(config_path)
            if configs:
                break
    assert configs

    with Pool(20) as pl:
        _ = list(pl.imap(build_model, configs))
