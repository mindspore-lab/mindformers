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
Test module for testing yaml tools, MindFormerConfig.

How to run this:
windows:  pytest .\\tests\\ut\\test_yaml.py
linux:  pytest ./tests/ut/test_yaml.py

Note:
    the name of model yaml file should start with model.
    the name of model ckpt file should be same with yaml file.
example:
    clip_vit_b_32.yaml starts with clip,
    and clip_vit_b_32.ckpt is model ckpt file.
'''
import os

from mindformers import XFormerBook
from mindformers.tools import MindFormerConfig, logger


def test_yaml():
    '''
    Feature: MindFormerConfig
    Description: Test to transform yaml file as MindFormerConfig
    Exception: TypeError
    '''
    yaml_path = os.path.join(
        XFormerBook.get_project_path(), 'configs', 'clip', 'model_config', "clip_vit_b_32.yaml")
    yaml_content = MindFormerConfig(yaml_path)

    logger.info(yaml_content)
    logger.info("%s, %s", type(yaml_content), isinstance(yaml_content, dict))
