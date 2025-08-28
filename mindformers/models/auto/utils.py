# Copyright 2025 Huawei Technologies Co., Ltd
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
"""process logic of auto class"""
import os
import shutil
from mindformers.tools.logger import logger
from mindformers.mindformer_book import MindFormerBook


def get_default_yaml_file(model_name):
    default_yaml_file = ""
    for model_dict in MindFormerBook.get_trainer_support_task_list().values():
        if model_name in model_dict:
            default_yaml_file = model_dict.get(model_name)
            break
    return default_yaml_file


def set_default_yaml_file(yaml_name, yaml_file):
    if not os.path.exists(yaml_file):
        default_yaml_file = get_default_yaml_file(yaml_name)
        if os.path.realpath(default_yaml_file) and os.path.exists(default_yaml_file):
            shutil.copy(default_yaml_file, yaml_file)
            logger.info("default yaml config in %s is used.", yaml_file)
        else:
            raise FileNotFoundError(f'default yaml file path must be correct, but get {default_yaml_file}')
