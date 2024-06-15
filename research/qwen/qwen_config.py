# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Qwen Config API."""

from mindformers import LlamaConfig, MindFormerBook
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

__all__ = ['QwenConfig']


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class QwenConfig(LlamaConfig):
    """
    Qwen config class.

    Returns:
        Class, QwenConfig.
    """

    _support_list = MindFormerBook.get_config_support_list()['qwen']
