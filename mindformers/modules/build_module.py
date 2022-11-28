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
"""Build Module API."""
import inspect

from mindspore.nn import transformer

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


def build_module(
        config: dict = None, default_args: dict = None,
        module_type: str = 'modules', class_name: str = None, **kwargs):
    """Build module API."""
    if config is None and class_name is None:
        return None
    if config is not None:
        return MindFormerRegister.get_instance_from_cfg(
            config, MindFormerModuleType.MODULES, default_args=default_args)
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)


def register_ms_transformer():
    """ register MindSpore builtin mindformers class. """
    for module_name in dir(transformer):
        if module_name.startswith('__'):
            continue
        ms_transformer = getattr(transformer, module_name)
        if inspect.isclass(ms_transformer):
            MindFormerRegister.register_cls(
                ms_transformer, MindFormerModuleType.MODULES)


register_ms_transformer()
