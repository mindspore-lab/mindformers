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
"""Build Wrapper."""


import inspect

import mindspore as ms
from mindspore import nn, Tensor

from xformer.tools.register import XFormerRegister, XFormerModuleType


def build_wrapper(*args, config: dict = None, default_args: dict = None,
                  module_type: str = 'wrapper', class_name: str = None, **kwargs):
    """ Build Wrapper For XFormer. """
    if config is None and class_name is None:
        return None
    if config is not None:
        if config.scale_sense is not None:
            if not isinstance(config.scale_sense, int) and config.scale_sense.type is not None:
                config.scale_sense = build_wrapper(config.scale_sense)
            else:
                config.scale_sense = Tensor(config.scale_sense, ms.float32)
        return XFormerRegister.get_instance_from_cfg(
            config, XFormerModuleType.WRAPPER, default_args=default_args)
    return XFormerRegister.get_instance(module_type, class_name, *args, **kwargs)


def register_ms_wrap():
    """ register MindSpore builtin wrapper class. """
    for module_name in dir(nn.wrap):
        if module_name.startswith('__'):
            continue
        wrap = getattr(nn.wrap, module_name)
        if inspect.isclass(wrap):
            XFormerRegister.register_cls(
                wrap, XFormerModuleType.WRAPPER)


register_ms_wrap()
