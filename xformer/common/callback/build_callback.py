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
"""Build Callback API."""
import inspect

from mindspore.train import callback

from xformer.tools.register import XFormerRegister, XFormerModuleType


def build_callback(
        *args, config: dict = None, default_args: dict = None,
        module_type: str = 'callback', class_name: str = None, **kwargs):
    """Build callback API."""
    if config is None and class_name is None:
        return None
    if config is not None:
        cfg_callback = config
        if not isinstance(cfg_callback, list):
            return XFormerRegister.get_instance_from_cfg(
                cfg_callback, XFormerModuleType.CALLBACK, default_args=default_args)
        callbacks = []
        for callback_type in cfg_callback:
            callback_op = XFormerRegister.get_instance_from_cfg(
                callback_type, XFormerModuleType.CALLBACK)
            callbacks.append(callback_op)
        return callbacks
    return XFormerRegister.get_instance(module_type, class_name, *args, **kwargs)


def register_ms_cb():
    """ register MindSpore builtin LR class. """
    for module_name in dir(callback):
        if module_name.startswith('__'):
            continue
        cb = getattr(callback, module_name)
        if inspect.isclass(cb):
            XFormerRegister.register_cls(
                cb, XFormerModuleType.CALLBACK)


register_ms_cb()
