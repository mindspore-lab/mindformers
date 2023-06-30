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
"""Build Loss API."""
import inspect

from mindspore import nn

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType, MindFormerConfig


def build_loss(
        config: dict = None, default_args: dict = None,
        module_type: str = 'loss', class_name: str = None, **kwargs):
    r"""Build loss For MindFormer.
    Instantiate the loss from MindFormerRegister's registry.

    Args:
        config (dict): The task loss's config. Default: None.
        default_args (dict): The default argument of loss API. Default: None.
        module_type (str): The module type of MindFormerModuleType. Default: 'loss'.
        class_name (str): The class name of loss API. Default: None.

    Return:
        The function instance of loss API.

    Examples:
        >>> from mindformers import build_loss
        >>> loss_config = {'type': 'L1Loss'}
        >>> # 1) use config dict to build loss
        >>> loss_from_config = build_loss(loss_config)
        >>> # 2) use class name to build loss
        >>> loss_class_name = build_loss(class_name='L1Loss')
    """
    if config is None and class_name is None:
        return None
    if config is not None:
        if isinstance(config, dict) and not isinstance(config, MindFormerConfig):
            config = MindFormerConfig(**config)
        return MindFormerRegister.get_instance_from_cfg(
            config, MindFormerModuleType.LOSS, default_args=default_args)
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)


def register_ms_loss():
    """ register MindSpore builtin loss class. """
    for module_name in dir(nn.loss):
        if module_name.startswith('__'):
            continue
        loss = getattr(nn.loss, module_name)
        if inspect.isclass(loss):
            MindFormerRegister.register_cls(
                loss, MindFormerModuleType.LOSS)


register_ms_loss()
