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
"""Build Transforms API."""
import inspect

from mindspore.dataset import transforms as tf
from mindspore.dataset import vision as vs

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


def build_transforms(
        config: dict = None, default_args: dict = None,
        module_type: str = 'transforms', class_name: str = None, **kwargs):
    """Build transforms API."""
    if config is None and class_name is None:
        return None
    if config is not None:
        cfg_transforms = config
        if not isinstance(cfg_transforms, list):
            return MindFormerRegister.get_instance_from_cfg(
                cfg_transforms, MindFormerModuleType.TRANSFORMS, default_args=default_args)
        transforms = []
        for transform in cfg_transforms:
            transform_op = MindFormerRegister.get_instance_from_cfg(
                transform, MindFormerModuleType.TRANSFORMS)
            transforms.append(transform_op)
        return transforms
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)


def register_ms_c_transforms():
    """register MindSpore builtin c_transforms class."""
    for module_name in set(dir(tf.c_transforms) + dir(vs.c_transforms)):
        if module_name.startswith('__'):
            continue

        c_transforms = getattr(tf.c_transforms, module_name, None) \
            if getattr(tf.c_transforms, module_name, None) else getattr(vs.c_transforms, module_name)
        if inspect.isclass(c_transforms):
            class_name = 'C_' + c_transforms.__name__
            MindFormerRegister.register_cls(c_transforms, MindFormerModuleType.TRANSFORMS, alias=class_name)


def register_ms_py_transforms():
    """register MindSpore builtin py_transforms class."""
    for module_name in set(dir(tf.py_transforms) + dir(vs.py_transforms)):
        if module_name.startswith('__'):
            continue

        py_transforms = getattr(tf.py_transforms, module_name, None) \
            if getattr(tf.py_transforms, module_name, None) else getattr(vs.py_transforms, module_name)
        if inspect.isclass(py_transforms):
            class_name = 'PY_' + py_transforms.__name__
            MindFormerRegister.register_cls(py_transforms, MindFormerModuleType.TRANSFORMS, alias=class_name)


register_ms_c_transforms()
register_ms_py_transforms()
