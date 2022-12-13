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
"""Self-Define Vision Mask Policy."""
from mindspore.dataset.transforms import py_transforms
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


__all__ = ['SimMask', 'MaeMask']


@MindFormerRegister.register(MindFormerModuleType.MASK_POLICY)
class SimMask(py_transforms.PyTensorOperation):
    """SimMIM Mask Policy."""


@MindFormerRegister.register(MindFormerModuleType.MASK_POLICY)
class MaeMask(py_transforms.PyTensorOperation):
    """MAE Mask Policy."""
