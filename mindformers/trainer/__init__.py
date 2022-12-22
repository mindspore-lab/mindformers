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
"""MindFormers Trainer API."""
from .config_args import *
from .image_classification import *
from .masked_image_modeling import *
from .masked_language_modeling import *
from .general_task_trainer import *
from .contrastive_language_image_pretrain import *
from .translation import *
from .trainer import Trainer
from .base_trainer import BaseTrainer
from .build_trainer import build_trainer


__all__ = ['BaseTrainer', 'build_trainer', 'Trainer']
__all__.extend(config_args.__all__)
__all__.extend(image_classification.__all__)
__all__.extend(masked_image_modeling.__all__)
__all__.extend(masked_language_modeling.__all__)
__all__.extend(general_task_trainer.__all__)
__all__.extend(contrastive_language_image_pretrain.__all__)
__all__.extend(translation.__all__)
