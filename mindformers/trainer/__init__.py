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
from .image_to_text_retrieval import *
from .image_to_text_generation import *
from .translation import *
from .text_classfication import *
from .token_classification import *
from .question_answering import *
from .causal_language_modeling import *
from .trainer import Trainer
from .training_args import TrainingArguments
from .base_trainer import BaseTrainer
from .build_trainer import build_trainer


__all__ = ['BaseTrainer', 'Trainer', 'TrainingArguments']
__all__.extend(config_args.__all__)
__all__.extend(image_classification.__all__)
__all__.extend(masked_image_modeling.__all__)
__all__.extend(masked_language_modeling.__all__)
__all__.extend(general_task_trainer.__all__)
__all__.extend(contrastive_language_image_pretrain.__all__)
__all__.extend(image_to_text_retrieval.__all__)
__all__.extend(image_to_text_generation.__all__)
__all__.extend(translation.__all__)
__all__.extend(text_classfication.__all__)
__all__.extend(token_classification.__all__)
__all__.extend(question_answering.__all__)
__all__.extend(causal_language_modeling.__all__)
