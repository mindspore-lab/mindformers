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
from .general_task_trainer import GeneralTaskTrainer
from .multi_modal_to_text_generation import MultiModalToTextGenerationTrainer
from .causal_language_modeling import CausalLanguageModelingTrainer
from .trainer import Trainer
from .training_args import TrainingArguments
from .base_trainer import BaseTrainer
from .build_trainer import build_trainer

__all__ = ['Trainer', 'TrainingArguments']
