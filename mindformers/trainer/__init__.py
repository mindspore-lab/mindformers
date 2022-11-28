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

from .build_trainer import build_trainer
from .trainer import Trainer
from .base_trainer import BaseTrainer
from .utils import check_runner_config, check_keywords_in_name
from .image_classification import ImageClassificationTrainer
from .masked_image_modeling import MaskedImageModelingTrainer
