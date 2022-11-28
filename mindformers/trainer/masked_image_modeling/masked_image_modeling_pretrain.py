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
"""Masked Image Modeling Trainer."""
from typing import Callable, List

from mindformers.trainer.base_trainer import BaseTrainer
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.TRAINER, alias="mim")
class MaskedImageModelingTrainer(BaseTrainer):
    """Masked Image Modeling Trainer."""
    def __init__(self, model_name: str = None):
        super(MaskedImageModelingTrainer, self).__init__(model_name)
        self.model_name = model_name

    def train(self,
              config: dict = None,
              network: Callable = None,
              dataset: Callable = None,
              optimizer: Callable = None,
              callbacks: List[Callable] = None, **kwargs):
        """train for trainer."""
        # 自定义创建模型训练完整过程, 待补充

    def evaluate(self,
                 config: dict = None,
                 network: Callable = None,
                 dataset: Callable = None,
                 callbacks: List[Callable] = None, **kwargs):
        """evaluate for trainer."""
        # 自定义创建模型评估完整过程, 待补充
