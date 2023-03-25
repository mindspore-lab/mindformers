# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Causal Image Modeling Trainer."""
from typing import Optional, List, Union

from mindspore.train import Callback
from mindspore.nn import TrainOneStepCell, Optimizer, Cell
from mindspore.dataset import GeneratorDataset

from mindformers.dataset import BaseDataset
from mindformers.models import BaseModel

from mindformers.tools.register import MindFormerRegister, \
    MindFormerModuleType, MindFormerConfig
from ..config_args import ConfigArguments
from ..training_args import TrainingArguments
from ..base_trainer import BaseTrainer


@MindFormerRegister.register(MindFormerModuleType.TRAINER)
class CausalLanguageModelingTrainer(BaseTrainer):
    r"""CausalLanguageModelingTrainer Task For Trainer.
    Args:
        model_name (str): The model name of Task-Trainer. Default: None
    Examples:
        >>> from mindformers import CausalLanguageModelingTrainer
        >>> gen_trainer = CausalLanguageModelingTrainer(model_name="gpt2")
        >>> gen_trainer.train()
        >>> res = gen_trainer.predict(input_data = "hello world [MASK]")
    Raises:
        NotImplementedError: If train method or evaluate method or predict method not implemented.
    """
    def __init__(self, model_name: str = None):
        super(CausalLanguageModelingTrainer, self).__init__("text_generation", model_name)

    def train(self,
              config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
              network: Optional[Union[Cell, BaseModel]] = None,
              dataset: Optional[Union[BaseDataset, GeneratorDataset]] = None,
              wrapper: Optional[TrainOneStepCell] = None,
              optimizer: Optional[Optimizer] = None,
              callbacks: Optional[Union[Callback, List[Callback]]] = None,
              **kwargs):
        r"""Train task for CausalLanguageModeling Trainer.
        This function is used to train or fine-tune the network.

        The trainer interface is used to quickly start training for general task.
        It also allows users to customize the network, optimizer, dataset, wrapper, callback.

        Args:
            config (Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]]):
                The task config which is used to configure the dataset, the hyper-parameter, optimizer, etc.
                It supports config dict or MindFormerConfig or TrainingArguments or ConfigArguments class.
                Default: None.
            network (Optional[Union[Cell, BaseModel]]): The network for trainer.
                It supports model name or BaseModel or MindSpore Cell class.
                Default: None.
            dataset (Optional[Union[BaseDataset, GeneratorDataset]]): The training dataset.
                It support real dataset path or BaseDateset class or MindSpore Dataset class.
                Default: None.
            optimizer (Optional[Optimizer]): The training network's optimizer. It support Optimizer class of MindSpore.
                Default: None.
            wrapper (Optional[TrainOneStepCell]): Wraps the `network` with the `optimizer`.
                It support TrainOneStepCell class of MindSpore.
                Default: None.
            callbacks (Optional[Union[Callback, List[Callback]]]): The training callback function.
                It support CallBack or CallBack List of MindSpore.
                Default: None.

        Raises:
            NotImplementedError: If wrapper not implemented.
        """
        self.training_process(
            config=config,
            network=network,
            callbacks=callbacks,
            dataset=dataset,
            wrapper=wrapper,
            optimizer=optimizer,
            **kwargs)

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError(
            "The CausalLanguageModeling task does not support evaluate.")

    def predict(self, *args, **kwargs):
        raise NotImplementedError(
            "The CausalLanguageModeling task does not support predict.")
