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
from typing import Optional, List, Union

from mindspore.train import Callback
from mindspore.nn import TrainOneStepCell, Optimizer, Cell
from mindspore.dataset import GeneratorDataset

from mindformers.dataset import BaseDataset
from mindformers.models import BaseModel, BaseTokenizer

from mindformers.tools.register import MindFormerRegister, \
    MindFormerModuleType, MindFormerConfig
from ..config_args import ConfigArguments
from ..training_args import TrainingArguments
from ..base_trainer import BaseTrainer


@MindFormerRegister.register(MindFormerModuleType.TRAINER)
class MaskedLanguageModelingTrainer(BaseTrainer):
    r"""MaskedLanguageModeling Task For Trainer.
    Args:
        model_name (str): The model name of Task-Trainer. Default: None
    Examples:
        >>> from mindformers import MaskedLanguageModelingTrainer
        >>> def generator():
        >>>     data = np.random.randint(low=0, high=15, size=(128,)).astype(np.int32)
        >>>     input_mask = np.ones_like(data)
        >>>     token_type_id = np.zeros_like(data)
        >>>     next_sentence_lables = np.array([1]).astype(np.int32)
        >>>     masked_lm_positions = np.array([1, 2]).astype(np.int32)
        >>>     masked_lm_ids = np.array([1, 2]).astype(np.int32)
        >>>     masked_lm_weights = np.ones_like(masked_lm_ids)
        >>>     train_data = (data, input_mask, token_type_id, next_sentence_lables,
        ...                   masked_lm_positions, masked_lm_ids, masked_lm_weights)
        >>>     for _ in range(512):
        ...         yield train_data
        >>> dataset = GeneratorDataset(generator, column_names=["input_ids", "input_mask", "segment_ids",
        ...                                                     "next_sentence_labels", "masked_lm_positions",
        ...                                                     "masked_lm_ids", "masked_lm_weights"])
        >>> dataset = dataset.batch(batch_size=16)
        >>> mlm_trainer = MaskedLanguageModelingTrainer(model_name="bert_tiny_uncased")
        >>> mlm_trainer.train(dataset=dataset)
        >>> res = mlm_trainer.predict(input_data = "hello world [MASK]")
    Raises:
        NotImplementedError: If train method or evaluate method or predict method not implemented.
    """

    def __init__(self, model_name: str = None):
        super(MaskedLanguageModelingTrainer, self).__init__("fill_mask", model_name)

    def train(self,
              config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
              network: Optional[Union[Cell, BaseModel]] = None,
              dataset: Optional[Union[BaseDataset, GeneratorDataset]] = None,
              wrapper: Optional[TrainOneStepCell] = None,
              optimizer: Optional[Optimizer] = None,
              callbacks: Optional[Union[Callback, List[Callback]]] = None,
              **kwargs):
        r"""Train task for MaskedLanguageModeling Trainer.
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
            "The MaskedLanguageModeling task does not support evaluate.")

    def predict(self,
                config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
                input_data: Optional[Union[str, list]] = None,
                network: Optional[Union[str, BaseModel]] = None,
                tokenizer: Optional[BaseTokenizer] = None,
                **kwargs):
        """
        Executes the predict of the trainer.

        Args:
            config (Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]]):
                The task config which is used to configure the dataset, the hyper-parameter, optimizer, etc.
                It supports config dict or MindFormerConfig or TrainingArguments or ConfigArguments class.
                Default: None.
            input_data (Optional[Union[Tensor, str, list]]): The predict data. Default: None.
            network (Optional[Union[str, BaseModel]]): The network for trainer. It support model name supported
                or BaseModel class. Supported model name can refer to model support list. For .
                Default: None.
            tokenizer (Optional[BaseTokenizer]): The tokenizer for tokenizing the input text.
                Default: None.

        Returns:
            A list of prediction.

        """
        config = self.set_config(config)
        config.model.model_config.is_training = False

        if input_data is None:
            input_data = config.input_data

        if not isinstance(input_data, (str, list)):
            raise ValueError("Input data's type must be one of "
                             f"[str, list], but got type {type(input_data)}")

        config.model.model_config.batch_size = 1

        max_length = network.config.seq_length if network else config.model.model_config.seq_length

        return self.predict_process(config=config,
                                    input_data=input_data,
                                    task='fill_mask',
                                    network=network,
                                    tokenizer=tokenizer,
                                    max_length=max_length,
                                    padding="max_length",
                                    **kwargs)
