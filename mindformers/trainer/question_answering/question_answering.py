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
"""
Question Answer Trainer.
"""
from typing import Optional, List, Union
from mindspore.train import Callback
from mindspore.dataset import GeneratorDataset
from mindspore.nn import TrainOneStepCell, Optimizer, Cell

from mindformers.dataset import BaseDataset
from mindformers.models import BaseModel, BaseTokenizer
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerRegister, \
    MindFormerModuleType, MindFormerConfig
from ..base_trainer import BaseTrainer
from ..config_args import ConfigArguments
from ..training_args import TrainingArguments

__all__ = ['QuestionAnsweringTrainer']

@MindFormerRegister.register(MindFormerModuleType.TRAINER)
class QuestionAnsweringTrainer(BaseTrainer):
    r"""QuestionAnswering Task For Trainer.
    Args:
        model_name (str): The model name of Task-Trainer. Default: None
    Examples:
            >>> import numpy as np
            >>> from mindspore.nn import AdamWeightDecay, TrainOneStepCell
            >>> from mindformers.core.lr import build_lr
            >>> from mindformers.trainer import GeneralTaskTrainer
            >>> from mindformers.tools.register import MindFormerConfig
            >>> from mindformers.models import BertForQuestionAnswering, BertConfig
            >>> config = MindFormerConfig("configs/qa/run_qa_bert_base_uncased.yaml")
            >>> #1) use config to train
            >>> cls_task = QuestionAnsweringTrainer(model_name='qa_bert_base_uncased')
            >>> cls_task.train(config=config)
            >>> #2) use instance function to train
    Raises:
        NotImplementedError: If train method or evaluate method or predict method not implemented.
    """

    def __init__(self, model_name: str = None):
        super(QuestionAnsweringTrainer, self).__init__("question_answering", model_name)

    def train(self,
              config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
              network: Optional[Union[Cell, BaseModel]] = None,
              dataset: Optional[Union[BaseDataset, GeneratorDataset]] = None,
              wrapper: Optional[TrainOneStepCell] = None,
              optimizer: Optional[Optimizer] = None,
              callbacks: Optional[Union[Callback, List[Callback]]] = None,
              **kwargs):
        r"""Train task for TokenClassification Trainer.
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

    def evaluate(self,
                 config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
                 network: Optional[Union[Cell, BaseModel]] = None,
                 dataset: Optional[Union[BaseDataset, GeneratorDataset]] = None,
                 callbacks: Optional[Union[Callback, List[Callback]]] = None,
                 compute_metrics: Optional[Union[dict, set]] = None,
                 **kwargs):
        r"""Evaluate task for TokenClassification Trainer.
        This function is used to evaluate the network.

        The trainer interface is used to quickly start training for general task.
        It also allows users to customize the network, dataset, callbacks, compute_metrics.

        Args:
            config (Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]]):
                The task config which is used to configure the dataset, the hyper-parameter, optimizer, etc.
                It supports config dict or MindFormerConfig or TrainingArguments or ConfigArguments class.
                Default: None.
            network (Optional[Union[Cell, BaseModel]]): The network for trainer.
                It supports model name or BaseModel or MindSpore Cell class.
                Default: None.
            dataset (Optional[Union[BaseDataset]]): The evaluate dataset.
                It support real dataset path or BaseDateset class or MindSpore Dataset class.
                Default: None.
            callbacks (Optional[Union[Callback, List[Callback]]]): The eval callback function.
                It support CallBack or CallBack List of MindSpore.
                Default: None.
            compute_metrics (Optional[Union[dict, set]]): The metric of evaluating.
                It support dict or set in MindSpore's Metric class.
                Default: None.
        """
        metric_name = "QA Metric"
        kwargs.setdefault("metric_name", metric_name)
        super().evaluate_process(
            config=config,
            network=network,
            dataset=dataset,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            **kwargs
        )

    def predict(self,
                config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
                input_data: Optional[Union[str, list]] = None,
                network: Optional[Union[Cell, BaseModel]] = None,
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
            network (Optional[Union[Cell, BaseModel]]): The network for trainer.
                It supports model name or BaseModel or MindSpore Cell class.
                Default: None.
            tokenizer (Optional[BaseTokenizer]): The tokenizer for tokenizing the input text.
                Default: None.
        Returns:
            A list of prediction.
        """

        logger.info(".........Build Input Data For Predict..........")
        if input_data is None:
            if config.input_data:
                input_data = config.input_data
            else:
                input_data = "My name is Wolfgang and I live in Berlin - Where do I live?"

        if not isinstance(input_data, (str, list)):
            raise ValueError("Input data's type must be one of [str, list]")

        if isinstance(input_data, list):
            for item in input_data:
                if not isinstance(item, str):
                    raise ValueError("The element of input data list must be str")

        # This is a known issue, you need to specify batch size equal to 1 when creating bert model.
        config.model.model_config.batch_size = 1

        return self.predict_process(config=config,
                                    input_data=input_data,
                                    task='question_answering',
                                    network=network,
                                    tokenizer=tokenizer,
                                    **kwargs)
