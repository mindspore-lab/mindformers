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
import os
import time
from typing import Optional, List, Union

import numpy as np
from mindspore import Model
from mindspore.train import Callback
from mindspore.nn import TrainOneStepCell, Optimizer, Cell
from mindspore.dataset import GeneratorDataset

from mindformers.dataset import BaseDataset
from mindformers.models import BaseModel, BaseTokenizer
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType, MindFormerConfig
from mindformers.core import build_metric

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

    def evaluate(self,
                 config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
                 network: Optional[Union[Cell, BaseModel]] = None,
                 dataset: Optional[Union[BaseDataset, GeneratorDataset]] = None,
                 callbacks: Optional[Union[Callback, List[Callback]]] = None,
                 compute_metrics: Optional[Union[dict, set]] = None,
                 **kwargs):
        r"""Evaluate task for CausalLanguageModeling Trainer.
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
                It supports real dataset path or BaseDateset class or MindSpore Dataset class.
                Default: None.
            callbacks (Optional[Union[Callback, List[Callback]]]): The eval callback function.
                It supports CallBack or CallBack List of MindSpore.
                Default: None.
            compute_metrics (Optional[Union[dict, set]]): The metric of evaluating.
                It supports dict or set in MindSpore's Metric class.
                Default: None.
        """
        metric_name = "Text Generation Metric"
        kwargs.setdefault("metric_name", metric_name)

        is_enhanced_encoder = config.model.model_config.is_enhanced_encoder
        if is_enhanced_encoder:
            self.generate_evaluate(
                config,
                network=network,
                dataset=dataset,
                compute_metrics=compute_metrics,
                **kwargs)
        else:
            self.evaluate_process(
                config=config,
                network=network,
                dataset=dataset,
                callbacks=callbacks,
                compute_metrics=compute_metrics,
                **kwargs)

    def generate_evaluate(self,
                          config,
                          network=None,
                          dataset=None,
                          compute_metrics=None,
                          **kwargs):
        r"""Evaluate the text generate task. Return metrics with Rouge-1, Rouge-2, Rouge-l and BLEU. """
        metric_name = kwargs.get("metric_name")
        is_full_config = kwargs.get("is_full_config", False)
        config = self.set_config(config, is_full_config)

        # build dataset
        logger.info(".........Build Dataset For Evaluate..........")
        if dataset is None:
            dataset = self.create_eval_dataset()
        self.set_eval_dataset(dataset)

        # build network
        if network is None:
            network = self.create_network(default_args={"parallel_config": config.parallel_config,
                                                        "moe_config": config.moe_config})
        self.set_network(network, is_train=False)

        self.count_parameters()

        # build metric
        logger.info(".........Build Compute Metrics For Evaluate..........")
        if compute_metrics is None:
            compute_metrics = build_metric(config.metric)

        logger.info(".........Starting Init Evaluate Model..........")
        model = Model(network, eval_network=network)

        logger.info('.........Starting Evaluate Model..........')
        # generate config
        top_p = config.model.top_p
        top_k = config.model.top_k
        max_decode_length = config.model.max_decode_length

        total_tokens_num = 0
        total_time = 0.0001
        pad_token_id = self.config.model.model_config.pad_token_id
        for i, inputs in enumerate(dataset.create_dict_iterator()):
            input_ids = inputs['input_ids'].asnumpy()
            labels = inputs['label'].asnumpy()

            valid_length_each_example = []
            for j in range(input_ids.shape[0]):
                # As the nonzero returns the index and we need length
                valid_length_each_example.append(np.max(np.argwhere(input_ids[j] != pad_token_id)) + 1)
            valid_length_each_example = np.array(valid_length_each_example)

            start_time = time.time()
            outputs = model.predict_network.generate(input_ids, max_length=max_decode_length,
                                                     top_p=top_p, top_k=top_k)  # List[numpy]
            outputs = np.array(outputs)
            output_ids = []
            for j in range(input_ids.shape[0]):
                output_ids.append(outputs[j, int(valid_length_each_example[j]):].astype(np.int32))
            end_time = time.time()
            avg_cost_time = (end_time - start_time) / input_ids.shape[0]

            tokens_num = 0
            for batch_index in range(len(output_ids)):
                tokens_num += output_ids[batch_index].shape[0]
            if i != 0:
                total_tokens_num += tokens_num
                total_time += end_time - start_time

            logger.info('Epoch %s Finished, cost time %s,  every example cost time is %s, '
                        'generate speed: %s tokens/s, avg speed: %s tokens/s',
                        i + 1, end_time - start_time, avg_cost_time,
                        tokens_num / (end_time - start_time), total_tokens_num / total_time)
            compute_metrics.update(output_ids, labels)

        output = compute_metrics.eval()
        logger.info('metric: %s \n'
                    'rouge-1: %s \n'
                    'rouge-2: %s \n'
                    'rouge-l: %s \n'
                    'bleu-4:  %s ',
                    metric_name, output["rouge-1"], output["rouge-2"], output["rouge-l"], output["bleu-4"])

        logger.info('...........Evaluate Over!...............')

    def predict(self,
                config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
                input_data: Optional[Union[str, list, GeneratorDataset]] = None,
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
            input_data (Optional[Union[Tensor, str, list]]): The predict data. It supports 1) a text string to be
                translated, 1) a file name where each line is a text to be translated  and 3) a generator dataset.
                Default: None.
            network (Optional[Union[Cell, BaseModel]]): The network for trainer.
                It supports model name or BaseModel or MindSpore Cell class.
                Default: None.
            tokenizer (Optional[BaseTokenizer]): The tokenizer for tokenizing the input text.
                Default: None.
        Returns:
            A list of prediction.

        """
        if input_data is None:
            input_data = config.input_data

        if not isinstance(input_data, (str, list, GeneratorDataset)):
            raise ValueError("Input data's type must be one of "
                             f"[str, list, GeneratorDataset], but got type {type(input_data)}")

        if isinstance(input_data, str) and os.path.isfile(input_data):
            with open(input_data, 'r') as fp:
                input_data_list = []
                for line in fp:
                    input_data_list.extend(line)
            input_data = input_data_list

        return self.predict_process(config=config,
                                    input_data=input_data,
                                    task='text_generation',
                                    network=network,
                                    tokenizer=tokenizer,
                                    **kwargs)
