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
"""Text Classification Trainer."""
from typing import Optional, List, Union
from mindspore.train.model import Model
from mindspore.train import Callback
from mindspore.nn import TrainOneStepCell, Optimizer

from mindformers.common.callback import build_callback
from mindformers.common.metric import build_metric
from mindformers.dataset import build_dataset, check_dataset_config, BaseDataset
from mindformers.models import build_model, BaseModel, BaseTokenizer, BertTokenizer
from mindformers.common.lr import build_lr
from mindformers.common.optim import build_optim
from mindformers.wrapper import build_wrapper
from mindformers.tools.logger import logger
from mindformers.tools.utils import count_params
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.pipeline import pipeline
from ..config_args import ConfigArguments
from ..base_trainer import BaseTrainer
from ..utils import check_runner_config, resume_checkpoint_for_training


@MindFormerRegister.register(MindFormerModuleType.TRAINER)
class TextClassificationTrainer(BaseTrainer):
    r"""TextClassification Task For Trainer.
    Args:
        model_name (str): The model name of Task-Trainer. Default: None
    Raises:
        NotImplementedError: If train method or evaluate method or predict method not implemented.
    """

    def __init__(self, model_name: str = None):
        super(TextClassificationTrainer, self).__init__(model_name)
        self.model_name = model_name

    def train(self,
              config: Optional[Union[dict, ConfigArguments]] = None,
              network: Optional[Union[str, BaseModel]] = None,
              dataset: Optional[Union[str, BaseDataset]] = None,
              wrapper: Optional[TrainOneStepCell] = None,
              optimizer: Optional[Optimizer] = None,
              callbacks: Optional[Union[Callback, List[Callback]]] = None,
              **kwargs):
        r"""Train task for TextClassification Trainer.
        This function is used to train or fine-tune the network.

        The trainer interface is used to quickly start training for general task.
        It also allows users to customize the network, optimizer, dataset, wrapper, callback.

        Args:
            config (Optional[Union[dict, ConfigArguments]]): The task config which is used to
                configure the dataset, the hyper-parameter, optimizer, etc.
                It support config dict or ConfigArguments class.
                Default: None.
            network (Optional[Union[str, BaseModel]]): The network for trainer. It support model name supported
                or BaseModel class. Supported model name can refer to ****.
                Default: None.
            dataset (Optional[Union[str, BaseDataset]]): The training dataset. It support real dataset path or
                BaseDateset class or MindSpore Dataset class.
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

        Examples:
            >>> import numpy as np
            >>> from mindspore.dataset import GeneratorDataset
            >>> from mindspore.nn import AdamWeightDecay, TrainOneStepCell
            >>> from mindformers.common.lr import build_lr
            >>> from mindformers.trainer import GeneralTaskTrainer
            >>> from mindformers.tools.register import MindFormerConfig
            >>> from mindformers.models import BertForMultipleChoice, BertConfig
            >>> class MyDataLoader:
            ...    def __init__(self):
            ...        self._data = [np.zeros((24, 128, 768), np.float32) for _ in range(64)]
            ...        self._label = [np.ones((24, 128), np.float32) for _ in range(64)]
            ...    def __getitem__(self, index):
            ...        return self._data[index], self._label[index]
            ...    def __len__(self):
            ...        return len(self._data)
            >>> config = MindFormerConfig("configs/txtcls/run_txtcls_bert_base_uncased.yaml")
            >>> #1) use config to train
            >>> cls_task = TextClassificationTrainer(model_name='bert_for_multiple_choice')
            >>> cls_task.train(config=config)
            >>> #2) use instance function to evaluate
            >>> dataset = GeneratorDataset(source=MyDataLoader(), column_names=['text', 'label_id'])
            >>> dataset = dataset.batch(batch_size=2)
            >>> bert_config = BertConfig(batch_size=2)
            >>> network_with_loss = BertForMultipleChoice(bert_config)
            >>> lr_schedule = build_lr(class_name='WarmUpDecayLR', learning_rate=0.00005, end_learning_rate=0.000001,
            ...                   warmup_steps=100, decay_steps=1000)
            >>> optimizer = AdamWeightDecay(params=network_with_loss.trainable_params(),
            ...                             learning_rate=lr_schedule, weight_decay=0.01)
            >>> wrapper = TrainOneStepCell(network_with_loss, optimizer)
            >>> cls_task.train(config=config, wrapper=wrapper, dataset=dataset)
        """
        self.kwargs = kwargs
        # build dataset
        logger.info(".........Build Dataset..........")
        if dataset is None:
            check_dataset_config(config)
            dataset = build_dataset(config.train_dataset_task)
        check_runner_config(config, dataset)

        # build network
        logger.info(".........Build Net..........")
        if network is None:
            network = build_model(config.model, default_args={
                "parallel_config": config.parallel_config,
                "moe_config": config.moe_config})
        logger.info("Network Parameters: %s M.", str(count_params(network)))

        # build optimizer
        logger.info(".........Build Optimizer..........")
        if optimizer is None:
            steps_per_epoch = dataset.get_dataset_size()
            total_steps = steps_per_epoch * config.runner_config.epochs
            warmup_steps = config.lr_schedule.warmup_steps if config.lr_schedule.warmup_steps > 0 else int(
                0.1 * total_steps)
            lr_schedule = build_lr(class_name=config.lr_schedule.type,
                                   learning_rate=config.lr_schedule.start_lr,
                                   end_learning_rate=config.lr_schedule.end_lr,
                                   warmup_steps=warmup_steps,
                                   decay_steps=total_steps)

            decay_params = [p for n, p in network.parameters_and_names() if
                            not any(nd in n for nd in config.optimizer.no_decay)]
            no_decay_params = [p for n, p in network.parameters_and_names() if
                               any(nd in n for nd in config.optimizer.no_decay)]
            group_params = [{'params': decay_params, 'weight_decay': config.optimizer.weight_decay},
                            {'params': no_decay_params, 'weight_decay': 0.0}]

            optimizer = build_optim(class_name=config.optimizer.type,
                                    params=group_params,
                                    learning_rate=lr_schedule)

        # build callback
        logger.info(".........Build Callbacks for Train..........")
        if callbacks is None:
            callbacks = []
            if config.profile:
                callbacks.append(config.profile_cb)
            callbacks.extend(build_callback(config.callbacks, default_args={"learning_rate": optimizer.learning_rate}))

        # resume checkpoint
        if config.resume_or_finetune_checkpoint:
            logger.info(".............start resume training from checkpoint..................")
            resume_checkpoint_for_training(config, network, optimizer)

        # build runner wrapper
        logger.info(".........Build Running Wrapper..........")
        if wrapper is None:
            net_with_train = build_wrapper(config.runner_wrapper,
                                           default_args={"network": network, "optimizer": optimizer})
        elif isinstance(wrapper, TrainOneStepCell):
            net_with_train = wrapper
        else:
            raise NotImplementedError(f"Now not support this wrapper,"
                                      f"it should be TrainOneStepCell type, but get {wrapper}")

        # define Model and begin training
        logger.info(".........Starting Init Model..........")
        model = Model(net_with_train)

        logger.info(".........Starting Training Model..........")
        model.train(config.runner_config.epochs, dataset,
                    callbacks=callbacks,
                    dataset_sink_mode=config.runner_config.sink_mode,
                    sink_size=config.runner_config.per_epoch_size,
                    initial_epoch=config.runner_config.initial_epoch)
        logger.info(".........Training Over!.............")

    def evaluate(self,
                 config: Optional[Union[dict, ConfigArguments]] = None,
                 network: Optional[Union[str, BaseModel]] = None,
                 dataset: Optional[Union[str, BaseDataset]] = None,
                 callbacks: Optional[Union[Callback, List[Callback]]] = None,
                 compute_metrics: Optional[Union[dict, set]] = None,
                 **kwargs):
        r"""Evaluate task for TextClassification Trainer.
        This function is used to evaluate the network.

        The trainer interface is used to quickly start training for general task.
        It also allows users to customize the network, dataset, callbacks, compute_metrics.

        Args:
            config (Optional[Union[dict, ConfigArguments]]): The task config which is used to
                configure the dataset, the hyper-parameter, optimizer, etc.
                It support config dict or ConfigArguments class.
                Default: None.
            network (Optional[Union[str, BaseModel]]): The network for trainer. It support model name supported
                or BaseModel class. Supported model name can refer to ****.
                Default: None.
            dataset (Optional[Union[str, BaseDataset]]): The training dataset. It support real dataset path or
                BaseDateset class or MindSpore Dataset class.
                Default: None.
            callbacks (Optional[Union[Callback, List[Callback]]]): The training callback function.
                It support CallBack or CallBack List of MindSpore.
                Default: None.
            compute_metrics (Optional[Union[dict, set]]): The metric of evaluating.
                It support dict or set in MindSpore's Metric class.
                Default: None.

        Examples:
            >>> import numpy as np
            >>> from mindspore.dataset import GeneratorDataset
            >>> from mindformers.trainer import GeneralTaskTrainer
            >>> from mindformers.tools.register import MindFormerConfig
            >>> from mindformers.models import BertForMultipleChoice, BertConfig
            >>> from mindformers.common.metric import build_metric
            >>> class MyDataLoader:
            ...    def __init__(self):
            ...        self._data = [np.zeros((24, 128, 768), np.float32) for _ in range(64)]
            ...        self._label = [np.ones((24, 128), np.float32) for _ in range(64)]
            ...    def __getitem__(self, index):
            ...        return self._data[index], self._label[index]
            ...    def __len__(self):
            ...        return len(self._data)
            >>> config = MindFormerConfig("configs/txtcls/run_txtcls_bert_base_uncased.yaml")
            >>> #1) use config to evaluate
            >>> cls_task = TextClassificationTrainer(model_name='bert_for_multiple_choice')
            >>> cls_task.evaluate(config=config)
            >>> #1) use instance function to evaluate
            >>> dataset = GeneratorDataset(source=MyDataLoader(), column_names=['text', 'label_id'])
            >>> dataset = dataset.batch(batch_size=2)
            >>> bert_config = BertConfig(batch_size=2)
            >>> network = BertForMultipleChoice(bert_config)
            >>> compute_metrics = {'Accuracy Metric': build_metric(config.metric)}
            >>> cls_task.evaluate(config=config, network=network, dataset=dataset, compute_metrics=compute_metrics)
        """

        self.kwargs = kwargs
        # build dataset
        logger.info(".........Build Dataset..........")
        check_dataset_config(config)
        if dataset is None:
            dataset = build_dataset(config.eval_dataset_task)
        logger.info("Create eval dataset finish, dataset size:%d", dataset.get_dataset_size())

        # build network
        logger.info(".........Build Net..........")
        if network is None:
            network = build_model(config.model, default_args={
                "parallel_config": config.parallel_config,
                "moe_config": config.moe_config})
        network.set_train(mode=False)
        logger.info("Network Parameters: %s M.", str(count_params(network)))

        logger.info(".........Build Compute Metrics for Evaluate..........")
        if compute_metrics is None:
            compute_metrics = {'Top1 Accuracy': build_metric(config.metric)}

        # build callback
        logger.info(".........Build Callbacks for Evaluate..........")
        if callbacks is None:
            callbacks = []
            if config.profile:
                callbacks.append(config.profile_cb)
            callbacks.extend(build_callback(config.eval_callbacks))

        logger.info(".........Starting Init Model..........")
        model = Model(network, metrics=compute_metrics, eval_network=network)

        logger.info(".........Starting Evaling Model..........")
        output = model.eval(dataset,
                            callbacks=callbacks,
                            dataset_sink_mode=config.runner_config.sink_mode)
        logger.info('Top1 Accuracy=%s', str(output))
        logger.info(".........Evaluate Over!.............")

    def predict(self,
                config: Optional[Union[dict, ConfigArguments]] = None,
                input_data: Optional[Union[str, list]] = None,
                network: Optional[Union[str, BaseModel]] = None,
                tokenizer: Optional[BaseTokenizer] = None,
                **kwargs):
        """
        Executes the predict of the trainer.

        Args:
            config (Optional[Union[dict, ConfigArguments]]): The task config which is used to
                configure the dataset, the hyper-parameter, optimizer, etc.
                It support config dict or ConfigArguments class.
                Default: None.
            input_data (Optional[Union[Tensor, str, list]]): The predict data. Default: None.
            network (Optional[Union[str, BaseModel]]): The network for trainer. It support model name supported
                or BaseModel class. Supported model name can refer to model support list. For .
                Default: None.
            tokenizer (Optional[BaseTokenizer]): The tokenizer for tokenizing the input text.
                Default: None.

        Examples:
            >>> from mindformers import BertModel, TextClassificationTrainer
            >>> model = BertModel.from_pretrained('txtcls_bert_base_uncased_mnli')
            >>> trainer = TextClassificationTrainer(model_name="txtcls_bert_base_uncased_mnli")
            >>> input_data = ["The new rights are nice enough-Everyone really likes the newest benefits ",
            ...               "i don't know um do you do a lot of camping-I know exactly."]
            >>> res = trainer.predict(input_data=input_data, network=model)

        Returns:
            A list of prediction.

        """
        self.kwargs = kwargs
        logger.info(".........Build Input Data For Predict..........")
        if input_data is None:
            input_data = config.input_data
        if not isinstance(input_data, (str, list)):
            raise ValueError("Input data's type must be one of [str, list]")

        if isinstance(input_data, list):
            for item in input_data:
                if not isinstance(item, str):
                    raise ValueError("The element of input data list must be str")

        # bert模型已知issue，由于bert模型在创建的时候需要batch_size参数，
        # 同时pipeline是一个样本一个样本进行处理，所以这里设定为1
        config.model.model_config.batch_size = 1

        top_k = kwargs.pop("top_k", None)
        if top_k is None and config.top_k is not None:
            top_k = config.top_k

        if tokenizer is None:
            tokenizer = BertTokenizer.from_pretrained("txtcls_bert_base_uncased_mnli")

        logger.info(".........Build Net..........")
        if network is None:
            network = build_model(config.model)

        if network is not None:
            logger.info("Network Parameters: %s M.", str(count_params(network)))

        pipeline_task = pipeline(task='text_classification',
                                 tokenizer=tokenizer,
                                 model=network,
                                 max_length=network.config.seq_length,
                                 padding="max_length",
                                 **kwargs)
        output_result = pipeline_task(input_data, top_k=top_k)

        logger.info("output result is: %s", output_result)

        logger.info(".........predict result finished..........")
        logger.info(".........Predict Over!.............")
        return output_result
