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
"""General Task Example For Trainer."""
from typing import Optional, List, Union

from mindspore.train.model import Model
from mindspore.train import Callback
from mindspore.nn import TrainOneStepCell, Optimizer

from mindformers.dataset import BaseDataset
from mindformers.models import BaseModel
from mindformers.common.lr import build_lr
from mindformers.common.optim import build_optim
from mindformers.wrapper import build_wrapper
from mindformers.common.callback import build_callback
from mindformers.tools.logger import logger
from mindformers.tools.utils import count_params
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from ..base_trainer import BaseTrainer
from ..config_args import ConfigArguments
from ..utils import check_runner_config, resume_checkpoint_for_training


@MindFormerRegister.register(MindFormerModuleType.TRAINER, alias="general")
class GeneralTaskTrainer(BaseTrainer):
    r"""General Task Example For Trainer.
    Args:
        model_name (str): The model name of Task-Trainer. Default: None
    Raises:
        NotImplementedError: If train method or evaluate method or predict method not implemented.
    """
    def __init__(self, model_name: str = None):
        super(GeneralTaskTrainer, self).__init__(model_name)
        self.kwargs = None

    def train(self,
              config: Optional[Union[dict, ConfigArguments]] = None,
              network: Optional[Union[str, BaseModel]] = None,
              dataset: Optional[Union[str, BaseDataset]] = None,
              wrapper: Optional[TrainOneStepCell] = None,
              optimizer: Optional[Optimizer] = None,
              callbacks: Optional[Union[Callback, List[Callback]]] = None,
              **kwargs):
        r"""Train task for General-Trainer.
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
            NotImplementedError: If network or dataset not implemented.

        Examples:
            >>> import numpy as np
            >>> from mindspore.dataset import GeneratorDataset
            >>> from mindformers.trainer import GeneralTaskTrainer
            >>> from mindformers.tools.register import MindFormerConfig
            >>> from mindformers.models import VitModel, VitConfig
            >>> class MyDataLoader:
            ...    def __init__(self):
            ...        self._data = [np.zeros((3, 224, 224), np.float32) for _ in range(64)]
            ...        self._label = [np.ones(1000, np.float32) for _ in range(64)]
            ...
            ...    def __getitem__(self, index):
            ...        return self._data[index], self._label[index]
            ...
            ...    def __len__(self):
            ...        return len(self._data)
            >>> config = MindFormerConfig("configs/general/run_general_task.yaml")
            >>> dataset = GeneratorDataset(source=MyDataLoader(), column_names=['image', 'label'])
            >>> dataset = dataset.batch(batch_size=2)
            >>> vit_config = VitConfig(batch_size=2)
            >>> network_with_loss = VitModel(vit_config)
            >>> general_task = GeneralTaskTrainer(model_name='vit')
            >>> general_task.train(config=config, network=network_with_loss, dataset=dataset)
        """
        self.kwargs = kwargs
        # build dataset
        logger.info(".........Build Dataset..........")
        if dataset is None:
            raise NotImplementedError("train dataset must be define, but get None.")
        check_runner_config(config, dataset)

        # build network
        logger.info(".........Build Net..........")
        if network is None and wrapper is None:
            raise NotImplementedError("train network must be define, but get None.")

        if network is not None:
            logger.info("Network Parameters: %s M.", str(count_params(network)))

        # build optimizer
        logger.info(".........Build Optimizer..........")
        if optimizer is None and wrapper is None:
            # build learning rate schedule
            logger.info(".........Build LR Schedule..........")
            lr_schedule = build_lr(config.lr_schedule)
            group_params = network.trainable_params()
            if lr_schedule is not None:
                optimizer = build_optim(
                    config.optimizer,
                    default_args={"params": group_params,
                                  "learning_rate": lr_schedule})
            else:
                assert config.optimizer.learning_rate, "learning_rate must be input"
                optimizer = build_optim(
                    config.optimizer,
                    default_args={"params": group_params})

        # build callback
        logger.info(".........Build Callbacks for Train..........")
        if callbacks is None:
            callbacks = []
            if config.profile:
                callbacks.append(config.profile_cb)
            callbacks.extend(build_callback(
                config.callbacks, default_args={"learning_rate": optimizer.learning_rate}))

        # resume checkpoint
        if config.resume_or_finetune_checkpoint is not None and config.resume_or_finetune_checkpoint != '':
            logger.info(".............start resume training from checkpoint..................")
            resume_checkpoint_for_training(config, network, optimizer)

        # build runner wrapper
        logger.info(".........Build Running Wrapper..........")
        if wrapper is None:
            model = build_wrapper(config.runner_wrapper, default_args={"network": network, "optimizer": optimizer})
        elif isinstance(wrapper, TrainOneStepCell):
            model = wrapper
            logger.info("Network params: %s M.", str(count_params(model.network)))
        else:
            raise NotImplementedError(f"Now not support this wrapper,"
                                      f"it should be TrainOneStepCell type, but get {wrapper}")

        # define Model and begin training
        logger.info(".........Starting Init Train Model..........")
        model = Model(model)

        model.train(
            config.runner_config.epochs, dataset, callbacks=callbacks,
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
        r"""Evaluate task for General-Trainer.
        This function is used to evaluate the network.

        The trainer interface is used to quickly start training for general task.
        It also allows users to customize the network, dataset, callback, compute_metrics.

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

        Raises:
            NotImplementedError: If network or dataset or compute_metrics not implemented.

        Examples:
            >>> import numpy as np
            >>> from mindspore.nn import Accuracy
            >>> from mindspore.dataset import GeneratorDataset
            >>> from mindformers.trainer import GeneralTaskTrainer
            >>> from mindformers.tools.register import MindFormerConfig
            >>> from mindformers.models import VitModel, VitConfig
            >>> class MyDataLoader:
            ...    def __init__(self):
            ...        self._data = [np.zeros((3, 224, 224), np.float32) for _ in range(64)]
            ...        self._label = np.random.randint(1000, size=64)
            ...
            ...    def __getitem__(self, index):
            ...        return self._data[index], self._label[index]
            ...
            ...    def __len__(self):
            ...        return len(self._data)
            >>> config = MindFormerConfig("configs/general/run_general_task.yaml")
            >>> dataset = GeneratorDataset(source=MyDataLoader(), column_names=['image', 'label'])
            >>> dataset = dataset.batch(batch_size=2)
            >>> vit_config = VitConfig(batch_size=2)
            >>> network = VitModel(vit_config)
            >>> compute_metrics = {"Accuracy": Accuracy(eval_type='classification')}
            >>> general_task = GeneralTaskTrainer(model_name='vit')
            >>> general_task.evaluate(config=config, network=network, dataset=dataset, compute_metrics=compute_metrics)
        """
        self.kwargs = kwargs
        # build dataset
        logger.info(".........Build Dataset..........")
        if dataset is None:
            raise NotImplementedError("eval dataset must be define, but get None.")
        check_runner_config(config, dataset)

        # build network
        logger.info(".........Build Net..........")
        if network is None:
            raise NotImplementedError("train network must be define, but get None.")
        network.set_train(mode=False)
        logger.info("Network Parameters: %s M.", str(count_params(network)))

        # define metric
        logger.info(".........Build Compute Metrics for Evaluate..........")
        if compute_metrics is None:
            raise NotImplementedError("eval metrics must be define, but get None.")

        # define callback
        logger.info(".........Build Callbacks for Evaluate..........")
        if callbacks is None:
            callbacks = build_callback(config.eval_callbacks)

        # define Model and begin training
        logger.info(".........Starting Init Train Model..........")
        model = Model(network, metrics=compute_metrics, eval_network=network)

        output = model.eval(dataset, callbacks=callbacks, dataset_sink_mode=config.runner_config.sink_mode)
        logger.info('Top1 Accuracy=%s', str(output))
        logger.info(".........Evaluate Over!.............")
