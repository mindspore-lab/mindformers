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
from mindformers.trainer.base_trainer import BaseTrainer
from mindformers.trainer.utils import check_runner_config
from mindformers.tools.logger import logger
from mindformers.tools.utils import count_params
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from ..config_args import ConfigArguments


@MindFormerRegister.register(MindFormerModuleType.TRAINER, alias="general")
class GeneralTaskTrainer(BaseTrainer):
    """General Task Example For Trainer."""
    def __init__(self, model_name: str = None):
        super(GeneralTaskTrainer, self).__init__(model_name)
        self.model_name = model_name
        self.kwargs = None

    def train(self,
              config: Optional[Union[dict, ConfigArguments]] = None,
              network: Optional[Union[str, BaseModel]] = None,
              dataset: Optional[Union[str, BaseDataset]] = None,
              wrapper: Optional[TrainOneStepCell] = None,
              optimizer: Optional[Optimizer] = None,
              callbacks: Optional[Union[Callback, List[Callback]]] = None,
              **kwargs):
        """train for trainer."""
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
            logger.info("Network params: %s M.", str(count_params(network)))

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
        if callbacks is None:
            callbacks = []
            if config.profile:
                callbacks.append(config.profile_cb)
            callbacks.extend(build_callback(
                config.callbacks, default_args={"learning_rate": optimizer.learning_rate}))

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
            sink_size=config.runner_config.per_epoch_size)

    def evaluate(self,
                 config: Optional[Union[dict, ConfigArguments]] = None,
                 network: Optional[Union[str, BaseModel]] = None,
                 dataset: Optional[Union[str, BaseDataset]] = None,
                 callbacks: Optional[Union[Callback, List[Callback]]] = None,
                 compute_metrics: Optional[Union[dict, set]] = None,
                 **kwargs):
        """eval for trainer."""
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
        logger.info("Network params: %s M.", str(count_params(network)))

        # define metric
        if compute_metrics is None:
            raise NotImplementedError("eval metrics must be define, but get None.")

        # define callback
        if callbacks is None:
            raise NotImplementedError("eval callbacks must be define, but get None.")

        # define Model and begin training
        logger.info(".........Starting Init Train Model..........")
        model = Model(network, metrics=compute_metrics)

        model.eval(dataset, callbacks=callbacks, dataset_sink_mode=config.runner_config.sink_mode)
