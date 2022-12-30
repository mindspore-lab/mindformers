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
"""Contrastive Language Image Pretrain Trainer."""
from typing import Optional, List, Union

from mindspore.train.model import Model
from mindspore.train import Callback
from mindspore.nn import Optimizer

from mindformers.common.callback import build_callback
from mindformers.dataset import build_dataset, check_dataset_config, BaseDataset
from mindformers.models import build_model, BaseModel
from mindformers.common.lr import build_lr
from mindformers.common.optim import build_optim
from mindformers.tools.logger import logger
from mindformers.tools.utils import count_params
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from ..config_args import ConfigArguments
from ..base_trainer import BaseTrainer
from ..utils import check_runner_config, resume_checkpoint_for_training


@MindFormerRegister.register(MindFormerModuleType.TRAINER)
class ContrastiveLanguageImagePretrainTrainer(BaseTrainer):
    """ Contrastive Language Image Pretrain Trainer."""
    def __init__(self, model_name: str = None):
        super(ContrastiveLanguageImagePretrainTrainer, self).__init__(model_name)
        self.model_name = model_name
        self.kwargs = None

    def train(self,
              config: Optional[Union[dict, ConfigArguments]] = None,
              network: Optional[Union[str, BaseModel]] = None,
              dataset: Optional[Union[str, BaseDataset]] = None,
              optimizer: Optional[Optimizer] = None,
              callbacks: Optional[Union[Callback, List[Callback]]] = None,
              **kwargs):
        """train for trainer."""
        self.kwargs = kwargs

        # build dataset
        logger.info(".........Build Dataset..........")
        check_dataset_config(config)
        if dataset is None:
            dataset = build_dataset(config.train_dataset_task)
        check_runner_config(config, dataset)

        # build network
        logger.info(".........Build Net..........")
        if network is None:
            network = build_model(config.model)
        logger.info("Network Parameters: %s M.", str(count_params(network)))

        # build optimizer
        logger.info(".........Build Optimizer..........")
        if optimizer is None:
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

        logger.info(".........Build Callbacks for Train..........")
        if callbacks is None:
            callbacks = []
            if config.profile:
                callbacks.append(config.profile_cb)
            callbacks.extend(build_callback(
                config.callbacks))

        # resume checkpoint
        if config.resume_or_finetune_checkpoint is not None and config.resume_or_finetune_checkpoint != '':
            logger.info(".............start resume training from checkpoint..................")
            resume_checkpoint_for_training(config, network, optimizer)

        # define Model and begin training
        logger.info(".........Starting Init Train Model..........")
        model = Model(network, optimizer=optimizer)

        model.train(
            config.runner_config.epochs, dataset, callbacks=callbacks,
            dataset_sink_mode=config.runner_config.sink_mode,
            sink_size=config.runner_config.per_epoch_size,
            initial_epoch=config.runner_config.initial_epoch)
        logger.info(".........Training Over!.............")
