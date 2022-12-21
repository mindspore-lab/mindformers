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

from mindspore.train.model import Model

from mindformers.dataset import build_dataset, check_dataset_config
from mindformers.models import build_model
from mindformers.common.lr import build_lr
from mindformers.common.optim import build_optim
from mindformers.common.callback import build_callback
from mindformers.wrapper import build_wrapper
from mindformers.trainer.base_trainer import BaseTrainer
from mindformers.trainer.utils import check_runner_config
from mindformers.tools.logger import logger
from mindformers.tools.utils import count_params
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.TRAINER, alias="translation")
class TranslationTrainer(BaseTrainer):
    """Masked Image Modeling Trainer."""
    def __init__(self, model_name: str = None):
        super(TranslationTrainer, self).__init__(model_name)
        self.model_name = model_name
        self.kwargs = None

    def train(self,
              config: dict = None,
              network: Callable = None,
              dataset: Callable = None,
              optimizer: Callable = None,
              callbacks: List[Callable] = None, **kwargs):
        """train for trainer."""
        # DIY model training, TODO
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
            network = build_model(config.model, default_args={
                "parallel_config": config.parallel_config,
                "moe_config": config.moe_config})
        logger.info("Network params: %s M.", str(count_params(network)))

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

        # build callback
        # build callback
        if callbacks is None:
            callbacks = []
            if config.profile:
                callbacks.append(config.profile_cb)
            callbacks.extend(build_callback(
                config.callbacks, default_args={"learning_rate": optimizer.learning_rate}))

        # build runner wrapper
        logger.info(".........Build Running Wrapper..........")
        model = build_wrapper(config.runner_wrapper, default_args={"network": network, "optimizer": optimizer})

        # define Model and begin training
        logger.info(".........Starting Init Train Model..........")
        model = Model(model)

        model.train(
            config.runner_config.epochs, dataset, callbacks=callbacks,
            dataset_sink_mode=config.runner_config.sink_mode,
            sink_size=config.runner_config.per_epoch_size)

    def evaluate(self,
                 config: dict = None,
                 network: Callable = None,
                 dataset: Callable = None,
                 callbacks: List[Callable] = None, **kwargs):
        """evaluate for trainer."""
        # DIY model eval, TODO
