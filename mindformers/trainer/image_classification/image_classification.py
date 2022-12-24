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
"""Image Classification Trainer."""
from typing import Callable, List

from mindspore.nn import TrainOneStepCell
from mindspore.train.model import Model

import mindspore as ms

from mindformers.common.metric import build_metric
from mindformers.common.callback import build_callback
from mindformers.dataset import build_dataset, check_dataset_config
from mindformers.models import build_model
from mindformers.common.lr import build_lr
from mindformers.common.optim import build_optim
from mindformers.wrapper import build_wrapper
from mindformers.trainer.utils import check_runner_config, check_model_config, check_image_lr_config
from mindformers.tools.logger import logger
from mindformers.tools.utils import count_params
from mindformers.trainer.base_trainer import BaseTrainer
from mindformers.pipeline import ImageClassificationForPipeline
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

__all__ = ['ImageClassificationTrainer']


@MindFormerRegister.register(MindFormerModuleType.TRAINER, alias="image_classification")
class ImageClassificationTrainer(BaseTrainer):
    """Image Classification Trainer."""

    def __init__(self, model_name: str = None):
        super(ImageClassificationTrainer, self).__init__(model_name)
        self.model_name = model_name

    def train(self,
              config: dict = None,
              network: Callable = None,
              dataset: Callable = None,
              optimizer: Callable = None,
              wrapper: Callable = None,
              callbacks: List[Callable] = None, **kwargs):
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
        check_model_config(config)
        if network is None:
            network = build_model(config.model, default_args={
                "parallel_config": config.parallel_config,
                "moe_config": config.moe_config})
        logger.info("网络参数量：%s M.", str(count_params(network)))

        # build optimizer
        logger.info(".........Build Optimizer..........")
        if optimizer is None:
            # build learning rate schedule
            logger.info(".........Build LR Schedule..........")
            check_image_lr_config(config)
            lr_schedule = build_lr(config.lr_schedule)
            group_params = network.trainable_params()
            if lr_schedule is not None:
                optimizer = build_optim(
                    config.optimizer,
                    default_args={"params": group_params,
                                  "learning_rate": lr_schedule})
            else:
                if config.optimizer.learning_rate is None:
                    raise ValueError("learning_rate must be input")
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

        # load pretrain or resume ckpt
        if config.resume_checkpoint_path is not None and config.resume_checkpoint_path != "":
            params_dict = ms.load_checkpoint(config.resume_checkpoint_path)
            params = ms.load_param_into_net(network, params_dict)
            logger.info(".........net_not_load %s..........", str(params))

        logger.info(".........Starting Training Model..........")
        model.train(config.runner_config.epochs, dataset,
                    callbacks=callbacks,
                    dataset_sink_mode=config.runner_config.sink_mode,
                    sink_size=config.runner_config.per_epoch_size)

    def evaluate(self,
                 config: dict = None,
                 network: Callable = None,
                 dataset: Callable = None,
                 compute_metrics: dict = None, **kwargs):
        """evaluate for trainer."""

        self.kwargs = kwargs
        # build dataset
        logger.info(".........Build Dataset..........")
        check_dataset_config(config)
        if dataset is None:
            dataset = build_dataset(config.eval_dataset_task, do_eval=True)

        # build network
        logger.info(".........Build Net..........")
        check_model_config(config)
        if network is None:
            network = build_model(config.model, default_args={
                "parallel_config": config.parallel_config,
                "moe_config": config.moe_config})
        network.set_train(mode=False)
        logger.info("网络参数量：%s M.", str(count_params(network)))

        # load pretrain or resume ckpt
        if config.resume_checkpoint_path is not None and config.resume_checkpoint_path != "":
            params_dict = ms.load_checkpoint(config.resume_checkpoint_path)
            params = ms.load_param_into_net(network, params_dict)
            logger.info(".........net_not_load %s..........", str(params))

        logger.info(".........Build Metrics..........")
        if compute_metrics is None:
            compute_metrics = {'acc': build_metric(config.metric)}

        logger.info(".........Starting Init Model..........")
        model = Model(network, metrics=compute_metrics, eval_network=network)

        logger.info(".........Starting Evaling Model..........")
        output = model.eval(dataset, dataset_sink_mode=config.runner_config.sink_mode)
        logger.info('accuracy=%s', str(output))

    def predict(self,
                config: dict = None,
                network: Callable = None,
                dataset: Callable = None, **kwargs):
        """predict for trainer."""

    # TODO: use Pipeline to define
    pipeline = ImageClassificationForPipeline()
