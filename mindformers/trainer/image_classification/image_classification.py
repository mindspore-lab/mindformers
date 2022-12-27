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
from typing import Optional, List, Union

import numpy as np
from PIL.Image import Image

from mindspore.train.model import Model
from mindspore.train import Callback
from mindspore.nn import TrainOneStepCell, Optimizer
from mindspore import Tensor

from mindformers.common.metric import build_metric
from mindformers.common.callback import build_callback
from mindformers.dataset import build_dataset, check_dataset_config, BaseDataset
from mindformers.models import build_model, build_processor, \
    BaseModel, BaseImageProcessor
from mindformers.pipeline import pipeline
from mindformers.common.lr import build_lr
from mindformers.common.optim import build_optim
from mindformers.wrapper import build_wrapper
from mindformers.tools.logger import logger
from mindformers.tools.utils import count_params
from mindformers.tools.image_tools import load_image
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from ..config_args import ConfigArguments
from ..base_trainer import BaseTrainer
from ..utils import check_runner_config, check_model_config, \
    check_image_lr_config, resume_checkpoint_for_training


__all__ = ['ImageClassificationTrainer']


@MindFormerRegister.register(MindFormerModuleType.TRAINER, alias="image_classification")
class ImageClassificationTrainer(BaseTrainer):
    """Image Classification Trainer."""

    def __init__(self, model_name: str = None):
        super(ImageClassificationTrainer, self).__init__(model_name)
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
        logger.info("Network Parameters: %s M.", str(count_params(network)))

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
        logger.info(".........Build Callbacks for Train..........")
        if callbacks is None:
            callbacks = []
            if config.profile:
                callbacks.append(config.profile_cb)
            callbacks.extend(build_callback(
                config.callbacks, default_args={"learning_rate": optimizer.learning_rate}))

        # resume checkpoint
        if config.resume_checkpoint_path is not None and config.resume_checkpoint_path != '':
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
        """evaluate for trainer."""

        self.kwargs = kwargs
        # build dataset
        logger.info(".........Build Dataset..........")
        check_dataset_config(config)
        if dataset is None:
            dataset = build_dataset(config.eval_dataset_task)
        logger.info("Create eval dataset finish, dataset size:%d", dataset.get_dataset_size())

        # build network
        logger.info(".........Build Net..........")
        check_model_config(config)
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
                input_data: Optional[Union[Tensor, np.ndarray, Image, str, list]] = None,
                network: Optional[Union[str, BaseModel]] = None,
                image_processor: Optional[BaseImageProcessor] = None, **kwargs):
        """predict for trainer."""
        self.kwargs = kwargs
        logger.info(".........Build Input Data For Predict..........")
        if input_data is None:
            input_data = config.input_data
        if not isinstance(input_data, (Tensor, np.ndarray, Image, str, list)):
            raise ValueError("Input data's type must be one of "
                             "[str, ms.Tensor, np.ndarray, PIL.Image.Image, list]")
        batch_input_data = []
        if isinstance(input_data, str):
            batch_input_data.append(load_image(input_data))
        elif isinstance(input_data, list):
            for data_path in input_data:
                batch_input_data.append(load_image(data_path))
        else:
            batch_input_data = input_data

        logger.info(".........Build Net..........")
        if network is None:
            network = build_model(config.model)

        if network is not None:
            logger.info("Network Parameters: %s M.", str(count_params(network)))

        if image_processor is None:
            image_processor = build_processor(config.processor.image_processor)

        pipeline_task = pipeline(task='image_classification',
                                 model=network,
                                 image_processor=image_processor, **kwargs)
        output_result = pipeline_task(batch_input_data)
        logger.info("output result is: %s", str(output_result))
        logger.info(".........Predict Over!.............")
        return output_result
