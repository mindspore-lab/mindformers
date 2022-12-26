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

import numpy as np
from PIL.Image import Image

from mindspore.train.model import Model
from mindspore.train import Callback
from mindspore.nn import TrainOneStepCell, Optimizer
from mindspore import Tensor

from mindformers.dataset import BaseDataset
from mindformers.models import BaseModel, BaseTokenizer, BaseFeatureExtractor
from mindformers.pipeline import pipeline
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
        if config.resume_checkpoint_path is not None and config.resume_checkpoint_path != '':
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
        logger.info("Network Parameters: %s M.", str(count_params(network)))

        # define metric
        logger.info(".........Build Compute Metrics for Evaluate..........")
        if compute_metrics is None:
            raise NotImplementedError("eval metrics must be define, but get None.")

        # define callback
        logger.info(".........Build Callbacks for Evaluate..........")
        if callbacks is None:
            raise NotImplementedError("eval callbacks must be define, but get None.")

        # define Model and begin training
        logger.info(".........Starting Init Train Model..........")
        model = Model(network, metrics=compute_metrics)

        model.eval(dataset, callbacks=callbacks, dataset_sink_mode=config.runner_config.sink_mode)
        logger.info(".........Evaluate Over!.............")

    def predict(self,
                input_data: Optional[Union[Tensor, np.ndarray, Image, str, list]] = None,
                network: Optional[Union[str, BaseModel]] = None,
                tokenizer: Optional[BaseTokenizer] = None,
                feature_extractor: Optional[BaseFeatureExtractor] = None, **kwargs):
        """predict for trainer."""
        if not isinstance(input_data, (Tensor, np.ndarray, Image, str, list)):
            raise ValueError("Input data's type must be one of "
                             "[str, ms.Tensor, np.ndarray, PIL.Image.Image, list]")

        logger.info(".........Build Net..........")
        if network is None:
            raise NotImplementedError("train network must be define, but get None.")

        if network is not None:
            logger.info("Network Parameters: %s M.", str(count_params(network)))

        if tokenizer is None:
            raise NotImplementedError("tokenizer must be define, but get None.")

        if feature_extractor is None:
            raise NotImplementedError("feature_extractor must be define, but get None.")

        pipeline_task = pipeline(task='general',
                                 model=network,
                                 tokenizer=tokenizer,
                                 feature_extractor=feature_extractor, **kwargs)
        output_result = pipeline_task(input_data)
        logger.info("output result is: %s", str(output_result))
        logger.info(".........Predict Over!.............")
        return output_result
