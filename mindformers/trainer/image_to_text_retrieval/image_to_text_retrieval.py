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
"""Image-to-text Retrieval Trainer."""
from typing import List, Optional, Union

import numpy as np
import mindspore as ms
from mindspore import dtype as mstype
from mindspore.train.model import Model
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn import TrainOneStepCell, Optimizer
from mindspore.train import Callback

from mindformers.dataset import build_dataset, check_dataset_config, BaseDataset
from mindformers.models import build_model, BaseModel
from mindformers.core.callback import build_callback
from mindformers.core.lr import build_lr
from mindformers.core.optim import build_optim
from mindformers.wrapper import build_wrapper
from mindformers.trainer.base_trainer import BaseTrainer
from mindformers.trainer.utils import check_runner_config
from mindformers.tools.logger import logger
from mindformers.tools.utils import count_params
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from .eval_utils import get_metrics_ms
from ..config_args import ConfigArguments
from ..base_trainer import BaseTrainer
from ..utils import check_runner_config



@MindFormerRegister.register(MindFormerModuleType.TRAINER, alias="itr")
class ImageToTextRetrievalTrainer(BaseTrainer):
    """Image-to-text Retrieval Trainer."""
    def __init__(self, model_name: str = None):
        super(ImageToTextRetrievalTrainer, self).__init__(model_name)
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
        """
        Training task for ImageToTextRetrievalTrainer Trainer.
        """
        # check mindspore version
        # currently, filip only support training under mindspore2.0
        if not ms.__version__.startswith('2.0'):
            raise NotImplementedError(f"Currently, filip only support training under mindspore2.0, "
                                      f"but with mindspore {ms.__version__}")

        self.kwargs = kwargs

        # no_weight_decay_params filter
        def decay_filter(param_name):
            no_decay_params = config.no_decay_params
            for keyword in no_decay_params:
                if keyword in param_name.name:
                    return False
            return True

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

        network = network.to_float(mstype.float16)

        logger.info("网络参数量：%s M.", str(count_params(network)))

        # build optimizer
        logger.info(".........Build Optimizer..........")
        if optimizer is None:
            # build learning rate schedule
            logger.info(".........Build LR Schedule..........")
            lr_schedule = build_lr(config.lr_schedule)

            params = network.trainable_params()
            decay_params = list(filter(decay_filter, params))
            other_params = list(filter(lambda x: not decay_filter(x), params))
            group_params = [{'params': decay_params, 'weight_decay': config.optimizer.weight_decay},
                            {'params': other_params},
                            {'order_params': params}]
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
        update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2. ** 20,
                                                 scale_factor=2,
                                                 scale_window=1000)
        if wrapper is None:
            model = build_wrapper(config.runner_wrapper,
                                  default_args={"network": network,
                                                "optimizer": optimizer,
                                                "scale_update_cell": update_cell})
        elif isinstance(wrapper, TrainOneStepCell):
            model = wrapper
        else:
            raise NotImplementedError(f"Now not support this wrapper,"
                                      f"it should be TrainOneStepCell type, but get {wrapper}")

        # define Model and begin training
        logger.info(".........Starting Init Train Model..........")
        model = Model(model)

        model.train(
            config.runner_config.epochs, dataset, callbacks=callbacks,
            dataset_sink_mode=config.runner_config.sink_mode,
            sink_size=config.runner_config.sink_size)


    def evaluate(self,
                 config: Optional[Union[dict, ConfigArguments]] = None,
                 network: Optional[Union[str, BaseModel]] = None,
                 dataset: Optional[Union[str, BaseDataset]] = None,
                 callbacks: Optional[Union[Callback, List[Callback]]] = None,
                 **kwargs):
        """
        Evaluation task for ImageToTextRetrievalTrainer Trainer.
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
                "moe_config": config.moe_config,
                "is_training": False})

        network = network.to_float(mstype.float16)

        logger.info("Network Parameters: %s M.", str(count_params(network)))

        # build callback
        logger.info(".........Build Callbacks for Evaluate..........")
        if callbacks is None:
            callbacks = []
            if config.profile:
                callbacks.append(config.profile_cb)
            callbacks.extend(build_callback(config.eval_callbacks))

        logger.info(".........Starting Evaling Model..........")

        image_feature_list = []
        text_feature_list = []
        for data in dataset:
            image, text = data
            image_feature_batch = network.get_image_feature(image)
            image_feature_batch = image_feature_batch.asnumpy()
            image_feature_batch = image_feature_batch / (
                np.linalg.norm(image_feature_batch, axis=-1, keepdims=True) + 1e-6)
            image_feature_list.append(image_feature_batch)
            text_feature_batch = network.get_text_feature(text)
            text_feature_batch = text_feature_batch.asnumpy()
            text_feature_batch = text_feature_batch / (
                np.linalg.norm(text_feature_batch, axis=-1, keepdims=True) + 1e-6)
            text_feature_list.append(text_feature_batch)
        image_feature = np.vstack(image_feature_list)
        text_feature = np.vstack(text_feature_list)
        metrics = get_metrics_ms(image_feature, text_feature)
        img_acc1, img_acc5, img_acc10 = (metrics["image_to_text_R@1"],
                                         metrics["image_to_text_R@5"],
                                         metrics["image_to_text_R@10"])
        txt_acc1, txt_acc5, txt_acc10 = (metrics["text_to_image_R@1"],
                                         metrics["text_to_image_R@5"],
                                         metrics["text_to_image_R@10"])
        logger.info('img: acc1: {:.2f}%, acc5: {:.2f}%, acc10: {:.2f}%'
                    .format(img_acc1.item() * 100, img_acc5.item() * 100, img_acc10.item() * 100))
        logger.info('txt: acc1: {:.2f}%, acc5: {:.2f}%, acc10: {:.2f}%'
                    .format(txt_acc1.item() * 100, txt_acc5.item() * 100, txt_acc10.item() * 100))

        logger.info(".........Evaluate Over!.............")
