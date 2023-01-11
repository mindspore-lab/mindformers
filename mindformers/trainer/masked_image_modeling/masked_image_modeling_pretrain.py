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
from typing import Optional, List, Union

from mindspore.train.model import Model
from mindspore.train import Callback
from mindspore.nn import TrainOneStepCell, Optimizer

from mindformers.common.callback import build_callback
from mindformers.dataset import build_dataset, check_dataset_config, BaseDataset
from mindformers.models import build_model, BaseModel
from mindformers.common.lr import build_lr
from mindformers.common.optim import build_optim
from mindformers.wrapper import build_wrapper
from mindformers.tools.logger import logger
from mindformers.tools.utils import count_params
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from ..config_args import ConfigArguments
from ..base_trainer import BaseTrainer
from ..utils import check_runner_config, resume_checkpoint_for_training


@MindFormerRegister.register(MindFormerModuleType.TRAINER, alias="mim")
class MaskedImageModelingTrainer(BaseTrainer):
    r"""MaskedImageModeling Task For Trainer.
    Args:
        model_name (str): The model name of Task-Trainer. Default: None
    Raises:
        NotImplementedError: If train method or evaluate method or predict method not implemented.
    """
    def __init__(self, model_name: str = None):
        super(MaskedImageModelingTrainer, self).__init__(model_name)
        self.kwargs = None

    def train(self,
              config: Optional[Union[dict, ConfigArguments]] = None,
              network: Optional[Union[str, BaseModel]] = None,
              dataset: Optional[Union[str, BaseDataset]] = None,
              wrapper: Optional[TrainOneStepCell] = None,
              optimizer: Optional[Optimizer] = None,
              callbacks: Optional[Union[Callback, List[Callback]]] = None,
              **kwargs):
        r"""Train task for MaskedImageModeling Trainer.
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
            >>> from mindspore.nn import AdamWeightDecay, WarmUpLR, \
            ...      DynamicLossScaleUpdateCell, TrainOneStepWithLossScaleCell
            >>> from mindformers.trainer import GeneralTaskTrainer
            >>> from mindformers.tools.register import MindFormerConfig
            >>> from mindformers.models import MaeModel, MaeConfig
            >>> config = MindFormerConfig("configs/mae/run_mae_vit_base_p16_224_800ep.yaml")
            >>> #1) use config to train
            >>> cls_task = MaskedImageModelingTrainer(model_name='mae')
            >>> cls_task.train(config=config)
            >>> #2) use instance function to evaluate
            >>> dataset = build_dataset(config.train_dataset_task)
            >>> mae_config = MaeConfig(batch_size=2)
            >>> network_with_loss = MaeModel(mae_config)
            >>> lr_schedule = WarmUpLR(learning_rate=0.001, warmup_steps=100)
            >>> optimizer = AdamWeightDecay(beta1=0.009, beta2=0.999,
            ...                             learning_rate=lr_schedule,
            ...                             params=network_with_loss.trainable_params())
            >>> loss_scale = DynamicLossScaleUpdateCell(loss_scale_value=2**12, scale_factor=2, scale_window=1000)
            >>> wrapper = TrainOneStepWithLossScaleCell(network_with_loss, optimizer, scale_sense=loss_scale)
            >>> cls_task.train(config=config, wrapper=wrapper, dataset=dataset)
        """
        # 自定义创建模型训练完整过程, 待补充
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
