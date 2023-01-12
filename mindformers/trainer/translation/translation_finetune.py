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
"""Translation Modeling Trainer."""
from typing import Optional, List, Union

from mindspore.train.model import Model
from mindspore.train import Callback
from mindspore.nn import TrainOneStepCell, Optimizer
from mindspore.dataset import GeneratorDataset


from mindformers.common.callback import build_callback
from mindformers.dataset import build_dataset, check_dataset_config, BaseDataset
from mindformers.models import build_model, BaseModel, BaseTokenizer
from mindformers.common.lr import build_lr
from mindformers.common.optim import build_optim
from mindformers.wrapper import build_wrapper
from mindformers.tools.logger import logger
from mindformers.tools.utils import count_params
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.pipeline import pipeline
from ...dataset.dataloader import build_dataset_loader
from ..config_args import ConfigArguments
from ..base_trainer import BaseTrainer
from ..utils import check_runner_config, resume_checkpoint_for_training


@MindFormerRegister.register(MindFormerModuleType.TRAINER, alias="translation")
class TranslationTrainer(BaseTrainer):
    r"""Translation Task For Trainer.
    Args:
        model_name (str): The model name of Task-Trainer. Default: None
    Raises:
        NotImplementedError: If train method or evaluate method or predict method not implemented.
    """
    def __init__(self, model_name: str = None):
        name = model_name
        if isinstance(model_name, BaseModel):
            name = model_name.__class__
        super(TranslationTrainer, self).__init__(name)
        self.kwargs = None

    def train(self,
              config: Optional[Union[dict, ConfigArguments]] = None,
              network: Optional[Union[str, BaseModel]] = None,
              dataset: Optional[Union[str, BaseDataset]] = None,
              wrapper: Optional[TrainOneStepCell] = None,
              optimizer: Optional[Optimizer] = None,
              callbacks: Optional[Union[Callback, List[Callback]]] = None,
              **kwargs):
        r"""Train task for Translation Trainer.
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

        Examples:
            >>> import numpy as np
            >>> from mindspore.dataset import GeneratorDataset
            >>> from mindspore.nn import AdamWeightDecay, WarmUpLR, DynamicLossScaleUpdateCell, \
            ... TrainOneStepWithLossScaleCell
            >>> from mindformers.trainer import TranslationTrainer
            >>> from mindformers.tools.register import MindFormerConfig
            >>> from mindformers.models import T5ForConditionalGeneration, T5Config
            >>> from mindformers import build_dataset
            >>> # follow the instruction in t5 section in the README.md and download wmt16 dataset.
            >>> # change the dataset_files path of configs/t5/wmt16_dataset.yaml
            >>> config = MindFormerConfig("configs/t5/run_t5_tiny_on_wmt16.yaml")
            >>> task = TranslationTrainer(model_name='t5_small')
            >>> task.train(config=config)

        Raises:
            NotImplementedError: If wrapper not implemented.
        """
        self.kwargs = kwargs
        # build dataset
        logger.info(".........Build Dataset..........")
        check_dataset_config(config)
        if dataset is None:
            dataset = build_dataset(config.train_dataset_task)
        check_runner_config(config, dataset)
        step_per_epoch = dataset.get_dataset_size()
        total_steps = config.runner_config.epochs * step_per_epoch
        # build network
        if network is None:
            logger.info(".........Build Net..........")
            network = build_model(config.model, default_args={
                "parallel_config": config.parallel_config,
                "moe_config": config.moe_config})
        logger.info("Network Parameters: %s M.", str(count_params(network)))

        # build optimizer
        logger.info(".........Build Optimizer..........")
        if optimizer is None:
            # build learning rate schedule
            logger.info(".........Build LR Schedule..........")
            if config and config.lr_schedule and config.lr_schedule.decay_steps == -1:
                config.lr_schedule.decay_steps = total_steps
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

    def predict(self,
                config: Optional[Union[dict, ConfigArguments]] = None,
                input_data: Optional[Union[str, list, GeneratorDataset]] = None,
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
            >>> from mindformers import T5ForConditionalGeneration, TranslationTrainer
            >>> model = T5ForConditionalGeneration.from_pretrained('t5_small')
            >>> mim_trainer = TranslationTrainer(model_name="t5_small")
            >>> res = mim_trainer.predict(input_data="hello world", network=model)
            [{'translation_text': ['hello world']}]

        Returns:
            A list of prediction.

        """
        self.kwargs = kwargs

        if input_data is None:
            input_data = build_dataset_loader(config.eval_dataset.data_loader)

        if not isinstance(input_data, (str, list, GeneratorDataset)):
            raise ValueError("Input data's type must be one of "
                             f"[str, list, GeneratorDataset], but got type {type(input_data)}")


        logger.info(".........Build Net..........")
        if network is None:
            network = build_model(config.model)

        if network is not None:
            logger.info("Network Parameters: %s M.", str(count_params(network)))

        save_file = kwargs.pop("save_file", None)
        if save_file is None:
            if config and config.save_file is not None:
                save_file = config.save_file
            else:
                save_file = "results.txt"

        pipeline_task = pipeline(task='translation',
                                 tokenizer=tokenizer,
                                 model=network, **kwargs)
        output_result = pipeline_task(input_data, **kwargs)

        logger.info(".........start to write the output result to: %s.........", save_file)
        with open(save_file, 'w') as file:
            if isinstance(output_result, list):
                for item in output_result:
                    file.write(str(item) + '\n')
            else:
                file.write(str(output_result))
            file.close()

        logger.info(".........writing result finished..........")
        logger.info(".........Predict Over!.............")
        return output_result
