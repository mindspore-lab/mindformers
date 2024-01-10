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
"""Trainer API For Import."""
import os
import shutil
from collections import OrderedDict
from typing import List, Optional, Union

import numpy as np
from PIL.Image import Image

import mindspore as ms
from mindspore import Tensor
from mindspore.common import set_seed
from mindspore._checkparam import args_type_check
from mindspore import load_checkpoint, load_param_into_net
from mindspore.nn import TrainOneStepCell, Optimizer, Cell
from mindspore.train import Callback
from mindspore.dataset import GeneratorDataset
from mindspore.dataset.engine.datasets import BatchDataset, RepeatDataset, Dataset

from mindformers.core.parallel_config import build_parallel_config
from mindformers.dataset import build_dataset, build_dataset_loader, \
    check_dataset_config, BaseDataset
from mindformers.mindformer_book import MindFormerBook
from mindformers.models import BaseModel, BaseImageProcessor, \
    BaseTokenizer, BaseAudioProcessor
from mindformers.tools.utils import set_output_path, set_strategy_save_path
from mindformers.tools.logger import logger
from mindformers.tools.utils import get_real_rank, get_real_group_size
from mindformers.tools.register import MindFormerConfig
from mindformers.tools.register.config import ordered_yaml_dump
from .build_trainer import build_trainer
from .config_args import ConfigArguments
from .training_args import TrainingArguments
from .utils import check_train_data_loader_type, check_eval_data_loader_type, \
    check_optimizer_and_lr_type, check_wrapper_config, config2dict

__all__ = ['Trainer']

SUPPORT_TASKS = MindFormerBook().get_trainer_support_task_list()
SUPPORT_MODEL_NAMES = MindFormerBook().get_model_name_support_list()
SUPPORT_PIPELINES = MindFormerBook().get_pipeline_support_task_list()
SUPPORT_PIPELINE_INPUT_DATA = MindFormerBook().get_pipeline_support_input_data_list()
CURRENT_PROJECT_PATH = MindFormerBook().get_project_path()
DEFAULT_CHECKPOINT_DIR = 'checkpoint'
DEFAULT_CONFIG_DIR = 'configs'


class Trainer:
    r"""
    Executor of general task trainers. It can initialize a trainer instance of the specific task through the task name
    and configuration file. It provides users with the ability to implement different tasks by encapsulating
    the training, fine-tuning evaluation and prediction of trainer instance. It also allows users to customize
    the model, optimizer, dataset, tokenizer, processor, train_one_step, callback, and metric.

    Args:
        args (Optional[Union[str, TrainingArguments]]):
            The task config which is used to configure the dataset, the hyper-parameter, optimizer, etc.
            It supports yaml path or MindFormerConfig or TrainingArguments class.
            Default: None.
        task (str):
            Supported task type can refer to
            https://mindformers.readthedocs.io/zh-cn/latest/docs/model_support_list.html#.

            Default: 'general'.
        model (Optional[Union[str, BaseModel]]):
            The network for trainer. It supports model name supported or BaseModel.
            Supported model type can refer to
            https://mindformers.readthedocs.io/zh-cn/latest/docs/model_support_list.html#.
            Default: None.
        model_name (Optional[Union[str]]):
            Supported model name can refer to
            https://mindformers.readthedocs.io/zh-cn/latest/docs/model_support_list.html#.

            When the incoming model or wrapper is a custom instance,
            it is recommended to specify the supported model_name to get the base configuration of the model type.
            Default: None.
        pet_method (Optional[Union[str]]):
            Supported pet method name can refer to
            https://mindformers.readthedocs.io/zh-cn/latest/docs/model_support_list.html#llm.

            Default: ''.
        train_dataset (Optional[Union[str, BaseDataset]]):
            The training dataset. It support real dataset path or BaseDateset class or MindSpore Dataset class.
            Default: None.
        eval_dataset (Optional[Union[str, BaseDataset]]):
            The evaluate dataset. It support real dataset path or BaseDateset class or MindSpore Dataset class.
            Default: None.
        tokenizer (Optional[BaseTokenizer]):
            The tokenizer for text preprocessing. It support BaseTokenizer class.
            Default: None.
        image_processor (Optional[BaseImageProcessor]):
            The processor for image preprocessing. It supports BaseImageProcessor class.
            Default: None.
        audio_processor (Optional[BaseAudioProcessor]):
            The processor for audio preprocessing. It supports BaseAudioProcessor class.
            Default: None.
        optimizers (Optional[Optimizer]):
            The training network's optimizer. It support Optimizer class of MindSpore.
            Default: None.
        wrapper (Optional[TrainOneStepCell]):
            Wraps the `network` with the `optimizer`. It supports TrainOneStepCell class of MindSpore.
            Default: None.
        callbacks (Optional[Union[Callback, List[Callback]]]):
            The training callback function. It supports CallBack or CallBack List of MindSpore.
            Default: None.
        eval_callbacks (Optional[Union[Callback, List[Callback]]]):
            The evaluate callback function. It supports CallBack or CallBack List of MindSpore.
            Default: None.
        compute_metrics (Optional[Union[dict, set]]):
            The metric of evaluating. It supports dict or set in MindSpore's Metric class.
            Default: None.
        save_config (bool):
            Save current the config of task.
            Default: False.

    Raises:
        KeyError: If 'task' or 'model' not in supported trainer.
    """
    @args_type_check(
        args=(str, MindFormerConfig, TrainingArguments), task=str, model=(str, BaseModel),
        model_name=str, train_dataset=(str, BaseDataset, Dataset), eval_dataset=(str, BaseDataset, Dataset),
        tokenizer=BaseTokenizer, image_processor=BaseImageProcessor, audio_processor=BaseAudioProcessor,
        optimizers=Optimizer, wrapper=TrainOneStepCell, pet_method=str, callbacks=(Callback, list),
        eval_callbacks=(Callback, list), compute_metrics=(dict, set), save_config=bool)
    def __init__(self,
                 args: Optional[Union[str, MindFormerConfig, TrainingArguments]] = None,
                 task: Optional[str] = 'general',
                 model: Optional[Union[str, BaseModel]] = None,
                 model_name: Optional[Union[str]] = None,
                 train_dataset: Optional[Union[str, BaseDataset]] = None,
                 eval_dataset: Optional[Union[str, BaseDataset]] = None,
                 tokenizer: Optional[BaseTokenizer] = None,
                 image_processor: Optional[BaseImageProcessor] = None,
                 audio_processor: Optional[BaseAudioProcessor] = None,
                 optimizers: Optional[Optimizer] = None,
                 wrapper: Optional[TrainOneStepCell] = None,
                 pet_method: Optional[str] = '',
                 callbacks: Optional[Union[Callback, List[Callback]]] = None,
                 eval_callbacks: Optional[Union[Callback, List[Callback]]] = None,
                 compute_metrics: Optional[Union[dict, set]] = None,
                 save_config: bool = False):
        self.task = task
        self.model = model
        self.model_name = model_name
        self.pet_method = pet_method
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.optimizers = optimizers
        self.wrapper = wrapper
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.audio_processor = audio_processor
        self.callbacks = callbacks
        self.eval_callbacks = eval_callbacks
        self.compute_metrics = compute_metrics
        self.default_checkpoint_name_or_path = None
        self.configs_directory = os.path.join('.', DEFAULT_CONFIG_DIR)

        if not os.path.exists(os.path.join('.', DEFAULT_CONFIG_DIR)):
            configs_directory = os.path.join('.', DEFAULT_CONFIG_DIR)
            if os.path.exists(os.path.join(CURRENT_PROJECT_PATH, DEFAULT_CONFIG_DIR)):
                mindformers_configs_directory = os.path.join(CURRENT_PROJECT_PATH, DEFAULT_CONFIG_DIR)
                # python 3.7 版本不支持dirs_exist_ok入参, python 3.8及以上版本支持
                try:
                    # adapt to python 3.8+
                    # pylint: disable=E1123
                    shutil.copytree(mindformers_configs_directory, configs_directory, dirs_exist_ok=True)
                except TypeError:
                    try:
                        # adapt to python 3.7
                        shutil.copytree(mindformers_configs_directory, configs_directory)
                    except FileExistsError:
                        pass
        if wrapper is not None:
            if model is not None:
                logger.warning(
                    'wrapper has existed, input model invalid, it should be include in wrapper.')
            if optimizers is not None:
                logger.warning(
                    'wrapper has existed, input optimizers invalid, it should be include in wrapper.')
            if self.model_name is None:
                logger.warning("wrapper has existed, you are advised to pass the args of model_name."
                               "Supported model name can refer to"
                               "https://mindformers.readthedocs.io/zh-cn/latest/docs/model_support_list.html#.")

        if task not in SUPPORT_TASKS.keys():
            raise ValueError(
                "The value of task must be in {}, but get {}".format(SUPPORT_TASKS.keys(), task))

        if isinstance(self.model, (Cell, BaseModel)):
            logger.info("The model instance has been entered, "
                        "and the model will not be created from model_config")
            if pet_method:
                logger.warning("pet_method is not valid when a model instance is passed in."
                               "Currently, only part of the model in MindFormers is supported."
                               "Please pass in the model keyword")
            self.is_model_instance = True
        else:
            self.is_model_instance = False
            self.model_name = 'common'

        if isinstance(self.model, str):
            model = model + '_{}'.format(pet_method) if pet_method else model
            assert model in SUPPORT_MODEL_NAMES, \
                f"model must be in {SUPPORT_MODEL_NAMES} when model's type is string, but get {model}."
            self.model_name = model
            self.model = None

        if self.is_model_instance and self.model_name is None:
            logger.warning(
                "Recognizing that a model instance is sent and model_name is None,")
            logger.warning(
                "it is recommended to select a model configuration that corresponds "
                "to the support of MindFormers based on the instance model and set model_name.")
            logger.warning(
                "Otherwise, they will default to a general configuration."
                "You are advised to pass instances such as optimizers, metric, tokenizer, and processor")
            self.model_name = 'common'

        default_config_path = SUPPORT_TASKS.get(self.task).get(self.model_name)
        relative_config_path = default_config_path[default_config_path.rfind("configs/"):]
        current_config_path = os.path.join(os.getcwd(), relative_config_path)
        if os.path.exists(current_config_path):
            default_config_path = current_config_path
        config_path = default_config_path
        task_config = MindFormerConfig(config_path)

        if self.model_name == "common":
            if self.model is not None:
                task_config.trainer.model_name = self.model.__class__.__name__
            if self.wrapper is not None:
                task_config.trainer.model_name = self.wrapper.network.__class__.__name__

        if args is None:
            self.config = task_config
        else:
            if isinstance(args, MindFormerConfig):
                config_path = None
                task_config = args
            elif isinstance(args, str):
                assert os.path.realpath(args) and os.path.exists(args), \
                    f"config path must be exist, but get {args}."
                assert args.endswith(('.yaml', '.yml')), \
                    f"config file must be end with .yaml or .yml, but get {args}"
                config_path = args
                task_config = MindFormerConfig(config_path)
            elif isinstance(args, ConfigArguments):
                if hasattr(args, 'train_dataset'):
                    check_train_data_loader_type(args, task_config)
                if hasattr(args, 'eval_dataset'):
                    check_eval_data_loader_type(args, task_config)
                if hasattr(args, 'optimizer'):
                    check_optimizer_and_lr_type(args, task_config)
                if hasattr(args, 'runner_wrapper'):
                    check_wrapper_config(args, task_config)
                task_config.merge_from_dict(args.__dict__)
            elif isinstance(args, TrainingArguments):
                logger.warning(
                    "When using the TrainingArguments class, "
                    "its arguments will override the default config configuration.")
                args.convert_args_to_mindformers_config(task_config)

            self.config = task_config

        if config_path:
            logger.info(f"Load configs in {config_path} to build trainer.")

        self._config_type_check(self.config)

        # check dataset config
        if isinstance(train_dataset, str):
            assert os.path.exists(train_dataset), \
                f"train dataset path must be exist, but get {train_dataset}."
            self.config.train_dataset.data_loader.dataset_dir = train_dataset
            self.train_dataset = None
        if isinstance(eval_dataset, str):
            assert os.path.exists(eval_dataset), \
                f"eval dataset path must be exist, but get {eval_dataset}."
            self.config.eval_dataset.data_loader.dataset_dir = eval_dataset
            self.eval_dataset = None

        if tokenizer is not None:
            if self.config.train_dataset is not None:
                self.config.train_dataset.tokenizer = tokenizer
            if self.config.eval_dataset is not None:
                self.config.eval_dataset.tokenizer = tokenizer
        check_dataset_config(self.config)

        # build parallel config
        self.rank_id = get_real_rank()
        self.device_num = get_real_group_size()
        self.config.rank_id = self.rank_id
        self.config.device_num = self.device_num

        self.is_set_parallel_config = False
        self.is_set_moe_config = False
        self.is_set_recompute_config = False

        # set seed
        if self.config.seed and \
                ms.context.get_auto_parallel_context("parallel_mode") \
                not in ["semi_auto_parallel", "auto_parallel"]:
            set_seed(self.config.seed)
            np.random.seed(self.config.seed)

        # set output directory
        set_output_path(self.config.output_dir)
        set_strategy_save_path(self.config.parallel)

        # build task trainer
        self.trainer = build_trainer(self.config.trainer)
        if self.trainer is None:
            raise ModuleNotFoundError("config must be contain 'trainer' key, but get None.")

        if save_config:
            self._save_config_to_yaml(self.config)
            logger.info("save running config success of %s_new.", task_config.trainer.model_name.lower())

    @args_type_check(train_checkpoint=(str, bool), resume_training=bool,
                     auto_trans_ckpt=bool, do_eval=bool)
    def train(self,
              train_checkpoint: Optional[Union[str, bool]] = False,
              resume_training: Optional[bool] = None,
              auto_trans_ckpt: Optional[bool] = None,
              do_eval: bool = False):
        """
        The training API of Trainer. After setting custom settings, implement training by calling the
        training method of task-trainer instance.

        Args:
            train_checkpoint (Optional[Union[str, bool]]):
                Used to restore training or fine-tune the weight of the network.
                It supports real checkpoint path or valid model name of mindformers or bool value.
                if it's true, the last checkpoint file saved from the previous training round is automatically used.
            resume_training (bool):
                Whether to perform resume training. Default: False.
            auto_trans_ckpt:
                auto transform checkpoint to load in distributed model
            do_eval (bool):
                Whether evaluations are performed during training. Default: False.

        Returns:
            None

        Raises:
            TypeError: if train_checkpoint is not bool or str type.
        """
        if train_checkpoint is not None and \
                not isinstance(train_checkpoint, (bool, str)):
            raise TypeError(f"train_checkpoint must be one of [None, string, bool], "
                            f"but get {train_checkpoint}")
        if train_checkpoint is False:
            train_checkpoint = None

        do_eval = do_eval or self.config.do_eval
        if do_eval:
            if self.eval_dataset is None:
                logger.info("do_eval is enabled, building eval_dataset from config.")
                self.eval_dataset = build_dataset(self.config.eval_dataset_task)
            if self.eval_dataset is None:
                raise ValueError(f"if do_eval is true, eval_dataset must be input, "
                                 f"the task {self.task} is not support eval now.")
            # open do_eval for trainer config
            self.config.do_eval = True

        if train_checkpoint is True:
            self.config.model.model_config.checkpoint_name_or_path = None
            self.config.load_checkpoint = self.get_last_checkpoint()
        elif isinstance(train_checkpoint, str):
            self.config.model.model_config.checkpoint_name_or_path = None
            self.config.load_checkpoint = train_checkpoint
        else:
            self.default_checkpoint_name_or_path = self.config.model.model_config.checkpoint_name_or_path
            self.config.model.model_config.checkpoint_name_or_path = None

        if resume_training is not None:
            self.config.resume_training = resume_training
        if auto_trans_ckpt is not None:
            self.config.auto_trans_ckpt = auto_trans_ckpt

        if self.is_model_instance:
            self._reset_model_instance(is_train=True)

        self.trainer.train(
            config=self.config, network=self.model,
            dataset=self.train_dataset, optimizer=self.optimizers,
            eval_dataset=self.eval_dataset if do_eval else None,
            wrapper=self.wrapper,
            callbacks=self.callbacks,
            is_full_config=True)

    @args_type_check(finetune_checkpoint=(str, bool), resume_training=bool,
                     auto_trans_ckpt=bool, do_eval=bool)
    def finetune(self,
                 finetune_checkpoint: Optional[Union[str, bool]] = False,
                 resume_training: Optional[bool] = None,
                 auto_trans_ckpt: Optional[bool] = None,
                 do_eval: bool = False):
        """
        The fine-tuning API of Trainer. After setting custom settings, implement fine-tuning by calling the
        training method of task-trainer instance.

        Args:
            finetune_checkpoint (Optional[Union[str, bool]]):
                Used to restore training or fine-tune the weight of the network.
                It supports real checkpoint path or valid model name of mindformers or bool value.
                if it's true, the last checkpoint file saved from the previous training round is automatically used.
                if resume_training is true, this checkpoint will be used to restore training of the network.
                Default: False.
            resume_training (bool):
                Whether to perform resume training. Default: False
            auto_trans_ckpt:
                auto transform checkpoint to load in distributed model
            do_eval (bool):
                Whether evaluations are performed during training. Default: False.

        Returns:
            None

        Raises:
            TypeError: if load_checkpoint is not bool or str type.
        """
        if finetune_checkpoint is not None and \
                not isinstance(finetune_checkpoint, (bool, str)):
            raise TypeError(f"finetune_checkpoint must be one of [None, string, bool], "
                            f"but get {finetune_checkpoint}")
        if finetune_checkpoint is False:
            finetune_checkpoint = None

        do_eval = do_eval or self.config.do_eval
        if do_eval:
            if self.eval_dataset is None:
                logger.info("do_eval is enabled, building eval_dataset from config.")
                self.eval_dataset = build_dataset(self.config.eval_dataset_task)
            if self.eval_dataset is None:
                raise ValueError(f"if do_eval is true, eval_dataset must be input, "
                                 f"the task {self.task} is not support eval now.")
            # open do_eval for trainer config
            self.config.do_eval = True

        if finetune_checkpoint is True:
            self.config.model.model_config.checkpoint_name_or_path = None
            self.config.load_checkpoint = self.get_last_checkpoint()
        elif isinstance(finetune_checkpoint, str):
            self.config.model.model_config.checkpoint_name_or_path = None
            self.config.load_checkpoint = finetune_checkpoint
        else:
            self.default_checkpoint_name_or_path = self.config.model.model_config.checkpoint_name_or_path
            if auto_trans_ckpt:
                if self.is_model_instance:
                    logger.warning(
                        "When a model instance is identified,"
                        "the weights that are currently proposed to be fine-tune are specified"
                        "by the finetune_checkpoint argument or a model instance "
                        "with the model weights already loaded is imported")
                else:
                    self.config.load_checkpoint = self.config.model.model_config.checkpoint_name_or_path
                self.config.model.model_config.checkpoint_name_or_path = None
            else:
                self.config.load_checkpoint = None
            self.config.model.model_config.checkpoint_name_or_path = None

        if resume_training is not None:
            self.config.resume_training = resume_training
        if auto_trans_ckpt is not None:
            self.config.auto_trans_ckpt = auto_trans_ckpt

        if self.is_model_instance:
            self._reset_model_instance(is_train=True)

        self.trainer.train(
            config=self.config, network=self.model,
            dataset=self.train_dataset, optimizer=self.optimizers,
            eval_dataset=self.eval_dataset if do_eval else None,
            wrapper=self.wrapper,
            callbacks=self.callbacks,
            is_full_config=True)

    @args_type_check(eval_checkpoint=(str, bool), auto_trans_ckpt=bool)
    def evaluate(self,
                 eval_checkpoint: Optional[Union[str, bool]] = False,
                 auto_trans_ckpt: Optional[bool] = None,
                 **kwargs):
        """
        The evaluation API of Trainer. After setting custom settings, implement evaluation by calling the
        evaluation method of task-trainer instance.

        Args:
            eval_checkpoint (Optional[Union[str, bool]]):
                Used to evaluate the weight of the network.
                It supports real checkpoint path or valid model name of mindformers or bool value.
                if it's true, the last checkpoint file saved from the previous training round is automatically used.
                Default: False.
            auto_trans_ckpt:
                auto transform checkpoint to load in distributed model

        Returns:
            None

        Raises:
            TypeError: if eval_checkpoint is not bool or str type.
        """
        if eval_checkpoint is not None and not isinstance(eval_checkpoint, (bool, str)):
            raise TypeError(f"eval_checkpoint must be one of [None, string, bool], "
                            f"but get {eval_checkpoint}")

        if eval_checkpoint is False:
            eval_checkpoint = None

        if eval_checkpoint is True:
            self.config.model.model_config.checkpoint_name_or_path = None
            self.config.load_checkpoint = self.get_last_checkpoint()
        elif isinstance(eval_checkpoint, str):
            self.config.model.model_config.checkpoint_name_or_path = None
            self.config.load_checkpoint = eval_checkpoint
        else:
            self.default_checkpoint_name_or_path = self.config.model.model_config.checkpoint_name_or_path
            if auto_trans_ckpt:
                if self.is_model_instance:
                    logger.warning(
                        "When a model instance is identified,"
                        "the weights that are currently proposed to be evaluated are specified"
                        "by the eval_checkpoint argument or a model instance "
                        "with the model weights already loaded is imported")
                else:
                    self.config.load_checkpoint = self.config.model.model_config.checkpoint_name_or_path
                self.config.model.model_config.checkpoint_name_or_path = None
            else:
                self.config.load_checkpoint = None

        if auto_trans_ckpt is not None:
            self.config.auto_trans_ckpt = auto_trans_ckpt

        if self.is_model_instance:
            self._reset_model_instance(is_train=False)

        self.trainer.evaluate(
            config=self.config, network=self.model,
            dataset=self.eval_dataset, callbacks=self.eval_callbacks,
            is_full_config=True, **kwargs)

    @args_type_check(predict_checkpoint=(str, bool), auto_trans_ckpt=bool,
                     input_data=(GeneratorDataset, Tensor, np.ndarray, Image, str, list),
                     batch_size=int)
    def predict(self,
                predict_checkpoint: Optional[Union[str, bool]] = None,
                auto_trans_ckpt: Optional[bool] = None,
                input_data: Optional[Union[GeneratorDataset,
                                           Tensor, np.ndarray, Image, str, list]] = None,
                batch_size: int = None,
                **kwargs):
        """
        The prediction API of Trainer. After setting custom settings, implement prediction by calling the
        prediction method of task-trainer instance.

        Args:
            predict_checkpoint (Optional[Union[str, bool]]):
                Used to predict the weight of the network.
                It supports real checkpoint path or valid model name of mindformers or bool value.
                if it's true, the last checkpoint file saved from the previous training round is automatically used.
                Default: False.
            auto_trans_ckpt:
                auto transform checkpoint to load in distributed model
            input_data (Optional[Union[Tensor, np.ndarray, Image, str, list]]):
                The predict data. Default: None.
            batch_size (Optional[int]):
                Batch size of predict data. Default: None.

        Return:
            predict result (dict).

        Raises:
            TypeError: if predict_checkpoint is not bool or str type.
            TypeError: if input_data is not Tensor or np.ndarray or Image or str or list.
        """
        if predict_checkpoint is not None and not isinstance(predict_checkpoint, (bool, str)):
            raise TypeError(f"predict_checkpoint must be one of [None, string, bool], "
                            f"but get {predict_checkpoint}")

        if self.task not in SUPPORT_PIPELINES.keys():
            raise NotImplementedError(f"The {self.task} not support predict, "
                                      f"now this tasks {SUPPORT_PIPELINES.keys()} is support predict.")

        if batch_size is not None:
            kwargs["batch_size"] = batch_size

        if predict_checkpoint is False:
            predict_checkpoint = None

        if predict_checkpoint is True:
            self.config.model.model_config.checkpoint_name_or_path = None
            self.config.load_checkpoint = self.get_last_checkpoint()
        elif isinstance(predict_checkpoint, str):
            self.config.model.model_config.checkpoint_name_or_path = None
            self.config.load_checkpoint = predict_checkpoint
        else:
            self.default_checkpoint_name_or_path = self.config.model.model_config.checkpoint_name_or_path
            if auto_trans_ckpt:
                if self.is_model_instance:
                    logger.warning(
                        "When a model instance is identified,"
                        "the weights that are currently proposed to be evaluated are specified"
                        "by the eval_checkpoint argument or a model instance "
                        "with the model weights already loaded is imported")
                else:
                    self.config.load_checkpoint = self.config.model.model_config.checkpoint_name_or_path
                self.config.model.model_config.checkpoint_name_or_path = None
            else:
                self.config.load_checkpoint = None

        if auto_trans_ckpt is not None:
            self.config.auto_trans_ckpt = auto_trans_ckpt

        if input_data is None:
            input_data = build_dataset_loader(self.config.eval_dataset.data_loader)
            logger.info("dataset by config is used as input_data.")

        assert isinstance(input_data, (GeneratorDataset, BaseDataset, RepeatDataset, BatchDataset, Tensor,
                                       np.ndarray, Image, str, list)), \
            "Input data's type must be one of [GeneratorDataset," \
            " str, ms.Tensor, np.ndarray, PIL.Image.Image]"

        if self.is_model_instance:
            self._reset_model_instance(is_train=False)

        output_result = self.trainer.predict(
            config=self.config, input_data=input_data,
            network=self.model, image_processor=self.image_processor,
            audio_processor=self.audio_processor,
            tokenizer=self.tokenizer,
            is_full_config=True,
            **kwargs)
        return output_result

    @args_type_check(predict_checkpoint=(str, bool), auto_trans_ckpt=bool)
    def export(self,
               predict_checkpoint: Optional[Union[str, bool]] = None,
               auto_trans_ckpt: Optional[bool] = None):
        """
        The export API of Trainer. After setting custom settings, implement export by calling the
        export method of task-trainer instance.

        Args:
            predict_checkpoint (Optional[Union[str, bool]]):
                Used to predict the weight of the network.
                It supports real checkpoint path or valid model name of mindformers or bool value.
                if it's true, the last checkpoint file saved from the previous training round is automatically used.
                Default: False.
            auto_trans_ckpt:
                auto transform checkpoint to load in distributed model

        Return:
            None

        Raises:
            TypeError: if predict_checkpoint is not bool or str type.
        """
        if predict_checkpoint is not None and not isinstance(predict_checkpoint, (bool, str)):
            raise TypeError(f"predict_checkpoint must be one of [None, string, bool], "
                            f"but get {predict_checkpoint}")

        if self.task not in SUPPORT_PIPELINES.keys():
            raise NotImplementedError(f"The {self.task} not support predict, "
                                      f"now this tasks {SUPPORT_PIPELINES.keys()} is support predict.")

        if predict_checkpoint is False:
            predict_checkpoint = None

        if predict_checkpoint is True:
            self.config.model.model_config.checkpoint_name_or_path = None
            self.config.load_checkpoint = self.get_last_checkpoint()
        elif isinstance(predict_checkpoint, str):
            self.config.model.model_config.checkpoint_name_or_path = None
            self.config.load_checkpoint = predict_checkpoint
        else:
            self.default_checkpoint_name_or_path = self.config.model.model_config.checkpoint_name_or_path
            if auto_trans_ckpt:
                if self.is_model_instance:
                    logger.warning(
                        "When a model instance is identified,"
                        "the weights that are currently proposed to be evaluated are specified"
                        "by the eval_checkpoint argument or a model instance "
                        "with the model weights already loaded is imported")
                else:
                    self.config.load_checkpoint = self.config.model.model_config.checkpoint_name_or_path
                self.config.model.model_config.checkpoint_name_or_path = None
            else:
                self.config.load_checkpoint = None

        if auto_trans_ckpt is not None:
            self.config.auto_trans_ckpt = auto_trans_ckpt

        if self.is_model_instance:
            self._reset_model_instance(is_train=False)

        self.trainer.export(config=self.config,
                            network=self.model,
                            is_full_config=True)

    @args_type_check(data_parallel=int, model_parallel=int, expert_parallel=int, pipeline_stage=int,
                     micro_batch_interleave_num=int, micro_batch_num=int, use_seq_parallel=bool, optimizer_shard=bool,
                     gradient_aggregation_group=int, vocab_emb_dp=bool)
    def set_parallel_config(
            self, data_parallel=1, model_parallel=1, expert_parallel=1, pipeline_stage=1, micro_batch_interleave_num=1,
            micro_batch_num=1, use_seq_parallel=False, optimizer_shard=False,
            gradient_aggregation_group=4, vocab_emb_dp=True):
        """
        set_parallel_config for the setting global data parallel, model parallel and fusion group.
        The parallel configure setting for Trainer.

        Args:
            data_parallel (int):
                The data parallel way. The input data will be sliced into n parts for each layer
                according to the data parallel way. Default: 1.
            model_parallel (int):
                The model parallel way. The parameters of dense layers in Multi-head Attention and
                FeedForward layer will be sliced according to the model parallel way. Default: 1.
            expert_parallel (int):
                The expert parallel way. This is effective only when MoE (Mixture of Experts)
                is applied. This value specifies the number of partitions to split the experts into.
            pipeline_stage (int):
                The number of the pipeline stage. Should be a positive value. Default: 1.
            micro_batch_num (int):
                The micro size of the batches for the pipeline training. Default: 1.
            use_seq_parallel (bool):
                Whether to enable sequence parallel. Default False.
            optimizer_shard (bool):
                Whether to enable optimizer shard. Default False.
            gradient_aggregation_group (int):
                The fusion group size of the optimizer state sharding. Default: 4.
            vocab_emb_dp (bool):
                Shard embedding in model parallel or data parallel. Default: True.
            micro_batch_interleave_num (int):
                split num of batch size. Default: 1.

        Returns:
            None
        """
        self.config.parallel_config.data_parallel = data_parallel
        self.config.parallel_config.model_parallel = model_parallel
        self.config.parallel_config.expert_parallel = expert_parallel
        self.config.parallel_config.pipeline_stage = pipeline_stage
        self.config.parallel_config.use_seq_parallel = use_seq_parallel
        self.config.parallel_config.optimizer_shard = optimizer_shard
        self.config.parallel_config.micro_batch_num = micro_batch_num
        self.config.parallel_config.vocab_emb_dp = vocab_emb_dp
        self.config.parallel_config.gradient_aggregation_group = gradient_aggregation_group
        self.config.micro_batch_interleave_num = micro_batch_interleave_num

        self.is_set_parallel_config = True

    @args_type_check(recompute=bool, parallel_optimizer_comm_recompute=bool,
                     select_recompute=bool, mp_comm_recompute=bool,
                     recompute_slice_activation=bool)
    def set_recompute_config(self, recompute=False, parallel_optimizer_comm_recompute=False, select_recompute=False,
                             mp_comm_recompute=True, recompute_slice_activation=False):
        r"""
        Set recompute config.

        Args:
            recompute (bool):
                Enable recomputation of the transformer block or not. Default: False.
            select_recompute (bool):
                Only Enable recomputation of the attention layer or not. Default: False.
            parallel_optimizer_comm_recompute (bool):
                Specifies whether the communication operator allgathers
                introduced by optimizer shard are recomputed in auto parallel or semi auto parallel mode.
                Default: False.
            mp_comm_recompute (bool):
                Specifies whether the model parallel communication operators
                in the cell are recomputed in auto parallel or semi auto parallel mode. Default: True.
            recompute_slice_activation (bool):
                Slice the cell output which would remains in memory. Default: False.

        Returns:
            None
        """
        self.config.recompute_config.recompute = recompute
        self.config.recompute_config.select_recompute = select_recompute
        self.config.recompute_config.parallel_optimizer_comm_recompute = parallel_optimizer_comm_recompute
        self.config.recompute_config.mp_comm_recompute = mp_comm_recompute
        self.config.recompute_config.recompute_slice_activation = recompute_slice_activation

        self.is_set_recompute_config = True

    @args_type_check(expert_num=int, capacity_factor=float, aux_loss_factor=float, num_experts_chosen=int,
                     expert_group_size=int, group_wise_a2a=bool, comp_comm_parallel=bool, comp_comm_parallel_degree=int)
    def _set_moe_config(self,
                        expert_num=1,
                        capacity_factor=1.1,
                        aux_loss_factor=0.05,
                        num_experts_chosen=1,
                        expert_group_size=None,
                        group_wise_a2a=False,
                        comp_comm_parallel=False,
                        comp_comm_parallel_degree=2):
        """
        Sef the configuration of MoE (Mixture of Expert).

        Args:
            expert_num (int):
                The number of experts employed. Default: 1
            capacity_factor (float):
                The factor is used to indicate how much to expand expert capacity,
                which is >=1.0. Default: 1.1.
            aux_loss_factor (float):
                The factor is used to indicate how much the load balance loss (produced by the
                router) to be added to the entire model loss, which is < 1.0. Default: 0.05.
            num_experts_chosen (int):
                The number of experts is chosen by each token, it should not be larger
                than expert_num. Default: 1.
            expert_group_size (int):
                The number of tokens in each data parallel group. Default: None. This parameter is
                effective only when in AUTO_PARALLEL mode, and NOT SHARDING_PROPAGATION.
            group_wise_a2a (bool):
                Whether to enable group-wise alltoall communication, which can reduce communication
                time by converting part of intercommunication into intra communication. Default: False. This parameter
                is effective only when model parallel > 1 and data_parallel equal to expert parallel.
            comp_comm_parallel (bool):
                Whether to enable ffn compute and communication parallel, which can reduce pure
                communication time by splitting and overlapping compute and communication. Default: False.
            comp_comm_parallel_degree (int):
                The split number of compute and communication. The larger the numbers,
                the more overlap there will be but will consume more memory. Default: 2. This parameter is effective
                only when comp_comm_parallel enable.
        Returns:
            None
        """
        self.config.moe_config.expert_num = expert_num
        self.config.moe_config.capacity_factor = capacity_factor
        self.config.moe_config.aux_loss_factor = aux_loss_factor
        self.config.moe_config.num_experts_chosen = num_experts_chosen
        self.config.moe_config.expert_group_size = expert_group_size
        self.config.moe_config.group_wise_a2a = group_wise_a2a
        self.config.moe_config.comp_comm_parallel = comp_comm_parallel
        self.config.moe_config.comp_comm_parallel_degree = comp_comm_parallel_degree

        self.is_set_moe_config = True

    def _reset_model_instance(self, is_train=True):
        """Reset model instance for new model config."""
        if True not in [self.is_set_parallel_config, self.is_set_moe_config, self.is_set_recompute_config]:
            return

        if self.is_set_parallel_config:
            logger.info("The incoming model will be configured in parallel.")

        if self.is_set_recompute_config:
            logger.info("The incoming model will be configured in recompute.")

        if self.is_set_moe_config:
            logger.info("The incoming model will be configured in moe.")

        if not isinstance(self.model, BaseModel):
            raise NotImplementedError("Currently only the integrated model structure in MindFormers is supported.")

        build_parallel_config(self.config)
        model_config = self.model.config
        model_config.parallel_config = self.config.parallel_config
        model_config.moe_config = self.config.moe_config
        self.model.__init__(model_config)
        self.model.set_train(is_train)

    def get_train_dataloader(self):
        """get train dataloader of mindspore."""
        return build_dataset_loader(self.config.train_dataset.data_loader)

    def get_eval_dataloader(self):
        """get eval dataloader of mindspore."""
        return build_dataset_loader(self.config.eval_dataset.data_loader)

    def get_last_checkpoint(self):
        """get last checkpoint for resuming or finetune."""
        output_folder = self.config.output_dir
        checkpoint_dir = os.path.join(
            output_folder, DEFAULT_CHECKPOINT_DIR, 'rank_{}'.format(self.rank_id))
        output_checkpoint_path = [
            checkpoint for checkpoint in os.listdir(checkpoint_dir)
            if checkpoint.endswith('.ckpt')
        ]
        if not output_checkpoint_path:
            return None
        output_checkpoint_path = sorted(output_checkpoint_path,
                                        key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
        return os.path.join(checkpoint_dir, output_checkpoint_path[-1])

    def _save_config_to_yaml(self, config: dict = None):
        """save now config file to yaml file."""
        if config is None:
            config = self.config
        model_name = self.config.trainer.model_name
        config_dict = _reset_config_for_save(config)
        config_dir = os.path.join(
            self.configs_directory, model_name.lower() + '_new')
        if not os.path.exists(config_dir):
            os.makedirs(config_dir, exist_ok=True)
        run_yaml_path = os.path.join(config_dir, 'run_{}.yaml'.format(model_name.lower()))

        _save_config_to_yaml(run_yaml_path, config_dict)

    def _load_model_checkpoint(self):
        """Load model checkpoint to network."""
        checkpoint_name_or_path = self.config.model.model_config.checkpoint_name_or_path
        if checkpoint_name_or_path is None:
            logger.warning("checkpoint_name_or_path is None, not load input checkpoint.")
        elif isinstance(checkpoint_name_or_path, str):
            is_exist_path = os.path.exists(checkpoint_name_or_path)
            is_checkpoint_name = checkpoint_name_or_path in SUPPORT_MODEL_NAMES
            if is_exist_path:
                logger.info("now input valid checkpoint path, it will load to network.")
                checkpoint_dict = load_checkpoint(checkpoint_name_or_path)
                not_load_params = load_param_into_net(self.model, checkpoint_dict)
                logger.info("not load parameters is: %s", str(not_load_params))
            elif is_checkpoint_name:
                logger.info("now input valid checkpoint name, it will load to network.")
                if isinstance(self.model, (Cell, BaseModel)):
                    self.model.load_checkpoint(self.config.model.model_config)
                else:
                    logger.warning("model must be BaseModel or Cell type, but get %s", type(self.model))
            else:
                logger.warning("input checkpoint args is invalid, "
                               "it must be valid and real checkpoint path or a valid checkpoint name,"
                               "but get %s", checkpoint_name_or_path)
        else:
            raise TypeError(f"checkpoint_name_or_path type error, "
                            f"it should be one of [None, str], "
                            f"but get {type(checkpoint_name_or_path)}")

    def _check_checkpoint_config(self, checkpoint: Optional[Union[str, bool]] = None):
        """check checkpoint config."""
        if checkpoint is True:
            self.config.model.model_config.checkpoint_name_or_path = self.get_last_checkpoint()
        elif isinstance(checkpoint, str):
            self.config.model.model_config.checkpoint_name_or_path = checkpoint
        else:
            if self.default_checkpoint_name_or_path is not None:
                self.config.model.model_config.checkpoint_name_or_path = self.default_checkpoint_name_or_path

    def _config_type_check(self, config):
        if config.resume_training is not None and not isinstance(config.resume_training, bool):
            raise TypeError(f"resume_training must be bool, "
                            f"but get {config.resume_training}")
        if config.auto_trans_ckpt is not None and not isinstance(config.auto_trans_ckpt, bool):
            raise TypeError(f"auto_trans_ckpt must be bool, "
                            f"but get {config.auto_trans_ckpt}")


def _save_config_to_yaml(save_file_path: str = None, save_config: dict = None):
    """
    Save config to yaml file.

    Args:
        save_file_path (str): The real path to save yaml file. Default: None.
        save_config (dict): The task config. Default: None.

    Returns:
        None
    """
    if save_config is None:
        save_config = {}
    with open(save_file_path, 'w', encoding='utf-8') as file_pointer:
        file_pointer.write(
            ordered_yaml_dump(
                save_config,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False))


def _reset_config_for_save(config: dict = None):
    """
    Reset config according to the yaml file number.

    Args:
        config (dict): The task config. Default: None.

    Returns:
        the reset config
    """
    if config is None:
        config = {}
    config = config.copy()

    config_dict = OrderedDict()

    if config.get('model') is not None:
        model_config = config2dict(config.pop('model'))
        config_dict.setdefault('model', model_config)

    if config.get('processor') is not None:
        processor_config = config2dict(config.pop('processor'))
        config_dict.setdefault('processor', processor_config)

    if config.get('train_dataset_task') is not None and config.get('train_dataset') is not None:
        train_dataset_config = config2dict(config.pop('train_dataset'))
        train_dataset_task_config = config2dict(config.pop('train_dataset_task'))
        config_dict.setdefault('train_dataset', train_dataset_config)
        config_dict.setdefault('train_dataset_task', train_dataset_task_config)

    if config.get('eval_dataset_task') is not None and config.get('eval_dataset') is not None:
        eval_dataset_config = config2dict(config.pop('eval_dataset'))
        eval_dataset_task_config = config2dict(config.pop('eval_dataset_task'))
        config_dict.setdefault('train_dataset', eval_dataset_config)
        config_dict.setdefault('train_dataset_task', eval_dataset_task_config)

    if config.get('context') is not None:
        context_config = config2dict(config.pop('context'))
        parallel_context_config = config2dict(config.pop('parallel'))
        moe_conifg = config2dict(config.pop('moe_config'))
        recompute_config = config2dict(config.pop('recompute_config'))
        parallel_config = config2dict(config.pop('parallel_config'))
        config_dict.setdefault('context', context_config)
        config_dict.setdefault('parallel', parallel_context_config)
        config_dict.setdefault('moe_conifg', moe_conifg)
        config_dict.setdefault('recompute_config', recompute_config)
        config_dict.setdefault('parallel_config', parallel_config)

    if config.get('runner_config') is not None:
        runner_config = config2dict(config.pop('runner_config'))
        config_dict.setdefault('runner_config', runner_config)

    if config.get('runner_wrapper') is not None:
        wrapper_config = config2dict(config.pop('runner_wrapper'))
        config_dict.setdefault('runner_wrapper', wrapper_config)

    if config.get('optimizer') is not None:
        optim_config = config2dict(config.pop('optimizer'))
        config_dict.setdefault('optimizer', optim_config)

    if config.get('lr_schedule') is not None:
        lr_config = config2dict(config.pop('lr_schedule'))
        config_dict.setdefault('lr_schedule', lr_config)

    if config.get('callbacks') is not None:
        cb_config = config2dict(config.pop('callbacks'))
        config_dict.setdefault('callbacks', cb_config)

    run_config = config2dict(config)
    for key, value in run_config.items():
        config_dict.setdefault(key, value)

    return config_dict
