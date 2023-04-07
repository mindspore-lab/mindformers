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
"""Base Trainer."""
import os
import shutil
from typing import Optional, Union, List

import mindspore as ms
from mindspore.train.model import Model
from mindspore.train import Callback
from mindspore.dataset import GeneratorDataset
from mindspore.nn.wrap.cell_wrapper import _VirtualDatasetCell
from mindspore.nn import TrainOneStepCell, Optimizer, Cell, \
    PipelineCell, MicroBatchInterleaved

from mindformers.mindformer_book import MindFormerBook
from mindformers.core import build_lr, build_optim, build_callback, build_metric
from mindformers.core.parallel_config import build_parallel_config
from mindformers.dataset import build_dataset, check_dataset_config, BaseDataset
from mindformers.models import build_model, build_processor, build_tokenizer, BaseModel
from mindformers.wrapper import build_wrapper
from mindformers.tools.register import MindFormerConfig
from mindformers.tools.logger import logger
from mindformers.tools.utils import count_params
from .config_args import ConfigArguments
from .training_args import TrainingArguments
from .utils import check_runner_config, resume_checkpoint_for_training
from .optimizer_grouped_parameters import get_optimizer_grouped_parameters
from .utils import set_seed, check_train_data_loader_type, \
    check_eval_data_loader_type, check_optimizer_and_lr_type, check_wrapper_config

SUPPORT_TASKS = MindFormerBook().get_trainer_support_task_list()
SUPPORT_MODEL_NAMES = MindFormerBook().get_model_name_support_list()
SUPPORT_PIPELINES = MindFormerBook().get_pipeline_support_task_list()
SUPPORT_PIPELINE_INPUT_DATA = MindFormerBook().get_pipeline_support_input_data_list()
CURRENT_PROJECT_PATH = MindFormerBook().get_project_path()
DEFAULT_CONFIG_DIR = 'configs'


class BaseTrainer:
    r"""Base Task Trainer.
    Args:
        task (str): The task name supported.
        model_name (str): The model name of Task-Trainer. Default: None
    Raises:
        NotImplementedError: If train method or evaluate method or predict method not implemented.
    """

    def __init__(self, task: str = None, model_name: str = None):

        if model_name is None:
            model_name = "model name unspecified."
        if task is None:
            task = "task name unspecified."
        logger.info("Now Running Task is: %s, Model is: %s", task, model_name)

        self.model_name = model_name
        self.task = task
        self.config = None
        self.default_task_config = None

        self.train_dataset = None
        self.eval_dataset = None
        self.network = None
        self.optimizer = None
        self.image_processor = None
        self.audio_processor = None
        self.tokenizer = None
        self.callbacks = None
        self.eval_callbacks = None
        self.model_wrapper = None
        self.compute_metrics = None
        self.kwargs = None

        if not os.path.exists(os.path.join('.', DEFAULT_CONFIG_DIR)):
            configs_directory = os.path.join('.', DEFAULT_CONFIG_DIR)
            if os.path.exists(os.path.join(CURRENT_PROJECT_PATH, DEFAULT_CONFIG_DIR)):
                mindformers_configs_directory = os.path.join(CURRENT_PROJECT_PATH, DEFAULT_CONFIG_DIR)
                shutil.copytree(mindformers_configs_directory, configs_directory)

        if task not in SUPPORT_TASKS.keys():
            logger.warning("Input task name is not in the supported list or unspecified.")

        if task in SUPPORT_TASKS.keys() and model_name not in SUPPORT_TASKS.get(task).keys():
            logger.warning("Input model name is not in the supported list or unspecified.")
            logger.warning("See the list of supported task and model name: %s", SUPPORT_TASKS)
            logger.warning("The default model config: %s will now be used for the %s task ",
                           SUPPORT_TASKS.get(self.task).get("common"), task)
            self.model_name = "common"

    def set_config(self,
                   config: Optional[Union[dict, str, ConfigArguments, TrainingArguments]] = None,
                   is_full_config: bool = False):
        """Set the task config for task trainer."""
        if config is not None:
            if is_full_config:
                self.config = config
            else:
                self.setup_task_config()
                self.config = self._merge_config(config)
        else:
            self.setup_task_config()
            self.config = self.default_task_config
        build_parallel_config(self.config)
        self._check_global_batch_size_for_auto_parallel()

        # del SummaryMonitor, or it will crash
        for index, callbacks in enumerate(self.config.callbacks):
            if callbacks["type"] == "SummaryMonitor":
                del self.config.callbacks[index]

        return self.config

    def setup_task_config(self):
        """Setup the default task config."""
        task_config = None
        if self.task in SUPPORT_TASKS.keys() and self.model_name in SUPPORT_TASKS.get(self.task).keys():
            task_config = MindFormerConfig(SUPPORT_TASKS.get(self.task).get(self.model_name))

        if isinstance(task_config, MindFormerConfig):
            self.default_task_config = task_config
        else:
            logger.warning("If the default config arguments is not specified,"
                           "you must define the required model or wrapper, optimizer, and so on"
                           "in the train or evaluate or predict attribute function.")

    def _check_global_batch_size_for_auto_parallel(self):
        """Check global batch size in auto parallel mode."""
        batch_size = self.config.runner_config.batch_size
        dp = self.config.parallel_config.data_parallel
        micro_batch_num = self.config.parallel_config.micro_batch_num
        parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        full_batch = ms.get_auto_parallel_context("full_batch")
        pp = self.get_pipeline_stages()

        if parallel_mode in ["semi_auto_parallel", "auto_parallel"]:
            if full_batch:
                if pp > 1:
                    logger.info("Pipeline parallel was opened: pipeline_stages = %s, full batch is True, "
                                "batch size will be changed: "
                                "batch_size = batch_size * data_parallel * micro_batch_num = %s * %s * %s = %s).",
                                pp, batch_size, dp, micro_batch_num, batch_size * dp * micro_batch_num)
                    self.config.runner_config.batch_size = batch_size * dp * micro_batch_num
                else:
                    logger.info("The current parallel mode is %s, full batch is True, so batch size will be changed: "
                                "batch_size = batch_size * data_parallel = %s * %s = %s",
                                parallel_mode, batch_size, dp, batch_size * dp)
                    self.config.runner_config.batch_size = batch_size * dp
            else:
                if pp > 1:
                    logger.info("Pipeline parallel was opened: pipeline_stages = %s, full batch is False, "
                                "batch size will be changed: "
                                "batch_size = batch_size * micro_batch_num = %s * %s = %s).",
                                pp, batch_size, micro_batch_num, batch_size * micro_batch_num)
                    self.config.runner_config.batch_size = batch_size * micro_batch_num
        else:
            logger.info("The current parallel mode is %s, batch size will not be changed: batch_size = %s",
                        parallel_mode, batch_size)

    def _reset_dataset_batch_size(self):
        """Reset dataset batch size according to the global batch size of runner config."""
        check_dataset_config(self.config)

    def _merge_config(self, args):
        """Merge config from default task config."""
        if self.default_task_config is None:
            logger.warning("default task config is None, you will not be able to merge config parameters.")
            return args

        if isinstance(args, dict):
            self.default_task_config.merge_from_dict(args)
        elif isinstance(args, str):
            if not (os.path.realpath(args) and os.path.exists(args)):
                raise FileNotFoundError(f"config path must be exist, but get {args}.")
            if not args.endswith(('.yaml', '.yml')):
                raise ValueError(f"config file must be end with .yaml or .yml, but get {args}")
            self.default_task_config = MindFormerConfig(args)
        elif isinstance(args, ConfigArguments):
            if hasattr(args, 'train_dataset'):
                check_train_data_loader_type(args, self.default_task_config)
            if hasattr(args, 'eval_dataset'):
                check_eval_data_loader_type(args, self.default_task_config)
            if hasattr(args, 'optimizer'):
                check_optimizer_and_lr_type(args, self.default_task_config)
            if hasattr(args, 'runner_wrapper'):
                check_wrapper_config(args, self.default_task_config)
            self.default_task_config.merge_from_dict(args.__dict__)
        elif isinstance(args, TrainingArguments):
            args.convert_args_to_mindformers_config(self.default_task_config)
        else:
            logger.warning(
                "The type of config parameter to merge is not supported, "
                "currently supported types are [dict, config_path(str), ConfigArguments, TrainingArguments], "
                "but get %s", type(args))
        return self.default_task_config

    def create_train_dataset(self, default_args: dict = None):
        """Create the train dataset for training."""
        logger.info(".........Build Dataset From Config..........")
        self._reset_dataset_batch_size()
        train_dataset = build_dataset(self.config.train_dataset_task, default_args=default_args)
        return train_dataset

    def create_eval_dataset(self, default_args: dict = None):
        """Create the eval dataset for evaluate."""
        logger.info(".........Build Dataset From Config..........")
        self._reset_dataset_batch_size()
        eval_dataset = build_dataset(self.config.eval_dataset_task, default_args=default_args)
        return eval_dataset

    def create_network(self, default_args: dict = None):
        """Create the network for task trainer."""
        logger.info(".........Build Network From Config..........")
        network = build_model(self.config.model, default_args=default_args)
        return network

    def create_pipeline_network(self, default_args: dict = None):
        """Create the network of pipeline parallel for task trainer."""
        logger.info(".........Build Pipeline Network From Config..........")
        network = build_model(self.config.model, default_args=default_args)
        micro_batch_interleave_num = self.config.micro_batch_interleave_num
        micro_batch_num = self.config.parallel_config.micro_batch_num
        if micro_batch_interleave_num > 1:
            logger.info("micro_batch_interleave_num > 1, the double copy parallel feature is turned on.")
            network = PipelineCell(MicroBatchInterleaved(network, micro_batch_interleave_num),
                                   micro_size=micro_batch_num)
        else:
            network = PipelineCell(network, micro_size=micro_batch_num)
        network = _VirtualDatasetCell(network)

        if isinstance(network, (Cell, BaseModel)):
            network.set_train(True)

        if self.config.runner_wrapper is not None:
            self.config.runner_wrapper.type = "MFPipelineWithLossScaleCell" \
                if self.config.runner_wrapper.type != "MFPipelineWithLossScaleCell" else self.config.runner_wrapper.type
            self.config.runner_wrapper.micro_batch_num = self.config.parallel_config.micro_batch_num
            logger.warning("When using the pipeline parallel mode, "
                           "the MFPipelineWithLossScaleCell class is used by default.")
        else:
            logger.warning("When using the pipeline parallel mode, "
                           "because the wrapper class is not specified, "
                           "MindSpore's built-in PipelineCell is used by default")
        return network

    def create_image_processor(self, default_args: dict = None):
        """Create the image processor for predict."""
        logger.info(".........Build Image Processor From Config..........")
        self.image_processor = build_processor(
            self.config.processor.image_processor, default_args=default_args)
        return self.image_processor

    def create_tokenizer(self, default_args: dict = None):
        """Create the tokenizer for task trainer."""
        logger.info(".........Build Text Tokenizer From Config..........")
        self.tokenizer = build_tokenizer(
            self.config.processor.tokenizer, default_args=default_args)
        return self.tokenizer

    def create_optimizer_scheduler(self, network, layer_scale=False):
        """Create the optimizer for training."""
        logger.info(".........Build Optimizer From Config..........")
        # learning rate scale for multi-nodes training
        learning_scale = self.config.lr_scale
        scale_factor = self.config.lr_scale_factor

        # build learning rate schedule
        lr_schedule = self.create_lr_scheduler(learning_scale, scale_factor)

        weight_decay = self.config.optimizer.weight_decay if self.config.optimizer.weight_decay else 0.
        layer_decay = self.config.layer_decay if self.config.layer_decay else 1.0
        group_params = get_optimizer_grouped_parameters(network,
                                                        weight_decay,
                                                        lr_schedule,
                                                        layer_scale=layer_scale,
                                                        layer_decay=layer_decay)
        if lr_schedule is not None:
            self.optimizer = build_optim(
                self.config.optimizer,
                default_args={"params": group_params,
                              "learning_rate": lr_schedule})
        else:
            if self.config.optimizer.learning_rate is None:
                raise ValueError("learning_rate must be input")
            self.config.optimizer.learning_rate = self.learning_rate_scale(
                self.config.optimizer.learning_rate, scale_factor) \
                if learning_scale and scale_factor is not None else self.config.optimizer.learning_rate
            self.optimizer = build_optim(
                self.config.optimizer,
                default_args={"params": group_params})
        return self.optimizer

    def create_lr_scheduler(self, learning_scale: bool = False, scale_factor: int = 256):
        """Create the learning rate scheduler."""
        logger.info(".........Build LR Schedule From Config..........")
        train_data_size = self.get_train_data_size()

        if self.config.lr_schedule:
            warmup_epochs = self.config.lr_schedule.pop("warmup_epochs", None)
            warmup_ratio = self.config.lr_schedule.pop("warmup_ratio", None)

            if not self.config.runner_config.sink_mode:
                total_steps = int(self.config.runner_config.epochs * train_data_size)
            else:
                total_steps = int(self.config.runner_config.epochs * self.config.runner_config.per_epoch_size)

            if warmup_epochs is not None and warmup_ratio is not None:
                logger.warning("warmup_epochs and warmup_ratio are set simultaneously,"
                               "warmup_ratio takes precedence.")
                warmup_epochs = None

            if warmup_epochs is not None:
                logger.warning("warmup_epochs was set in lr_schedule,"
                               "it will multiply the data size to represent the warmup steps")
                self.config.lr_schedule.warmup_steps = int(warmup_epochs * train_data_size)

            if warmup_ratio is not None:
                self.config.lr_schedule.warmup_steps = int(total_steps * warmup_ratio)

            self.config.lr_schedule.total_steps = total_steps \
                if self.config.lr_schedule.total_steps is None or self.config.lr_schedule.total_steps == -1 \
                else int(self.config.lr_schedule.total_steps)

            self.config.lr_schedule.learning_rate = self.learning_rate_scale(
                self.config.lr_schedule.learning_rate, scale_factor) \
                if learning_scale and scale_factor is not None else self.config.lr_schedule.learning_rate
        lr_schedule = build_lr(self.config.lr_schedule)
        return lr_schedule

    def create_model_wrapper(self, network, optimizer):
        """Create the model wrapper for training."""
        logger.info(".........Build Model Wrapper for Train From Config..........")
        model_wrapper = build_wrapper(self.config.runner_wrapper,
                                      default_args={"network": network, "optimizer": optimizer})
        return model_wrapper

    def create_callbacks(self, default_args: dict = None):
        """Create the callback list for training."""
        logger.info(".........Build Callbacks for Train From Config..........")
        self.callbacks = []
        if self.config.profile:
            self.callbacks.append(self.config.profile_cb)
        self.callbacks.extend(build_callback(self.config.callbacks, default_args=default_args))
        return self.callbacks

    def create_eval_callbacks(self, default_args: dict = None):
        """Create the eval callback list for training."""
        logger.info(".........Build Callbacks for Evaluate From Config..........")
        self.eval_callbacks = []
        self.eval_callbacks.extend(build_callback(self.config.eval_callbacks, default_args=default_args))
        return self.eval_callbacks

    def create_metrics(self, metric_name: str = None):
        """Create Metrics For Evaluate or Fit."""
        if metric_name is None:
            metric_name = self.model_name + "_metric"

        self.compute_metrics = {metric_name: build_metric(self.config.metric)}
        return self.compute_metrics

    def count_parameters(self):
        """Count network parameters number."""
        if self.network is not None:
            logger.info("Network Parameters: %s M.", str(count_params(self.network)))
        elif self.model_wrapper is not None:
            logger.info("Network Parameters: %s M.", str(count_params(self.model_wrapper.network)))
        else:
            logger.warning("Network is None, parameters incalculable.")

    def set_seed(self, seed: int = None):
        """Set seed for training."""
        if seed is None:
            if self.config.seed is None:
                raise ValueError("seed is not set in config, it is None.")
            set_seed(self.config.seed)
        else:
            set_seed(seed)

    def set_train_dataset(self, dataset):
        """Set the attribute of train dataset."""
        if dataset is None:
            raise ValueError("Train dataset is None")
        self.train_dataset = dataset

    def set_eval_dataset(self, dataset):
        """Set the attribute of eval dataset ."""
        if dataset is None:
            raise ValueError("Eval dataset is None")
        self.eval_dataset = dataset

    def set_network(self, network, is_train: bool = True):
        """Set the attribute of network."""
        if network is None:
            raise ValueError("network is None")
        if isinstance(network, (Cell, BaseModel)):
            network.set_train(is_train)
        self.network = network

    def set_model_wrapper(self, model_wrapper):
        """Set the attribute of model_wrapper."""
        if model_wrapper is None:
            raise ValueError("model wrapper is None")
        self.model_wrapper = model_wrapper

    def get_train_data_size(self):
        """Get train dataset size."""
        if self.train_dataset is None:
            raise NotImplementedError("train dataset is None")
        return self.train_dataset.get_dataset_size()

    def get_eval_data_size(self):
        """Get eval dataset size."""
        if self.eval_dataset is None:
            raise NotImplementedError("train dataset is None")
        return self.eval_dataset.get_dataset_size()

    def get_pipeline_stages(self):
        """Get pipeline stages for task trainer."""
        pipeline_stages = self.config.parallel_config.pipeline_stage or ms.get_auto_parallel_context("pipeline_stages")
        return pipeline_stages

    def learning_rate_scale(self, base_learning_rate: float = 0., scale_factor: Optional[Union[float, int]] = 256.):
        """Scale learning rate for training."""
        if not isinstance(base_learning_rate, float):
            raise ValueError(f"learning rate must be float type, but get {type(base_learning_rate)}")
        if not isinstance(scale_factor, (float, int)):
            raise ValueError(f"scale_factor must be float or int type, but get {type(scale_factor)}")

        device_num = int(os.getenv("RANK_SIZE", "1"))
        per_device_batch_size = self.config.train_dataset.batch_size
        learning_rate = (base_learning_rate * device_num * per_device_batch_size) / scale_factor
        return learning_rate

    def training_process(
            self,
            config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
            network: Optional[Union[Cell, BaseModel]] = None,
            dataset: Optional[Union[BaseDataset, GeneratorDataset]] = None,
            optimizer: Optional[Optimizer] = None,
            wrapper: Optional[TrainOneStepCell] = None,
            callbacks: Optional[Union[Callback, List[Callback]]] = None,
            **kwargs):
        """Train or Fine-tune for BaseTrainer in MindFormers."""
        self.kwargs = kwargs
        is_full_config = kwargs.get("is_full_config", False)
        config = self.set_config(config, is_full_config)

        # build dataset
        logger.info(".........Build Dataset For Train..........")
        if dataset is None:
            dataset = self.create_train_dataset()
        self.set_train_dataset(dataset)
        check_runner_config(config, dataset)

        # build network
        logger.info(".........Build Net For Train..........")
        if network is None and wrapper is None:
            if self.get_pipeline_stages() > 1:
                network = self.create_pipeline_network(default_args={"parallel_config": config.parallel_config,
                                                                     "moe_config": config.moe_config})
            else:
                network = self.create_network(default_args={"parallel_config": config.parallel_config,
                                                            "moe_config": config.moe_config})
        if network is not None:
            self.set_network(network, is_train=True)

        if wrapper is not None:
            self.set_model_wrapper(wrapper)

        self.count_parameters()

        # build optimizer
        logger.info(".........Build Optimizer For Train..........")
        if optimizer is None and wrapper is None:
            optimizer = self.create_optimizer_scheduler(network, layer_scale=config.layer_scale)

        # build callback
        logger.info(".........Build Callbacks For Train..........")
        if callbacks is None:
            callbacks = self.create_callbacks(default_args={
                "learning_rate": optimizer.learning_rate if optimizer else wrapper.optimizer.learning_rate,
                "origin_epochs": config.runner_config.origin_epochs,
                "dataset_size": config.data_size,
                "micro_batch_num": config.parallel_config.micro_batch_num})

        # resume checkpoint
        if config.resume_or_finetune_checkpoint:
            logger.info(".............Start resume training from checkpoint..................")
            resume_checkpoint_for_training(config, network, optimizer)

        # build model wrapper
        if wrapper is None:
            logger.info(".........Build Running Wrapper From Config For Train..........")
            wrapper = self.create_model_wrapper(network, optimizer)

        # define Model and begin training
        logger.info(".........Starting Init Train Model..........")
        if wrapper is not None:
            model = Model(wrapper)
        else:
            model = Model(network, optimizer=optimizer)

        logger.info(".........Starting Training Model..........")
        logger.info(".........Model Compiling, Please Wait a Moment...........")
        model.train(config.runner_config.epochs, dataset,
                    callbacks=callbacks,
                    dataset_sink_mode=config.runner_config.sink_mode,
                    sink_size=config.runner_config.per_epoch_size,
                    initial_epoch=config.runner_config.initial_epoch)
        logger.info(".........Training Over!.............")

    def evaluate_process(
            self,
            config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
            network: Optional[Union[Cell, BaseModel]] = None,
            dataset: Optional[Union[BaseDataset, GeneratorDataset]] = None,
            callbacks: Optional[Union[Callback, List[Callback]]] = None,
            compute_metrics: Optional[Union[dict, set]] = None,
            **kwargs):
        """Evaluate for BaseTrainer in MindFormers."""
        metric_name = kwargs.get("metric_name")
        is_full_config = kwargs.get("is_full_config", False)
        config = self.set_config(config, is_full_config)

        # build dataset
        logger.info(".........Build Dataset For Evaluate..........")
        if dataset is None:
            dataset = self.create_eval_dataset()
        self.set_eval_dataset(dataset)

        # build network
        if network is None:
            network = self.create_network(default_args={"parallel_config": config.parallel_config,
                                                        "moe_config": config.moe_config})
        self.set_network(network, is_train=False)

        self.count_parameters()

        # build metric
        logger.info(".........Build Compute Metrics For Evaluate..........")
        if compute_metrics is None:
            compute_metrics = self.create_metrics(metric_name=metric_name)

        # build callback
        logger.info(".........Build Callbacks For Evaluate..........")
        if callbacks is None:
            callbacks = self.create_eval_callbacks()

        logger.info(".........Starting Init Evaluate Model..........")
        model = Model(network, metrics=compute_metrics, eval_network=network)

        logger.info(".........Starting Evaluate Model..........")
        output = model.eval(dataset,
                            callbacks=callbacks,
                            dataset_sink_mode=config.runner_config.sink_mode)
        logger.info('%s = %s', metric_name, str(output))
        logger.info(".........Evaluate Over!.............")
