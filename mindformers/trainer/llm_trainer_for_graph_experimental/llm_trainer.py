# Copyright 2025 Huawei Technologies Co., Ltd
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
"""LLMTrainer For Graph Mode."""
import os
import subprocess
from pprint import pprint
from typing import Optional, Union, List

import numpy as np

import mindspore as ms
from mindspore import ParallelMode, nn
from mindspore.common import set_seed
from mindspore.communication import get_group_size, get_rank, get_local_rank_size
from mindspore.dataset import GeneratorDataset
from mindspore.mint.distributed import barrier
from mindspore.nn import MicroBatchInterleaved
from mindspore.nn.utils import no_init_parameters
from mindspore.parallel.strategy import enable_save_strategy_online

from transformers import AutoTokenizer

from mindformers.checkpoint.checkpoint import load_checkpoint, get_checkpoint_path, CommonInfo
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.checkpoint.utils import compile_model
from mindformers.core import build_lr, build_optim
from mindformers.core.callback.callback import (
    MFLossMonitor,
    CheckpointMonitor,
    TopkBiasBalanceCallback,
    TrainCallBack,
    MaxLogitsMonitor,
)
from mindformers.dataset.llm_dataset import LLMDataset
from mindformers.models import build_network
from mindformers.pipeline import TextGenerationPipeline
from mindformers.tools.logger import logger
from mindformers.tools.register import (MindFormerConfig,
                                        MindFormerModuleType,
                                        MindFormerRegister)
from mindformers.tools.register.llm_template_v2 import TrainingParallelConfig
from mindformers.tools.utils import FILE_PERMISSION
from mindformers.trainer.optimizer_grouped_parameters import get_optimizer_grouped_parameters
from mindformers.utils import is_hf_safetensors_dir
from mindformers.utils.file_utils import set_output_path
from mindformers.utils.load_checkpoint_utils import process_hf_checkpoint
from mindformers.version_control import (check_is_reboot_node, check_tft_valid,
                                         check_tre_valid, check_tsp_valid)
from mindformers.wrapper.virtual_dataset_wrapper import _VirtualDatasetCell
from mindformers.wrapper.wrapper import (GradAccumulationCellWithMultiOutputs,
                                         MFPipelineWithLossScaleCell,
                                         MFTrainOneStepCell,
                                         PipelineCellWithMultiOutputs,
                                         DataOrderWrapperCell)

from .global_config import MFGlobalConfig

CALLBACK_HAS_SORT = [
    MFLossMonitor, CheckpointMonitor
]

NOT_LOAD_ANY_CHECKPOINT_MODE = "not_load_any_ckpt"

__all__ = ['LLMTrainer']



@MindFormerRegister.register(MindFormerModuleType.TRAINER, legacy=False)
class LLMTrainer:
    """
    LLM Model Trainer Class.
    This class provides training and inference capabilities for Large Language Models,
    handling dataset creation, model building, optimizer setup, and training loop execution.
    """
    def __init__(self) -> None:
        self.llm_model = None
        self.train_dataset = None
        self.callbacks = []
        self.global_batch_size = None
        self.dataset_batch_size = None
        self.predict_batch_size = None
        self.append_restore_info = None
        self.common_restore_info = None
        self.network_delay_inited = False
        self.optimizer_delay_inited = False
        self.lr_scheduler = None
        self.grouped_lr_scheduler = None

    def _setup_config(self, config: MindFormerConfig, is_train: bool = True) -> None:
        """Initialize and setup configuration for training or inference.

        This method sets up the configuration based on whether it's for training or inference mode.
        For training, it configures parallel context, batch sizes, and other training-specific settings.
        For inference, it validates parallel mode and sets data parallel size.

        Args:
            config (MindFormerConfig): Configuration object containing all training/inference settings.
            is_train (bool): Flag indicating whether setup is for training (True) or inference (False).
                            Defaults to True.

        Raises:
            ValueError: If config is None.
            TypeError: If config is not an instance of MindFormerConfig.
            RuntimeError: If parallel mode is not supported for the specified mode.
        """
        if config is None:
            raise ValueError("Configuration must be provided, but received None.")

        if not isinstance(config, MindFormerConfig):
            raise TypeError(f"Configuration must be of type MindFormerConfig, but received {type(config)}.")

        self.config = config
        self._set_model_config_adapter_old_format()

        if is_train:
            self._set_runner_seed(seed=self.config.training_args.training_seed, is_train=True)
            # from mindspore.parallel._cost_model_context import _set_rp_matmul_mem_coef
            # _set_rp_matmul_mem_coef(0.1)  # 全自动并行模式下，matmul内存占用系数
            self._set_optimizer_parallel_context()
            self._set_pipeline_parallel_context()
            self._set_dump_local_norm_parallel_context()
            self._reset_grad_accumulation_steps()
            self._set_data_parallel_size()
            self._set_dataset_strategy_parallel_context(mock_dataset_strategy=True)
            dataset_batch_size, train_global_batch_size = self._compute_train_batch_size_accord_gbs()
            self._set_train_global_batch_size(train_global_batch_size)
            self._set_train_dataset_batch_size(dataset_batch_size)
            self._set_model_config_for_muon_optimizer()
            if self.config.use_parallel:
                self._check_auto_parallel_mode_valid_for_training()
            logger.info("Training configuration setup completed successfully.")
        else:
            self._set_runner_seed(seed=self.config.infer_seed, is_train=False)
            self._check_parallel_mode_valid_for_predict()
            self._set_data_parallel_size()
            logger.info("Inference configuration setup completed successfully.")

        # set output directory
        set_output_path(self.config.output_dir)

        self._logging_host_info()
        mode_str = 'training' if is_train else 'inference'
        logger.info(
            f"Configuration setup completed for llm task in {mode_str} mode.")

    def _set_pipeline_parallel_context(self) -> None:
        """Set pipeline parallel context based on distributed parallel configuration.

        This method configures the pipeline parallel settings for the model training process.
        It sets the pipeline stages and pipeline configuration parameters such as interleave
        and scheduler type when in auto parallel mode.

        The pipeline configuration includes:
        - pipeline_model_parallel_size: Number of pipeline stages
        - pipeline_interleave: Whether to enable pipeline interleave
        - pipeline_scheduler: Scheduler type, default is "1f1b"
        """
        # New process uses distribute_parallel_config to set PP-related parallel configuration
        distribute_parallel_config = self.config.distribute_parallel_config
        if self._check_auto_parallel_mode_valid() and distribute_parallel_config:
            pipeline_stages = distribute_parallel_config.get("pipeline_model_parallel_size", 1)
            ms.set_auto_parallel_context(pipeline_stages=pipeline_stages)
            pipeline_parallel_config = self.config.distribute_parallel_config.pipeline_parallel_config
            if pipeline_parallel_config:
                pipeline_interleave = pipeline_parallel_config.get("pipeline_interleave", False)
                pipeline_scheduler = pipeline_parallel_config.get("pipeline_scheduler", "1f1b")
                ms.set_auto_parallel_context(
                    pipeline_config={
                        "pipeline_interleave": pipeline_interleave,
                        "pipeline_scheduler": pipeline_scheduler
                    })
            logger.info(f"Pipeline parallel context configured: pipeline_model_parallel_size={pipeline_stages}, "
                        f"pipeline_config={pipeline_parallel_config}")

    def _set_optimizer_parallel_context(self) -> None:
        """Set optimizer parallel context based on distributed parallel configuration.

        This method configures the optimizer parallel settings for model training.
        It enables parallel optimizer and sets optimizer level and weight shard size
        when the distribute parallel configuration is provided and parallel optimizer is enabled.

        The optimizer parallel configuration includes:
        - enable_parallel_optimizer: Whether to enable parallel optimizer
        - optimizer_level: Optimizer level, default is "level1"
        - optimizer_weight_shard_size: Weight shard size, default is -1
        """
        # New process uses distribute_parallel_config to set optimizer-related parallel configuration
        distribute_parallel_config = self.config.distribute_parallel_config
        if distribute_parallel_config:
            optimizer_parallel_config = distribute_parallel_config.optimizer_parallel_config
            enable_parallel_optimizer = optimizer_parallel_config.get("enable_parallel_optimizer", False)
            if enable_parallel_optimizer:
                optimizer_level = optimizer_parallel_config.get("optimizer_level", "level1")
                optimizer_weight_shard_size = optimizer_parallel_config.get("optimizer_weight_shard_size", -1)
                parallel_optimizer_threshold = optimizer_parallel_config.get("parallel_optimizer_threshold", 64)
                ms.set_auto_parallel_context(
                    enable_parallel_optimizer=enable_parallel_optimizer,
                    parallel_optimizer_config={
                        "optimizer_level": optimizer_level,
                        "optimizer_weight_shard_size": optimizer_weight_shard_size,
                        "parallel_optimizer_threshold": parallel_optimizer_threshold})
                logger.info(f"Optimizer parallel context configured: optimizer_level={optimizer_level}, "
                            f"optimizer_weight_shard_size={optimizer_weight_shard_size}")

    def _set_dataset_strategy_parallel_context(self, mock_dataset_strategy: bool = False) -> None:
        """Set dataset strategy parallel context based on distributed configuration.

        This method configures the dataset strategy for parallel processing. When mock_dataset_strategy
        is enabled, it generates a mock strategy for data parallel size acquisition. Otherwise,
        it automatically generates dataset strategy based on the actual data shapes and data
        parallel size in auto parallel mode.

        Args:
            mock_dataset_strategy (bool): Whether to generate a mock dataset strategy.
                                         Defaults to False.

        The dataset strategy determines how the dataset is distributed across devices in
        parallel training scenarios.
        """
        full_batch = ms.get_auto_parallel_context("full_batch")
        ds_stra = ms.get_auto_parallel_context("dataset_strategy")
        dp = self.config.distribute_parallel_config.data_parallel_size

        if mock_dataset_strategy:
            ms.set_auto_parallel_context(dataset_strategy=((dp, 1),))
            logger.warning("Generated mock dataset_strategy: [dp, 1], which will not take effect actually, "
                           "only used for dp acquisition during dataset creation")
            return

        if self._check_auto_parallel_mode_valid() and not full_batch:
            if isinstance(ds_stra, (list, tuple)):
                logger.warning("Manual setting is not recommended. In auto parallel mode with full_batch=False, "
                               "dataset_strategy will be automatically generated")
            # Get actual data shapes from the dataset
            output_shapes = self.train_dataset.output_shapes()
            # Generate fixed dataset_strategy based on dp split value
            dataset_strategy = [(dp,) + (1,) * (len(shape) - 1) for shape in output_shapes]
            ms.set_auto_parallel_context(dataset_strategy=tuple(dataset_strategy))
            logger.info(f"Current dataset_strategy is {dataset_strategy}")

    def _set_dump_local_norm_parallel_context(self) -> None:
        """Parse config.monitor_config and supplement settings for dumping local norm information.

        This method configures the parallel context for dumping local norm and device local norm
        based on the monitor configuration. If dump_path is not specified, it defaults to './dump'.
        The method sets up three types of norm dumping:
        1. dump_local_norm_path: Path for dumping local norm data
        2. dump_local_norm: Enable/disable local norm dumping
        3. dump_device_local_norm: Enable/disable device local norm dumping
        """
        if self.config.get('monitor_config'):
            monitor_config = self.config.monitor_config
            if not monitor_config.dump_path:
                monitor_config.dump_path = './dump'
                logger.info("`monitor_config.dump_path` is unset or set to empty, use default path './dump' instead.")
            ms.set_auto_parallel_context(dump_local_norm_path=monitor_config.dump_path)
            if monitor_config.local_norm:
                ms.set_auto_parallel_context(dump_local_norm=True)
            if monitor_config.device_local_norm:
                ms.set_auto_parallel_context(dump_device_local_norm=True)

    def _set_model_config_adapter_old_format(self) -> None:
        """Adapter method to convert old model config format to new format.

        This method transforms the configuration structure from the old format where
        `model_config` was a direct attribute of the config object to the new format
        where `model_config` is nested under the `model` attribute. This ensures
        backward compatibility with older configuration files.

        The transformation process:
        1. Creates a new `MindFormerConfig` object for the `model` attribute
        2. Moves the existing `model_config` to `self.config.model.model_config`
        3. Removes the old `model_config` attribute from the root config level
        """
        self.config.model = MindFormerConfig()
        self.config.model.model_config = self.config.model_config
        del self.config.model_config

    def _set_and_logging_training_step(self) -> None:
        """Check runner config and set training step parameters.

        This method calculates and configures the training steps based on the dataset size,
        epochs, and sink mode settings. It adjusts the number of epochs when sink mode
        is enabled and sink_size is specified. It also sets initial epoch and step values
        for training resumption.

        The method performs the following operations:
        1. Gets the training dataset size
        2. Sets original epochs value
        3. Initializes gradient accumulation steps if not set
        4. Sets initial epoch and step to 0 if not specified
        5. Adjusts epochs calculation based on sink mode and sink size
        6. Updates configuration with dataset size and training parameters
        """
        data_size = self._get_train_dataset_size()
        new_epochs = self.config.training_args.epochs
        self.config.training_args.origin_epochs = new_epochs

        if self.config.training_args.gradient_accumulation_steps is None:
            self.config.training_args.gradient_accumulation_steps = 1

        if self.config.training_args.initial_epoch is None:
            self.config.training_args.initial_epoch = 0

        if self.config.training_args.initial_step is None:
            self.config.training_args.initial_step = 0

        if self.config.training_args.sink_mode:
            if self.config.training_args.sink_size != -1:
                if self.config.training_args.sink_size <= 0:
                    raise ValueError("Per epoch size must be greater than 0 or equal to -1, "
                                     f"but got {self.config.training_args.sink_size}")
                if data_size < self.config.training_args.sink_size:
                    logger.warning("The data size %s (obtained from dataset.get_dataset_size()) is smaller "
                                   "than the sink_size %s (from config.training_args.sink_size), "
                                   "you should set config.training_args.sink_size to %s",
                                   data_size, self.config.training_args.sink_size, data_size)
                self.config.training_args.epochs = int((data_size / self.config.training_args.sink_size) * new_epochs)
            else:
                self.config.training_args.sink_size = data_size
        else:
            logger.warning("Sink mode is disabled, per epoch size is invalid and will be reset to -1.")
            self.config.training_args.sink_size = -1

        self.config.data_size = data_size
        logger.info("Configured training: epochs=%d, sink_size=%d",
                    self.config.training_args.origin_epochs, self.config.training_args.sink_size)
        logger.info("Training dataset preparation completed, dataset size: %d", data_size)

    def _set_train_dataset_batch_size(self, dataset_batch_size: int) -> None:
        """Set the batch size for training dataset.

        This method updates both the configuration and instance variable for dataset batch size.
        The batch size determines how many samples are processed together in each training step.

        Args:
            dataset_batch_size (int): The batch size to be set for training dataset processing.

        The method performs the following operations:
        1. Updates the dataset_batch_size in training arguments configuration
        2. Sets the instance variable for dataset batch size
        3. Logs the updated batch size information
        """
        self.config.training_args.dataset_batch_size = dataset_batch_size
        self.dataset_batch_size = dataset_batch_size
        logger.info(f"Training dataset batch size has been set to {dataset_batch_size}")

    def _set_train_global_batch_size(self, global_batch_size: int) -> None:
        """Set the global batch size for training.

        This method updates both the configuration and instance variable for global batch size.
        The global batch size represents the total batch size across all devices in distributed training,
        which is used to maintain consistent gradient updates regardless of the number of devices.

        Args:
            global_batch_size (int): The global batch size to be set for training process.

        The method performs the following operations:
        1. Updates the global_batch_size in training arguments configuration
        2. Sets the instance variable for global batch size
        3. Logs the updated global batch size information
        """
        self.config.training_args.global_batch_size = global_batch_size
        self.global_batch_size = global_batch_size
        logger.info(f"Training global batch size has been set to {global_batch_size}")

    def _set_llm_model(self, llm_model: Optional[nn.Cell] = None) -> None:
        """Set the LLM model instance for training.

        This method assigns the provided LLM model to the trainer instance.
        The model is required for subsequent training operations.

        Args:
            llm_model (Optional[nn.Cell]): The LLM model instance to be set.

        Raises:
            AttributeError: If llm_model is None.
        """
        if llm_model is None:
            raise AttributeError("llm_model is None, please provide a valid model instance")
        self.llm_model = llm_model

    def _set_train_dataset(self, train_dataset: Optional[GeneratorDataset] = None) -> None:
        """Set the training dataset instance for the trainer.

        This method assigns the provided training dataset to the trainer instance.
        The dataset is required for subsequent training operations.

        Args:
            train_dataset (Optional[GeneratorDataset]): The training dataset instance to be set.

        Raises:
            AttributeError: If train_dataset is None.
        """
        if train_dataset is None:
            raise AttributeError("train_dataset is None, please provide a valid dataset instance")
        self.train_dataset = train_dataset

    def _set_data_parallel_size(self) -> None:
        """Set the data parallel size in the distributed parallel configuration.

        This method computes the appropriate data parallel size based on the available
        devices and other parallel configuration settings, then updates the configuration.
        """
        self.config.distribute_parallel_config.data_parallel_size = self._compute_data_parallel_size()

    def _compute_data_parallel_size(self) -> int:
        """Compute the data parallel size based on distributed configuration.

        This method calculates the appropriate data parallel size based on the available
        devices and other parallel configuration settings. It ensures that the parallel
        configuration is valid and compatible with the device setup.

        Returns:
            int: The computed data parallel size.

        Raises:
            ValueError: If the parallel configuration is invalid or incompatible.
        """
        if not self._check_auto_parallel_mode_valid():
            return 1

        if self.config.distribute_parallel_config.data_parallel_size:
            return self.config.distribute_parallel_config.data_parallel_size

        device_num = self._get_device_num()

        tp = self.config.distribute_parallel_config.get("tensor_model_parallel_size", 1)
        pp = self.config.distribute_parallel_config.get("pipeline_model_parallel_size", 1)
        cp = self.config.distribute_parallel_config.get("context_parallel_size", 1)
        ep = self.config.distribute_parallel_config.get("expert_parallel_size", 1)

        parallel_product = tp * pp * cp

        if device_num % parallel_product != 0:
            raise ValueError(
                f"tensor_model_parallel_size*pipeline_model_parallel_size*context_parallel_size={parallel_product},"
                f"device_num % parallel_product = {device_num} % {parallel_product} != 0, "
                f"please set an appropriate parallel strategy")

        dp = device_num // parallel_product

        if self.config.run_mode == "predict" and self.config.predict_batch_size == 1:
            dp = 1

        if ep > 1 and (ep > dp * tp * cp or (dp * tp * cp) % ep != 0):
            raise ValueError(f"expert_parallel_size={ep} must be less than or equal to "
                             f"data_parallel_size*tensor_model_parallel_size*context_parallel_size={dp*tp*cp} or "
                             f"{dp*tp*cp} must be divisible by expert_parallel_size={ep} "
                             f"current expert_parallel_size={ep}")

        return dp

    def _compute_train_batch_size_accord_gbs(self) -> tuple:
        """Compute training batch size according to Global Batch Size (GBS).

        This method calculates the appropriate dataset batch size and global batch size based on
        user-specified GBS and micro_batch_size. It handles different modes including:
        1. Semi-auto/automatic parallel mode with various configurations
        2. Data parallel/standalone mode

        The method validates basic constraints and computes gradient accumulation steps or
        micro batch numbers based on the parallel configuration.

        Returns:
            tuple: A tuple containing (train_data_batch_size, global_batch_size)

        Raises:
            ValueError: If batch sizes are invalid or incompatible with parallel configuration.
        """
        # Validate basic constraints
        global_batch_size = self.config.training_args.global_batch_size
        micro_batch_size = self.config.training_args.micro_batch_size
        micro_batch_interleave_num = self.config.distribute_parallel_config.micro_batch_interleave_num
        data_parallel_size = self.config.distribute_parallel_config.data_parallel_size

        if global_batch_size <= 0 or micro_batch_size <= 0:
            raise ValueError("Batch sizes must be positive integers")

        # Original default values in config file may have been modified by user but won't take effect
        # They need to be calculated based on global_batch_size and micro_batch_size
        gradient_accumulation_steps = self.config.training_args.gradient_accumulation_steps

        if self._check_auto_parallel_mode_valid():  # Semi-auto/automatic parallel mode
            # Calculate gradient accumulation steps
            if global_batch_size % (data_parallel_size * micro_batch_size * micro_batch_interleave_num) != 0:
                raise ValueError(
                    f"Global batch size ({global_batch_size}) must be divisible by "
                    f"(data_parallel_size ({data_parallel_size}) * micro_batch_size ({micro_batch_size}) "
                    f"* micro_batch_interleave_num ({micro_batch_interleave_num}))"
                )

            # Regular gradient accumulation and pipeline parallel cannot be used simultaneously
            if gradient_accumulation_steps > 1 and self._get_pipeline_stages() > 1:
                raise ValueError("Gradient accumulation and pipeline parallel cannot be used simultaneously.")

            num_micro_batches = global_batch_size // (
                    data_parallel_size * micro_batch_size * micro_batch_interleave_num
            )

            grad_accu_condition = gradient_accumulation_steps > 1
            pp_condition = self._get_pipeline_stages() > 1

            if grad_accu_condition:  # User has set gradient accumulation logic
                self.config.training_args.gradient_accumulation_steps = num_micro_batches
                logger.info("Detected user-defined gradient_accumulation_steps. "
                            "With GBS=%d and micro_batch_size=%d, computed gradient accumulation "
                            "value is %d, resetting this value.",
                            global_batch_size, micro_batch_size, num_micro_batches)

            if pp_condition:  # User has set gradient accumulation logic in pipeline
                if num_micro_batches < self._get_pipeline_stages():
                    pp_size = self._get_pipeline_stages()
                    min_gbs = pp_size * data_parallel_size * micro_batch_size * micro_batch_interleave_num
                    raise ValueError(
                        f"MindSpore currently only supports pipeline-parallel's micro_batch_num "
                        f"must be greater than or equal to pipeline_model_parallel_size: {pp_size}, "
                        f"current micro_batch_num = global_batch_size // "
                        f"(data_parallel_size * micro_batch_size * micro_batch_interleave_num) = "
                        f"{global_batch_size} // {data_parallel_size} * {micro_batch_size} * "
                        f"{micro_batch_interleave_num} = {num_micro_batches}, "
                        f"please try to reduce pipeline_model_parallel_size <= {num_micro_batches} or "
                        f"increase global_batch_size >= {min_gbs}."
                    )
                self.config.distribute_parallel_config.micro_batch_num = num_micro_batches
                logger.info("Detected user-defined micro_batch_num. "
                            "With GBS=%d and micro_batch_size=%d, computed micro_batch_num value "
                            "is %d, resetting this value.",
                            global_batch_size, micro_batch_size, num_micro_batches)
            train_data_batch_size = global_batch_size // data_parallel_size
        else:
            # Each card reads data individually, aggregated to GBS size,
            # regular gradient accumulation scenarios available
            train_data_batch_size = global_batch_size // self._get_device_num()
            self._reset_distribute_parallel_config()

        return train_data_batch_size, global_batch_size

    def _reset_distribute_parallel_config(self) -> None:
        """Reset the distributed parallel configuration to default values.

        This method resets the `distribute_parallel_config` to its default configuration
        and explicitly sets the `data_parallel_size` to 1. This is typically used when
        falling back to a standalone or data parallel mode where complex parallel strategies
        are not needed or supported.

        The method performs the following operations:
        1. Updates the distribute_parallel_config with default values from TrainingParallelConfig
        2. Sets the data_parallel_size to 1
        3. Logs the configuration change for debugging purposes
        """
        self.config.distribute_parallel_config.update(TrainingParallelConfig().default_value())
        self.config.distribute_parallel_config.data_parallel_size = 1
        logger.info("Distributed parallel configuration has been reset to default config: %s.",
                    self.config.distribute_parallel_config)

    def _reset_grad_accumulation_steps(self) -> None:
        """Check and reset the gradient accumulation steps based on parallel configuration.

        This method validates the gradient accumulation steps configuration and adjusts it
        according to the parallel mode and pipeline stages. It ensures that gradient accumulation
        is properly configured for the current training setup.

        The method performs the following operations:
        1. Sets default gradient accumulation steps to 1 if not configured
        2. Validates that gradient accumulation steps is a positive integer
        3. Resets gradient accumulation steps to 1 when using pipeline parallelism
        4. Resets gradient accumulation steps to 1 when not in semi-auto parallel mode
        5. Disables Lazy Inline feature when gradient accumulation is not used with pipeline parallel
        6. Logs warnings and configuration changes for debugging purposes

        Raises:
            ValueError: If gradient_accumulation_steps is not a valid positive integer.
        """
        if self.config.training_args.gradient_accumulation_steps is None:
            self.config.training_args.gradient_accumulation_steps = 1

        if not isinstance(self.config.training_args.gradient_accumulation_steps, int) or \
                isinstance(self.config.training_args.gradient_accumulation_steps, bool):
            raise ValueError("gradient_accumulation_steps should be integer but got "
                             f"{type(self.config.training_args.gradient_accumulation_steps)}")

        if not self.config.training_args.gradient_accumulation_steps >= 1:
            raise ValueError("gradient_accumulation_steps should be greater or equal than 1, "
                             f"but got {self.config.training_args.gradient_accumulation_steps}")

        parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        pp = self._get_pipeline_stages()

        if self._check_auto_parallel_mode_valid() and pp > 1 \
                and self.config.training_args.gradient_accumulation_steps > 1:
            logger.warning("gradient_accumulation_steps will not take effect when using pipeline parallel, "
                           "reset the gradient_accumulation_steps from %s to 1.",
                           self.config.training_args.gradient_accumulation_steps)
            self.config.training_args.gradient_accumulation_steps = 1

        # Gradient accumulation not supported in data parallel/standalone mode for now
        if self.config.training_args.gradient_accumulation_steps > 1 and not self._check_auto_parallel_mode_valid():
            logger.warning("gradient_accumulation_steps currently need to be used in semi_auto_parallel mode, "
                           "but got %s mode, please check your runner config and parallel config. "
                           "Reset the gradient_accumulation_steps from %s to 1.",
                           parallel_mode, self.config.training_args.gradient_accumulation_steps)
            self.config.training_args.gradient_accumulation_steps = 1

        # Disable Lazy Inline feature only under this condition
        if self.config.training_args.gradient_accumulation_steps == 1 and pp == 1:
            logger.warning("The Lazy Inline compilation acceleration feature "
                           "only works with gradient_accumulation_steps > 1 "
                           "when not in pipeline parallel mode (pipeline_stage = 1). "
                           "Current pipeline stage=1 and gradient_accumulation_steps=1, "
                           "the feature is disabled by default.")
            self.config.model.model_config.disable_lazy_inline = True

    def _set_construct_args_key(self, column_names: List[str] = None) -> None:
        if self.config.train_dataset.construct_args_key is None and column_names is not None:
            self.config.train_dataset.construct_args_key = column_names
            logger.info("The config of train_dataset.construct_args_key has been set to %s.",
                        self.config.train_dataset.construct_args_key)

    def _set_model_config_for_muon_optimizer(self) -> None:
        # Enable max attention logits for Muon optimizer
        if self.config.optimizer.type == "Muon":
            self.config.model.model_config.monitor_max_attention_logit = True
            logger.info(
                "Currently identified Muon optimizer usage,"
                "the monitor_max_attention_logit of TransformerConfig has been set to True.")

    def _set_and_get_load_checkpoint_config(self) -> Optional[str]:
        """Set and get the load checkpoint configuration path.

        This method processes the checkpoint loading configuration by:
        1. Resolving the pretrained model directory path
        2. Determining the appropriate checkpoint path based on run mode
        3. Validating and resolving the final checkpoint path
        4. Updating the configuration with the resolved checkpoint path

        Returns:
            Optional[str]: The resolved checkpoint path, or None if no checkpoint is configured.

        The method handles both training and prediction modes, ensuring the correct
        checkpoint configuration is used for each scenario.
        """
        pretrained_model_dir = self._get_load_checkpoint(self.config.pretrained_model_dir)
        if self.config.run_mode != "predict":
            load_checkpoint_path_or_file = self.config.checkpoint_config.load_checkpoint
        else:
            load_checkpoint_path_or_file = self.config.load_checkpoint

        not_load_checkpoint_from_pretrained_model_dir = load_checkpoint_path_or_file == NOT_LOAD_ANY_CHECKPOINT_MODE
        if not_load_checkpoint_from_pretrained_model_dir:
            logger.info(f"Now load_checkpoint={NOT_LOAD_ANY_CHECKPOINT_MODE},"
                        "not loading any checkpoint from pretrained model directory as requested.")

        load_checkpoint_path_or_file = self._get_load_checkpoint(load_checkpoint_path_or_file)

        load_checkpoint_path_or_file = load_checkpoint_path_or_file \
            if (load_checkpoint_path_or_file is not None or not_load_checkpoint_from_pretrained_model_dir) \
            else pretrained_model_dir
        load_checkpoint_path_or_file = get_checkpoint_path(load_checkpoint_path_or_file)

        if self.config.run_mode != "predict":
            self.config.checkpoint_config.load_checkpoint = load_checkpoint_path_or_file
        else:
            self.config.load_checkpoint = load_checkpoint_path_or_file

        if load_checkpoint_path_or_file:
            logger.info(f'load_checkpoint config has been set to weight path: '
                        f'{load_checkpoint_path_or_file} for run_mode: {self.config.run_mode}')
        else:
            logger.info(f'load_checkpoint config is None,  which is invalid for run_mode: {self.config.run_mode}')
        return load_checkpoint_path_or_file

    def _set_learning_rate_scheduler(
            self,
            lr_scheduler: Optional[nn.learning_rate_schedule.LearningRateSchedule] = None,
            grouped_lr_scheduler: List[dict] = None) -> None:
        """Set learning rate scheduler(s) for model training.

        This method configures the learning rate scheduling strategy for the optimizer.
        It supports two modes:
        1. Standard mode: A single learning rate scheduler applied to all model parameters
        2. Grouped mode: Different learning rate schedulers for different parameter groups

        The grouped learning rate scheduler allows fine-grained control over learning rates
        for different parts of the model (e.g., different layers, embeddings, or attention
        mechanisms), which is useful for transfer learning, fine-tuning, or when different
        components require different learning rate schedules.

        Args:
            lr_scheduler (Optional[nn.learning_rate_schedule.LearningRateSchedule]):
                A single learning rate scheduler instance to be applied uniformly to all
                model parameters. This is used when all parameters should follow the same
                learning rate schedule. Examples include CosineAnnealingLR, PolynomialDecayLR, etc.
                If None, the scheduler will not be set (useful when only grouped schedulers are used).

            grouped_lr_scheduler (List[dict], optional):
                A list of dictionaries, where each dictionary represents a parameter group
                with its own learning rate scheduler. Each dictionary should contain:
                - 'params': List[str] - Parameter name patterns to match (supports wildcards)
                - 'lr_scheduler': LearningRateSchedule - The scheduler instance for this group
                - 'lr_config': MindFormerConfig - Configuration used to create the scheduler

                This allows different parameter groups to have independent learning rate schedules.
                For example, embeddings might use a different schedule than transformer layers.
                If None, grouped learning rate scheduling will not be used.

        Note:
            - Both schedulers can be set simultaneously. When both are provided, the grouped
              scheduler takes precedence for matching parameters, while the standard scheduler
              applies to unmatched parameters.
            - This method is typically called during optimizer creation in
              `create_optimizer_scheduler` method.
            - The method will log informational messages when schedulers are successfully set.

        Side Effects:
            - Sets `self.lr_scheduler` if lr_scheduler is provided
            - Sets `self.grouped_lr_scheduler` if grouped_lr_scheduler is provided
            - Logs configuration information for debugging purposes
        """
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler
            logger.info("The default learning rate scheduler has been set.")

        if grouped_lr_scheduler is not None:
            self.grouped_lr_scheduler = grouped_lr_scheduler
            logger.info("The group learning rate scheduler has been set.")

    def _train_dataset_restore_from_checkpoint(self, dataset: GeneratorDataset,
                                               load_checkpoint_path_or_file: Optional[str]) -> None:
        """Restore training dataset state from checkpoint.

        This method handles the restoration of training state when resuming from a checkpoint:
        1. Loads resume context from common.json if resume_training is enabled
        2. Calculates scaled initial step and epoch based on batch size differences
        3. Updates training arguments with restored values
        4. Sets up dataset skipping if data_skip_steps is configured

        Args:
            dataset (GeneratorDataset): The training dataset to potentially skip steps on.
            load_checkpoint_path_or_file (Optional[str]): Path to the checkpoint directory or file.

        The method performs the following operations:
        - Loads common.json to retrieve training state information
        - Scales step and epoch numbers based on global batch size differences
        - Updates training arguments with initial step, epoch, and loss scale values
        - Skips dataset steps if data_skip_steps is configured and ignore_data_skip is False
        """
        if self.config.training_args.resume_training and load_checkpoint_path_or_file:
            logger.info("Starting to load resume context from common.json...")
            common_file = os.path.join(load_checkpoint_path_or_file, 'common.json')
            if not os.path.exists(common_file):
                raise FileNotFoundError(f"No common.json found in directory '{load_checkpoint_path_or_file}'.")
            common_info = CommonInfo.load_common(common_file)
            step_scale = common_info.global_batch_size / self.global_batch_size
            self.config.training_args.initial_step = int(common_info.step_num * step_scale)
            self.config.training_args.initial_epoch = int(common_info.epoch_num * step_scale)

            resume_dict = {
                "step_num": self.config.training_args.initial_step,
                "epoch_num": self.config.training_args.initial_epoch,
                "loss_scale": 1
            }
            if self.config.training_args.scale_sense is not None:
                if hasattr(self.config.training_args.scale_sense, "loss_scale_value"):
                    self.config.training_args.scale_sense.loss_scale_value = common_info.loss_scale
                    resume_dict["loss_scale"] = self.config.training_args.scale_sense.loss_scale_value
                else:
                    self.config.training_args.scale_sense.scale_sense = common_info.loss_scale
                    resume_dict["loss_scale"] = self.config.training_args.scale_sense
            self.append_restore_info = [resume_dict]
            self.common_restore_info = common_info
        else:
            self.config.training_args.initial_epoch = 0
            self.config.training_args.initial_step = 0

        # Check if dataset steps should be skipped
        if self.config.training_args.data_skip_steps or self.config.training_args.resume_training:
            if not self.config.training_args.ignore_data_skip:
                if self.config.training_args.data_skip_steps:
                    data_skip_steps = self.config.training_args.data_skip_steps
                else:
                    data_skip_steps = self.config.training_args.initial_step
                if data_skip_steps > 0:
                    dataset.set_init_step(data_skip_steps)
                    logger.info(f"Dataset will skip {data_skip_steps} steps.")
            else:
                ignore_flag = self.config.training_args.ignore_data_skip
                logger.info(
                    f"ignore_data_skip is set to {ignore_flag}, dataset skipping is ignored.")

    def _load_or_resume_checkpoint_for_train(self,
                                             load_checkpoint_path_or_file: Optional[str],
                                             training_model_wrapper: object, network: nn.Cell,
                                             train_dataset: GeneratorDataset, optimizer: nn.Optimizer) -> None:
        """Load or resume checkpoint for training.

        This method handles loading checkpoints for training, with special handling for:
        1. Parallel training scenarios
        2. Resume training from a previous checkpoint
        3. Balanced load of checkpoint weights

        Args:
            load_checkpoint_path_or_file (Optional[str]): Path to the checkpoint file or directory.
            training_model_wrapper (object): The wrapped model API for training.
            network (nn.Cell): The neural network model.
            train_dataset (GeneratorDataset): The training dataset.
            optimizer (nn.Optimizer): The optimizer used for training.

        The method performs the following operations:
        - Enables online strategy saving for parallel training
        - Compiles the model for parallel execution
        - Resumes training from checkpoint with global step scaling if batch size changed
        - Loads checkpoint weights with balanced load option
        """
        if load_checkpoint_path_or_file and not check_is_reboot_node():
            if self.config.use_parallel:
                enable_save_strategy_online()
                compile_model(training_model_wrapper, train_dataset, mode=0, sink_mode=True,
                              epoch=self.config.training_args.epochs, sink_size=1)

            if self.config.training_args.resume_training:
                logger.info("Starting to resume training from checkpoint...")
                global_step = self.common_restore_info.global_step
                if self.common_restore_info.global_batch_size != self.global_batch_size:
                    global_step = int(self.common_restore_info.global_step *
                                      (self.common_restore_info.global_batch_size / self.global_batch_size))
                old_gbs = self.common_restore_info.global_batch_size
                logger.info(f"Scaled global step: {self.common_restore_info.global_step} → {global_step} "
                            f"(batch size changed from {old_gbs} to {self.global_batch_size})")
                load_checkpoint(
                    checkpoint=load_checkpoint_path_or_file, network=network, optimizer=optimizer,
                    global_step=global_step,
                    balanced_load=self.config.checkpoint_config.balanced_load
                )
            else:
                logger.info(f"Loading checkpoint from {load_checkpoint_path_or_file}...")
                load_checkpoint(checkpoint=load_checkpoint_path_or_file,
                                network=network,
                                balanced_load=self.config.checkpoint_config.balanced_load)

    def _wrap_network_with_tool_cells(self, network: nn.Cell) -> nn.Cell:
        """Wrap the network with tool cells for training process.

        This method wraps the network with various tool cells based on the training configuration:
        1. Micro-batch interleaving for double copy parallel feature
        2. Gradient accumulation cell for gradient accumulation training
        3. Pipeline cell for pipeline parallel training
        4. Virtual dataset cell for auto parallel training

        Args:
            network (nn.Cell): The base network to be wrapped.

        Returns:
            nn.Cell: The wrapped network with appropriate tool cells.

        The method performs the following operations:
        - Applies MicroBatchInterleaved wrapper when micro_batch_interleave_num > 1
        - Applies GradAccumulationCellWithMultiOutputs when gradient_accumulation_steps > 1
          and not using pipeline parallel
        - Applies PipelineCellWithMultiOutputs when pipeline stages > 1
        - Applies _VirtualDatasetCell for auto parallel mode in graph mode
        """
        micro_batch_interleave_num = self.config.distribute_parallel_config.micro_batch_interleave_num
        gradient_accumulation_steps = self.config.training_args.gradient_accumulation_steps
        pp = self._get_pipeline_stages()

        if micro_batch_interleave_num > 1:
            logger.info("micro_batch_interleave_num > 1, enabling the double copy parallel feature.")
            network = MicroBatchInterleaved(network, micro_batch_interleave_num)

        if gradient_accumulation_steps > 1 and not pp > 1:
            logger.info("gradient_accumulation_steps > 1, wrapping network with GradAccumulationCell. "
                        "It is recommended to use the `Lazy Inline` feature to reduce compilation time.")
            network = GradAccumulationCellWithMultiOutputs(network, gradient_accumulation_steps)

        if pp > 1:
            micro_batch_num = self.config.distribute_parallel_config.micro_batch_num
            seq_split_num = self.config.distribute_parallel_config.pipeline_parallel_config.seq_split_num
            if seq_split_num > 1:
                if self.config.recompute_config.recompute:
                    raise ValueError("When using sequential pipeline, full recompute cannot be applied.")
                if self.config.training_args.calculate_per_token_loss:
                    raise ValueError("When using sequential pipeline, per-token loss calculation cannot be applied.")
            logger.info(f"Pipeline parallel enabled with {pp} stages, wrapping network with PipelineCell.")
            network = PipelineCellWithMultiOutputs(network, micro_size=micro_batch_num)

        if self._check_auto_parallel_mode_valid() and ms.get_context('mode') == 0:
            logger.info("Auto parallel mode enabled in graph mode, wrapping network with _VirtualDatasetCell.")
            network = _VirtualDatasetCell(network)
            ds_broadcast_level = ms.context.get_context("dataset_broadcast_opt_level")
            if ds_broadcast_level > 0:
                # pylint: disable=W0212
                network._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")

        return network

    def _init_parameters_data(self, network: nn.Cell, optimizer: Optional[nn.Optimizer] = None) -> None:
        """Initialize network and optimizer parameters data.

        This method initializes the parameters data for both network and optimizer when needed:
        1. Initializes network parameters if network_delay_inited flag is set
        2. Initializes optimizer parameters if optimizer_delay_inited flag is set and optimizer is provided

        Args:
            network (nn.Cell): The neural network model whose parameters need to be initialized.
            optimizer (Optional[nn.Optimizer]): The optimizer whose parameters need to be initialized.
                Defaults to None.

        The method performs the following operations:
        - Calls network.init_parameters_data() when network_delay_inited is True
        - Calls optimizer.init_parameters_data() when optimizer_delay_inited is True and optimizer is not None
        """
        if self.network_delay_inited:
            logger.info("Initializing network parameters data with delay initialization...")
            network.init_parameters_data()

        if self.optimizer_delay_inited and optimizer is not None:
            logger.info("Initializing optimizer parameters data with delay initialization...")
            optimizer.init_parameters_data()

    def _create_training_log_callback(self) -> MFLossMonitor:
        """Create training log callback for monitoring basic training information.

        This method creates a `MFLossMonitor` callback that logs essential training metrics
        including learning rate, loss, and training progress information.

        Returns:
            MFLossMonitor: Configured callback for logging training basics.

        The callback includes information about:
        - Learning rate scheduling
        - Training progress (epochs, steps)
        - Batch size configurations (global, micro-batch)
        - Model-specific information (MoE, MTP)
        - Loss calculation settings
        """
        calculate_per_token_loss = getattr(self.config.training_args, 'calculate_per_token_loss', False)
        step_interval = getattr(self.config.monitor_config, "step_interval", 1)

        mf_training_info_callback = MFLossMonitor(
            learning_rate=self.config.optimizer.learning_rate,
            per_print_times=step_interval,
            micro_batch_num=self.config.distribute_parallel_config.micro_batch_num,
            micro_batch_interleave_num=self.config.distribute_parallel_config.micro_batch_interleave_num,
            origin_epochs=self.config.training_args.origin_epochs,
            dataset_size=self.config.data_size,
            initial_epoch=self.config.training_args.initial_epoch,
            initial_step=self.config.training_args.initial_step,
            global_batch_size=self.global_batch_size,
            gradient_accumulation_steps=self.config.training_args.gradient_accumulation_steps,
            calculate_per_token_loss=calculate_per_token_loss,
            print_separate_loss=self.config.training_args.get("print_separate_loss", True),
            is_moe_model=self.llm_model.is_moe_model(),
            is_mtp_model=self.llm_model.is_mtp_model()
        )
        logger.info("Created training log callback for basic training information monitoring.")
        return mf_training_info_callback

    def _create_save_checkpoint_callback(self) -> CheckpointMonitor:
        """Create checkpoint saving callback for model training.

        This method creates a `CheckpointMonitor` callback that handles:
        1. Periodic saving of model checkpoints during training
        2. Configuration of checkpoint saving parameters
        3. Health monitoring of saved checkpoints
        4. Asynchronous saving control based on fault tolerance settings

        Returns:
            CheckpointMonitor: Configured callback for checkpoint saving.

        The callback supports various checkpoint saving configurations including:
        - Save frequency (steps or seconds)
        - Checkpoint format and storage options
        - Health monitoring with embedding norm thresholds
        - Asynchronous saving with fault tolerance compatibility
        """
        use_checkpoint_health_monitor = getattr(self.config.training_args, "use_checkpoint_health_monitor", False)
        embedding_local_norm_threshold = getattr(self.config.training_args, "embedding_local_norm_threshold", 1.0)
        embedding_size = self._get_embedding_size() if use_checkpoint_health_monitor else None

        checkpoint_format = self.config.checkpoint_config.get('checkpoint_format', 'safetensors')
        if checkpoint_format == 'ckpt':
            raise ValueError("Checkpoint format 'ckpt' is not supported. Please use 'safetensors' instead.")

        async_save = self.config.checkpoint_config.get('async_save', False)
        if check_tft_valid() or check_tre_valid() or check_tsp_valid():
            async_save = False
            logger.warning("In TrainFaultTolerance mode, asynchronous checkpoint saving must be disabled. "
                           "Setting async_save to False.")

        save_ckpt_callback = CheckpointMonitor(
            use_legacy_format=True,  # After fixing the bug in checkpoint 2.0, set it to False.
            prefix=self.config.checkpoint_config.get('prefix', 'llm'),
            directory=self.config.checkpoint_config.get('directory', self.config.output_dir),
            save_checkpoint_steps=self.config.checkpoint_config.get('save_checkpoint_steps', 1),
            save_checkpoint_seconds=self.config.checkpoint_config.get('save_checkpoint_seconds', 0),
            keep_checkpoint_max=self.config.checkpoint_config.get('keep_checkpoint_max', 5),
            keep_checkpoint_per_n_minutes=self.config.checkpoint_config.get('keep_checkpoint_per_n_minutes', 0),
            integrated_save=self.config.checkpoint_config.get('integrated_save', False),
            save_network_params=self.config.checkpoint_config.get('save_network_params', False),
            save_trainable_params=self.config.checkpoint_config.get('save_trainable_params', False),
            async_save=async_save,
            append_info=self.append_restore_info,
            exception_save=self.config.checkpoint_config.get('exception_save', False),
            global_batch_size=self.global_batch_size,
            checkpoint_format=checkpoint_format,
            remove_redundancy=self.config.checkpoint_config.get('remove_redundancy', False),
            embedding_size=embedding_size,
            embedding_local_norm_threshold=embedding_local_norm_threshold,
            use_checkpoint_health_monitor=use_checkpoint_health_monitor,
            health_ckpts_record_dir=self.config.checkpoint_config.get('health_ckpts_record_dir',
                                                                      self.config.output_dir),
        )

        logger.info("Created checkpoint saving callback with configuration: "
                    f"format={checkpoint_format}, async_save={async_save}, "
                    f"save_steps={self.config.checkpoint_config.get('save_checkpoint_steps', 1)}")
        return save_ckpt_callback

    def create_train_dataset(self) -> GeneratorDataset:
        """Create training dataset for LLM training.

        This method creates and configures a training dataset based on the configuration settings.
        It handles different data loader types and applies appropriate dataset processing.

        Returns:
            GeneratorDataset: Configured training dataset ready for model training.

        Raises:
            ValueError: If dataset broadcast optimization level is incompatible with sink mode,
                       or if broadcast data configuration is invalid for certain data loaders.

        The method performs the following operations:
        - Validates dataset broadcast optimization level compatibility
        - Creates LLMDataset instance with provided configuration
        - Configures dataset sharding information
        - Creates data loader with specified column names
        - Builds final dataset with batching and processing options
        - Applies special processing for BlendedMegatronDatasetDataLoader if needed
        """
        llm_dataset = LLMDataset(dataset_config=self.config.train_dataset)
        dataset_seed = self.config.training_args.dataset_seed or self.config.training_args.training_seed or 1234

        llm_dataset.set_ms_dataset_config(
            dataset_seed=dataset_seed,
            prefetch_size=self.config.train_dataset.get("prefetch_size", 1),
            numa_enable=self.config.train_dataset.get("numa_enable", False)
        )

        shard_id, num_shards = llm_dataset.generate_shard_info(
            data_parallel_size=self.config.distribute_parallel_config.get("data_parallel_size", 1)
        )

        create_compressed_eod_mask = llm_dataset.is_create_compressed_eod_mask()
        create_attention_mask = llm_dataset.is_create_attention_mask()
        input_columns = llm_dataset.get_default_input_columns(create_attention_mask, create_compressed_eod_mask)
        self._set_construct_args_key(input_columns)

        if create_compressed_eod_mask:
            self.config.model.model_config.use_eod_attn_mask_compression = True

        data_loader = llm_dataset.create_data_loader(
            column_names=input_columns,
            shard_id=shard_id,
            num_shards=num_shards
        )
        micro_batch_num = self.config.distribute_parallel_config.get("micro_batch_num",1) \
            if self._get_pipeline_stages() > 1 \
            else self.config.training_args.get("gradient_accumulation_steps", 1)
        train_dataset = llm_dataset.create_dataset(
            data_loader=data_loader,
            data_batch_size=self.config.training_args.dataset_batch_size,
            drop_remainder=self.config.train_dataset.get("drop_remainder", True),
            input_columns=input_columns,
            output_columns=self.config.train_dataset.get("output_columns", input_columns),
            micro_batch_num=micro_batch_num,
            eod_reset=self.config.train_dataset.get("eod_reset", False),
            eod_token_id=self.config.train_dataset.get("eod_token_id", None),
            dynamic_batch=False,
            pad_token_id=self.config.train_dataset.get("pad_token_id", None),
            num_parallel_workers=self.config.train_dataset.get("num_parallel_workers", 8),
            use_llm_token_profile=self.config.profile.get("use_llm_token_profile", False),
            profile_llm_token_config=self.config.profile.get("token_profile_config", {})
        )

        # Postprocess for BlendedMegatronDatasetDataLoader
        if llm_dataset.get_data_loader_type() == "BlendedMegatronDatasetDataLoader":
            logger.info("Processing BlendedMegatronDatasetDataLoader with specialized handling")
            data_loader_config = self.config.train_dataset.get("data_loader")
            megatron_dataset_sizes = data_loader_config.get("sizes")
            # Reset dataset size to remove redundant data and align with global batch size
            train_dataset = train_dataset.take(int(megatron_dataset_sizes[0]) // self.global_batch_size)
            dataset_size = train_dataset.get_dataset_size()
            logger.info(f"Using BlendedMegatronDatasetDataLoader, reset dataset size to {dataset_size}.")

        logger.info("Training dataset created successfully")
        return train_dataset

    def create_model(self) -> nn.Cell:
        """Create neural network model based on configuration.

        This method creates a neural network model by building it from the configuration settings.
        It supports both training and prediction modes with different parameter initialization strategies.

        Returns:
            nn.Cell: The constructed neural network model ready for training or inference.

        The method performs the following operations:
        - Determines if network parameters should be delay initialized based on checkpoint configuration
        - Prepares model arguments based on run mode (train or predict)
        - Constructs the model using the `build_network` function
        - Applies optional pipeline stage checking for parallel training
        - Uses no_init_parameters context when delay initialization is enabled
        """

        def create_network(network_kwargs: dict = None) -> nn.Cell:
            """Internal function to create network from configuration.

            Args:
                network_kwargs (dict, optional): Additional keyword arguments for model construction.

            Returns:
                nn.Cell: Built network model.
            """
            logger.info("Building network from configuration...")

            if self.config.get("pretrained_model_dir", None):
                self.config.model.pretrained_model_dir = self.config.pretrained_model_dir

            if self.config.get("generation_config", None):
                self.config.model.generation_config = self.config.generation_config

            network = build_network(self.config.model, default_args=network_kwargs)

            if hasattr(network, "check_pipeline_stage") and callable(network.check_pipeline_stage):
                network.check_pipeline_stage()  # Check pipeline stage configuration

            return network

        # Check if network parameters should be delay initialized
        self.network_delay_inited = self._check_load_checkpoint_valid()

        # Prepare model arguments based on run mode
        if self.config.run_mode != "predict":
            virtual_pipeline_model_parallel_size = getattr(
                self.config.distribute_parallel_config.pipeline_parallel_config,
                "virtual_pipeline_model_parallel_size", 1
            )
            pp_offset = getattr(
                self.config.distribute_parallel_config.pipeline_parallel_config,
                "pipeline_stage_offset", 0
            )

            calculate_per_token_loss = getattr(
                self.config.training_args, "calculate_per_token_loss", False
            )

            model_kwargs = {
                "parallel_config": self.config.distribute_parallel_config,
                "virtual_pipeline_model_parallel_size": virtual_pipeline_model_parallel_size,
                "offset": pp_offset,
                "calculate_per_token_loss": calculate_per_token_loss,
            }
            model_kwargs.update(**self.config.recompute_config)
            model_kwargs.update(**self.config.swap_config)
        else:
            model_kwargs = {
                "parallel_config": self.config.distribute_parallel_config
            }

        # Create model with or without parameter initialization
        if self.network_delay_inited:
            with no_init_parameters():
                model = create_network(network_kwargs=model_kwargs)
            logger.info("Model created with delay parameter initialization")
        else:
            model = create_network(network_kwargs=model_kwargs)
            logger.info("Model created with immediate parameter initialization")

        return model

    def create_optimizer_scheduler(self, network: nn.Cell) -> nn.Optimizer:
        """Create optimizer and learning rate scheduler for model training.

        This method constructs an optimizer with grouped parameters and learning rate schedule
        based on the configuration settings. It supports delayed parameter initialization
        when checkpoint loading is configured.

        Args:
            network (nn.Cell): The neural network model to optimize.

        Returns:
            nn.Optimizer: Configured optimizer with learning rate scheduler.

        Raises:
            NotImplementedError: If PmaAdamW optimizer is used but model doesn't implement get_model_parameters.
            ValueError: If learning_rate is not set in optimizer config when lr_schedule is None.

        The method performs the following operations:
        - Creates learning rate schedule using create_lr_scheduler
        - Handles special parameter grouping for PmaAdamW optimizer
        - Groups model parameters with appropriate weight decay
        - Builds optimizer using build_optim function
        - Applies delayed parameter initialization when configured
        """
        logger.info("Building optimizer and learning rate scheduler from configuration...")

        # Build learning rate schedule
        is_grouped_lr_scheduler = self.config.grouped_lr_schedule is not None
        grouped_lr_scheduler = None
        if is_grouped_lr_scheduler:
            lr_scheduler, grouped_lr_scheduler = self.create_grouped_lr_scheduler()
        else:
            lr_scheduler = self.create_lr_scheduler()
        self._set_learning_rate_scheduler(lr_scheduler, grouped_lr_scheduler)

        optimizer_type = self.config.optimizer.type

        # Handle special parameter grouping for PmaAdamW optimizer
        model_params = set()
        if optimizer_type in ("PmaAdamW", "FusedPmaAdamW", "Muon"):
            if hasattr(self.llm_model, "get_model_parameters"):
                model_params.update(self.llm_model.get_model_parameters())
            else:
                raise NotImplementedError(f"The {type(network)} has not implemented the interface: "
                                          f"get_model_parameters.")

        # Group parameters with weight decay
        weight_decay = self.config.optimizer.weight_decay if self.config.optimizer.weight_decay else 0.
        group_params = get_optimizer_grouped_parameters(
            network, weight_decay,
            optimizer_type=optimizer_type,
            grouped_lr_schedule=grouped_lr_scheduler,
            model_params=model_params)

        def create_optimizer() -> nn.Optimizer:
            """Internal function to create optimizer instance.

            Returns:
                nn.Optimizer: Built optimizer instance.

            Raises:
                ValueError: If learning_rate is not set in optimizer config.
            """
            optimizer_kwargs = {"params": group_params}
            if lr_scheduler is not None:
                optimizer_kwargs["learning_rate"] = lr_scheduler
            else:
                if self.config.optimizer.learning_rate is None:
                    raise ValueError("lr_schedule is None, please set learning_rate in optimizer config.")

            if optimizer_type == "Muon":
                optimizer_kwargs.update(**self._get_muon_optimizer_kwargs())

            optimizer = build_optim(self.config.optimizer, default_args=optimizer_kwargs)

            return optimizer

        # Check if optimizer parameters should be delay initialized
        self.optimizer_delay_inited = self._check_load_checkpoint_valid()

        # Create optimizer with or without parameter initialization
        if self.optimizer_delay_inited:
            with no_init_parameters():
                optimizer = create_optimizer()
            logger.info("Optimizer created with delay parameter initialization")
        else:
            optimizer = create_optimizer()
            logger.info("Optimizer created with immediate parameter initialization")

        logger.info("Optimizer and learning rate scheduler created successfully")
        return optimizer

    def create_lr_scheduler(
            self, lr_schedule_config: MindFormerConfig = None) -> nn.learning_rate_schedule.LearningRateSchedule:
        """Create learning rate scheduler based on configuration.

        This method builds a learning rate scheduler by processing the configuration settings
        and calculating the appropriate parameters for the learning rate schedule. It handles
        warmup configuration, total steps calculation, and scheduler instantiation.

        Args:
            lr_schedule_config (MindFormerConfig, optional):
                Learning rate scheduler configuration. If None, uses `self.config.lr_schedule`.
                The configuration should contain:
                - `type`: str - Scheduler type (e.g., "CosineWithWarmUpLR", "PolynomialWithWarmUpLR",
                  "ConstantWarmUpLR", "CosineAnnealingLR", etc.)
                - `learning_rate`: float - Base learning rate value
                - `warmup_ratio`: float, optional - Warmup ratio (required if warmup_epochs is set)
                - `warmup_steps`: int, optional - Number of warmup steps (can be set directly)
                - `warmup_lr_init`: float, optional - Initial learning rate during warmup
                - `total_steps`: int, optional - Total training steps (-1 means auto-calculate)
                - Other scheduler-specific parameters (e.g., `min_lr`, `max_lr`, `decay_steps`, etc.)

        Returns:
            nn.learning_rate_schedule.LearningRateSchedule:
                Built learning rate scheduler instance. The scheduler will dynamically adjust
                the learning rate during training based on the current step. Returns None if
                no valid configuration is provided (both lr_schedule_config and self.config.lr_schedule
                are None or empty).

        Note:
            - Total steps calculation:
              - If `sink_mode` is False: total_steps = epochs * train_dataset_size
              - If `sink_mode` is True: total_steps = epochs * sink_size
            - If `total_steps` is None or -1 in config, it will be auto-calculated
            - `warmup_epochs` is converted to `warmup_steps` using: warmup_steps = warmup_epochs * train_dataset_size
            - The method modifies the input config by popping `warmup_epochs` (if present)
            - If `warmup_lr_init` is not set and the scheduler supports it, a default value will be used
            - Supported scheduler types include: CosineWithWarmUpLR, PolynomialWithWarmUpLR,
              ConstantWarmUpLR, CosineAnnealingLR, CosineWithRestartsAndWarmUpLR, etc.

        Example:
            Configuration example:
            ```
            lr_schedule:
              type: "CosineWithWarmUpLR"
              learning_rate: 1e-4
              warmup_ratio: 0.1
              total_steps: -1  # Auto-calculate
            ```
        """
        logger.info("Building learning rate scheduler from configuration...")
        train_dataset_size = self._get_train_dataset_size()
        warmup_lr_init = None
        lr_schedule_config = self.config.lr_schedule if lr_schedule_config is None else lr_schedule_config

        if lr_schedule_config:
            warmup_lr_init = lr_schedule_config.get("warmup_lr_init", 0.)
            warmup_steps = lr_schedule_config.get("warmup_steps", 0)
            warmup_ratio = lr_schedule_config.get("warmup_ratio", 0.)
            lr_schedule_config.warmup_steps = warmup_steps
            lr_schedule_config.warmup_ratio = warmup_ratio

            # Calculate total training steps
            if not self.config.training_args.sink_mode:
                total_steps = int(self.config.training_args.epochs * train_dataset_size)
            else:
                total_steps = int(self.config.training_args.epochs * self.config.training_args.sink_size)

            # Set total_steps in `lr_schedule` if not explicitly defined
            if lr_schedule_config.total_steps is None or lr_schedule_config.total_steps == -1:
                lr_schedule_config.total_steps = total_steps
            else:
                lr_schedule_config.total_steps = int(lr_schedule_config.total_steps)

        # Build learning rate scheduler
        lr_schedule = build_lr(lr_schedule_config)

        # Apply default warmup_lr_init if not set
        if lr_schedule and hasattr(lr_schedule, "warmup_lr_init") and warmup_lr_init is None:
            logger.info(f"warmup_lr_init is not set. Using default value {lr_schedule.warmup_lr_init}.")

        logger.info("Learning rate scheduler created successfully")
        return lr_schedule

    def create_grouped_lr_scheduler(self) -> tuple[nn.learning_rate_schedule.LearningRateSchedule, list[dict]]:
        """Create grouped learning rate schedulers from configuration.

        This method builds a set of learning rate schedulers for different parameter groups,
        allowing fine-grained control over learning rates for different parts of the model.
        It creates a default scheduler for unmatched parameters and multiple group-specific
        schedulers based on parameter name patterns.

        The configuration structure (`self.config.grouped_lr_schedule`) should contain:
        - `default`: Configuration for the default learning rate scheduler (applied to all
          parameters that don't match any group patterns)
        - `grouped`: A list of dictionaries, each containing:
          - `params`: List[str] - Parameter name patterns to match (supports wildcards via fnmatch)
          - Other LR scheduler configuration keys (e.g., `type`, `learning_rate`, `warmup_steps`, etc.)

        This is particularly useful for:
        - Transfer learning: Different learning rates for pretrained vs. new layers
        - Fine-tuning: Lower learning rates for embeddings, higher for task-specific layers
        - Layer-wise decay: Gradually decreasing learning rates from top to bottom layers
        - Component-specific schedules: Different schedules for attention, MLP, embeddings, etc.

        Returns:
            tuple[nn.learning_rate_schedule.LearningRateSchedule, list[dict]]:
                A tuple containing:
                - Default learning rate scheduler: Applied to parameters that don't match
                  any group patterns in the grouped configuration
                - Grouped learning rate scheduler list: A list of dictionaries, each containing:
                  - 'params': List[str] - Parameter name patterns for this group
                  - 'lr_scheduler': LearningRateSchedule - The scheduler instance for this group
                  - 'lr_config': MindFormerConfig - The configuration used to create this scheduler

        Raises:
            ValueError: If any group configuration in `grouped` is missing the 'params' field
                       or if 'params' is empty.

        Note:
            - Parameter matching is done by name patterns, supporting wildcards (e.g., "*.embedding*")
            - The default scheduler is always created and will be used for unmatched parameters
            - Each group can have its own independent learning rate schedule configuration
            - The method modifies the input configuration dictionaries by popping 'params'

        Example:
            Configuration structure:
            ```
            grouped_lr_schedule:
              default:
                type: "ConstantWarmUpLR"
                learning_rate: 1.e-4
                warmup_ratio: 0.
                total_steps: -1
              grouped:
                - params: ["embedding*"]
                  type: "CosineWithWarmUpLR"
                  learning_rate: 1.e-5
                  warmup_ratio: 0.
                  total_steps: -1
                - params: ["*.self_attention*"]
                  type: "PolynomialWithWarmUpLR"
                  learning_rate: 2.e-4
                  warmup_ratio: 0.
                  total_steps: -1
            ```
        """
        logger.info("Building grouped learning rate scheduler from configuration...")
        default_lr_schedule_config = self.config.grouped_lr_schedule.default
        default_lr_scheduler = self.create_lr_scheduler(default_lr_schedule_config)

        grouped_lr_scheduler = []
        grouped_config = self.config.grouped_lr_schedule.grouped

        # Iterate over each grouped LR configuration
        for lr_config in grouped_config:
            params = lr_config.pop('params', None)
            if not params or not isinstance(params, list):
                raise ValueError(
                    "Got invalid 'params' in grouped_lr_schedule.grouped: each item must include "
                    "a non-empty 'params' list."
                )

            lr_config = MindFormerConfig(**lr_config)
            lr_scheduler = self.create_lr_scheduler(lr_config)
            grouped_lr_scheduler.append({
                'params': params,
                'lr_scheduler': lr_scheduler,
                'lr_config': lr_config
            })

        return default_lr_scheduler, grouped_lr_scheduler

    def create_model_wrapper(self, network: nn.Cell, optimizer: nn.Optimizer) -> Union[
        MFPipelineWithLossScaleCell, MFTrainOneStepCell]:
        """Create model wrapper with training tools and configurations.

        This method wraps the network with various training tools and creates an appropriate
        model wrapper based on the training configuration. It supports both pipeline parallel
        and regular training scenarios.

        Args:
            network (nn.Cell): The neural network model to be wrapped.
            optimizer (nn.Optimizer): The optimizer used for training.

        Returns:
            Union[MFPipelineWithLossScaleCell, MFTrainOneStepCell]: Configured model wrapper
            for training process.

        The method performs the following operations:
        - Wraps network with tool cells using _wrap_network_with_tool_cells
        - Extracts training configuration parameters
        - Determines appropriate model wrapper type based on training configuration
        - Configures wrapper with gradient clipping, loss scaling, and monitoring options
        """
        network_wrapper = self._wrap_network_with_tool_cells(network)

        logger.info("Building model wrapper for training from configuration...")
        calculate_per_token_loss = getattr(self.config.training_args, "calculate_per_token_loss", False)
        use_skip_data_by_global_norm = getattr(self.config.training_args, "use_skip_data_by_global_norm", False)
        global_norm_spike_threshold = getattr(self.config.training_args, "global_norm_spike_threshold", 1.0)

        # These two parameters are mutually exclusive, only one takes effect (>1) or neither (=1)
        gradient_accumulation_steps = self.config.training_args.get("gradient_accumulation_steps", 1)
        micro_batch_num = self.config.distribute_parallel_config.get("micro_batch_num", 1)

        if self._check_runner_wrapper_for_pipeline_parallel() or self._check_runner_wrapper_for_grad_accu():
            model_wrapper = MFPipelineWithLossScaleCell(
                network=network_wrapper,
                optimizer=optimizer,
                micro_batch_num=gradient_accumulation_steps if gradient_accumulation_steps > 1 else micro_batch_num,
                use_clip_grad=self.config.training_args.get("use_clip_grad", True),
                max_grad_norm=self.config.training_args.get("max_grad_norm", 1.0),
                scale_sense=self.config.training_args.get("scale_sense", 1.0),
                local_norm=False,
                # Only used to support printing embedding_local_norm capability, this independent configuration
                # is not externally provided, provided by TrainingStatusMonitor local_norm capability
                calculate_per_token_loss=calculate_per_token_loss,
                use_skip_data_by_global_norm=use_skip_data_by_global_norm,
                global_norm_spike_threshold=global_norm_spike_threshold,
                print_separate_loss=self.config.training_args.get("print_separate_loss", True),
                lr_scheduler=self.lr_scheduler,
                grouped_lr_scheduler=self.grouped_lr_scheduler,
            )
            logger.info("Created MFPipelineWithLossScaleCell model wrapper for pipeline parallel training")
        else:
            model_wrapper = MFTrainOneStepCell(
                network=network_wrapper,
                optimizer=optimizer,
                use_clip_grad=self.config.training_args.get("use_clip_grad", False),
                max_grad_norm=self.config.training_args.get("max_grad_norm", 1.0),
                scale_sense=self.config.training_args.get("scale_sense", 1.0),
                local_norm=False,
                # Only used to support printing embedding_local_norm capability, this independent configuration
                # is not externally provided, provided by TrainingStateMonitor local_norm capability
                calculate_per_token_loss=calculate_per_token_loss,
                use_skip_data_by_global_norm=use_skip_data_by_global_norm,
                global_norm_spike_threshold=global_norm_spike_threshold,
                print_separate_loss=self.config.training_args.get("print_separate_loss", True),
                lr_scheduler=self.lr_scheduler,
                grouped_lr_scheduler=self.grouped_lr_scheduler,
            )
            logger.info("Created MFTrainOneStepCell model wrapper for standard training")

        logger.info("Model wrapper created successfully")
        return model_wrapper

    def create_callback_list(self) -> list:
        """Create and configure callback list for training monitoring and control.

        This method builds a comprehensive list of callbacks that provide various functionalities
        during the training process, including logging, checkpointing, monitoring, profiling,
        and fault tolerance. The callbacks are created based on the configuration settings.

        Returns:
            list: A sorted list of configured callback instances for training.

        The method performs the following operations:
        - Creates training log callback for basic training information
        - Adds training status monitor callback for runtime monitoring
        - Includes top-k bias balance callback for MoE models
        - Sets up checkpoint saving callback
        - Processes user-defined callbacks from configuration
        - Adds fault tolerance callback when enabled
        - Incorporates profiling callback when profiling is enabled
        - Appends stop monitor callback for training termination control
        - Sorts callbacks in proper execution order
        """
        logger.info("Building callbacks for training...")

        # Create and add training log callback
        self.callbacks.append(self._create_training_log_callback())

        # Create and add checkpoint saving callback
        save_ckpt_callback = self._create_save_checkpoint_callback()
        self.callbacks.append(save_ckpt_callback)

        if self.config.optimizer.type == "Muon":
            self.callbacks.append(MaxLogitsMonitor())
            logger.info("Created max logits callback for Muon optimizer.")

        transformer_config = self._get_llm_transformer_config()
        if transformer_config.moe_router_enable_expert_bias:
            self.callbacks.append(
                TopkBiasBalanceCallback(
                    balance_via_topk_bias=transformer_config.moe_router_enable_expert_bias,
                    topk_bias_update_rate=transformer_config.moe_router_bias_update_rate,
                    expert_num=transformer_config.num_moe_experts,
                    micro_batch_num=self.config.distribute_parallel_config.get("micro_batch_num", 1),
                    gradient_accumulation_steps=self.config.training_args.get("gradient_accumulation_steps", 1))
            )
            logger.info("moe_router_enable_expert_bias is True, created top-k bias balance callback for MoE models.")

        # Add stop monitor callback for training termination
        if isinstance(self.config.training_args.stop_step, int) and self.config.training_args.stop_step > 0:
            self.callbacks.append(TrainCallBack(stop_step=self.config.training_args.stop_step))

        # Sort callbacks in proper execution order
        callbacks = self._sort_callbacks(self.callbacks)
        self.config.callbacks = callbacks

        logger.info("Callbacks created and configured successfully")
        return callbacks

    def create_tokenizer(self) -> AutoTokenizer:
        """Create tokenizer from pretrained model directory.

        This method creates a tokenizer instance by loading it from a pretrained model directory.
        It supports loading tokenizers with remote code execution when explicitly trusted.

        Returns:
            AutoTokenizer: Configured tokenizer instance loaded from pretrained model directory.

        Raises:
            ValueError: If the provided pretrained_model_dir path is not a valid directory.

        The method performs the following operations:
        - Imports AutoTokenizer from transformers library
        - Retrieves pretrained model directory and trust settings from configuration
        - Validates that the pretrained model directory is a valid path
        - Creates tokenizer instance using from_pretrained method
        """

        logger.info("Creating tokenizer from pretrained model directory...")
        pretrained_model_dir = self.config.get("pretrained_model_dir", "")
        trust_remote_code = self.config.get("trust_remote_code", False)

        pretrained_model_dir = os.path.realpath(pretrained_model_dir)
        if not os.path.isdir(pretrained_model_dir):
            raise ValueError(f"The current interface supports passing a local folder path, "
                             f"but the provided path '{pretrained_model_dir}' is not a valid directory.")

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_dir,
            trust_remote_code=trust_remote_code)

        logger.info("Tokenizer created successfully from %s", pretrained_model_dir)
        return tokenizer

    def train(self, config: MindFormerConfig = None) -> None:
        """Execute the complete training process for LLM model.

        This method orchestrates the entire training workflow including dataset preparation,
        model creation, optimizer setup, checkpoint loading, and training execution.

        Args:
            config (MindFormerConfig, optional): Configuration object containing training parameters.
                If not provided, uses the instance's existing config.

        The method performs the following operations:
        - Sets up training configuration and parallel context
        - Creates and configures training dataset
        - Handles checkpoint loading and resumption logic
        - Builds neural network model and optimizer
        - Initializes training callbacks and monitoring tools
        - Executes the training process
        - Cleans up resources after training completion
        """
        logger.info("Starting create LLM model training process...")

        # Setup training configuration and parallel context
        self._setup_config(config)

        # Create and configure training dataset
        train_dataset = self.create_train_dataset()
        self._set_train_dataset(train_dataset=train_dataset)
        self._set_dataset_strategy_parallel_context()

        # Handle checkpoint loading configuration
        load_checkpoint_path_or_file = self._set_and_get_load_checkpoint_config()

        # Restore training state from checkpoint if resuming
        self._train_dataset_restore_from_checkpoint(train_dataset, load_checkpoint_path_or_file)

        # Configure training steps and epochs
        self._set_and_logging_training_step()

        # Record configuration to global environment
        self._record_config_to_global_envs()

        # Build neural network model
        network = self.create_model()
        self._set_llm_model(llm_model=network)

        # Apply data order wrapper if specified
        construct_args_key = self.config.train_dataset.pop("construct_args_key", None)
        if construct_args_key is not None:
            network = DataOrderWrapperCell(construct_args_key, network)

        # Log model parameter count
        self._count_parameters(network, run_mode=self.config.run_mode)

        # Create optimizer and learning rate scheduler
        optimizer = self.create_optimizer_scheduler(network)

        # Wrap model with training tools
        model_forward_and_backward_wrapper = self.create_model_wrapper(network, optimizer)

        # Create training callbacks
        callbacks = self.create_callback_list()

        # Initialize training executor
        model_trainer = ms.Model(network=model_forward_and_backward_wrapper)

        # Convert HF checkpoint format if needed
        load_checkpoint_path_or_file = self._get_load_path_after_hf_convert(
            load_checkpoint_path_or_file, self.llm_model)

        # Load or resume checkpoint
        self._load_or_resume_checkpoint_for_train(
            load_checkpoint_path_or_file, model_trainer, network, train_dataset, optimizer)

        # Initialize model and optimizer parameters
        self._init_parameters_data(network, optimizer)

        # Print configuration for main ranks
        if get_rank() % get_local_rank_size() == 0 or get_rank() % get_local_rank_size() == get_local_rank_size() - 1:
            pprint(self.config)

        logger.info("Starting LLM Model Training Preparation")
        # Execute training
        model_trainer.train(
            self.config.training_args.epochs,
            train_dataset,
            callbacks=callbacks,
            dataset_sink_mode=True,
            sink_size=1,
            initial_epoch=self.config.training_args.initial_epoch)

        logger.info("LLM model training completed successfully")

    def predict(self, config: MindFormerConfig = None, **kwargs) -> None:
        """Execute the text generation prediction process.

        This method performs inference using the trained LLM model to generate text based on input data.
        It handles model loading, tokenizer creation, pipeline setup, and result generation.

        Args:
            config (MindFormerConfig, optional): Configuration object containing prediction parameters.
                If not provided, uses the instance's existing config.
            **kwargs: Additional keyword arguments passed to the text generation pipeline.

        The method performs the following operations:
        - Sets up prediction configuration and parallel context
        - Records configuration to global environment
        - Creates and initializes the neural network model
        - Converts and loads checkpoint weights if needed
        - Initializes model parameters
        - Creates tokenizer for text processing
        - Sets up text generation pipeline
        - Executes text generation on input data
        - Saves and logs the generated results
        """
        logger.info("Starting LLM model prediction process...")

        # Setup prediction configuration and parallel context
        self._setup_config(config, is_train=False)

        # Handle checkpoint loading configuration
        load_checkpoint_path_or_file = self._set_and_get_load_checkpoint_config()

        # Record configuration to global environment
        self._record_config_to_global_envs()

        # Create and initialize the neural network model
        network = self.create_model()
        self._count_parameters(network, run_mode=self.config.run_mode)

        # Convert and load checkpoint weights
        load_checkpoint_path_or_file = self._get_load_path_after_hf_convert(
            load_checkpoint_path_or_file, network) \
            if self.config.run_mode == "predict_with_train_model" else load_checkpoint_path_or_file
        network.load_weights(load_checkpoint_path_or_file)

        # Initialize model parameters
        self._init_parameters_data(network)

        # Create tokenizer for text processing
        tokenizer = self.create_tokenizer()

        # Setup text generation pipeline
        llm_pipeline = TextGenerationPipeline(
            class_name="text_generation",
            model=network,
            tokenizer=tokenizer,
            batch_size=self.config.predict_batch_size,
            adapter_id=self.config.adapter_id,
            **kwargs)

        # Print configuration for main ranks
        if get_rank() % get_local_rank_size() == 0 or get_rank() % get_local_rank_size() == get_local_rank_size() - 1:
            pprint(config)

        logger.info("Executing text generation on input data...")

        # Execute text generation on input data
        output_results = llm_pipeline(self.config.input_data)

        # Process and extract output information
        output_info = []
        for one_output in output_results:
            if isinstance(one_output, dict) and "info" in one_output:
                output_info.append(one_output["info"])
            else:
                output_info.append(one_output)

        # Save generated results to file
        flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        save_file = "text_generation_result.txt"
        with os.fdopen(os.open("text_generation_result.txt", flags_, FILE_PERMISSION), 'w') as file:
            file.write(str(output_info))

        # Log prediction results
        logger.info("Text generation completed. Output result: %s", str(output_info))
        logger.info("Output result saved to file: %s", save_file)
        logger.info("LLM model prediction process completed successfully")

    def _get_load_path_after_hf_convert(self, load_checkpoint_path_or_dir: str, network: nn.Cell) -> str:
        """Check if the checkpoint is in HF safetensors format and convert it if necessary.

        This method detects if the provided checkpoint is in HuggingFace safetensors format,
        and converts it to MindSpore safetensors format when needed. The conversion is
        performed when qkv_concat is enabled or when additional parameter support is not available.

        Args:
            load_checkpoint_path_or_dir (str): Path to the checkpoint file or directory.
            network (nn.Cell): The neural network model for which the checkpoint is being loaded.

        Returns:
            str: Path to the converted MindSpore safetensors checkpoint or the original path
                 if no conversion is needed.

        The method performs the following operations:
        - Checks if the checkpoint is in HF safetensors format
        - Converts HF safetensors to MS safetensors when required
        - Synchronizes across ranks in parallel training scenarios
        """
        if load_checkpoint_path_or_dir and is_hf_safetensors_dir(load_checkpoint_path_or_dir, network):
            logger.info("Checkpoint format is HF safetensors, converting to MS safetensors...")
            converted_sf_path = process_hf_checkpoint(network, self.config.output_dir, load_checkpoint_path_or_dir)
            # Wait for main rank to convert HF safetensors
            if self.config.use_parallel:
                barrier()
            logger.info("HF safetensors checkpoint converted successfully")
            return converted_sf_path
        return load_checkpoint_path_or_dir

    def _get_muon_optimizer_kwargs(self) -> dict:
        micro_batch_num = self.config.distribute_parallel_config.get("micro_batch_num", 1) \
            if self._get_pipeline_stages() > 1 \
            else self.config.training_args.get("gradient_accumulation_steps", 1)
        optimizer_kwargs = {
            "model": self.llm_model,
            "micro_batch_num": micro_batch_num
        }
        return optimizer_kwargs

    def _get_device_num(self) -> int:
        """Get the number of devices used for training or inference.

        This method returns the total number of devices (e.g., GPUs, NPUs) available
        for the current training or inference session. In parallel mode, it retrieves
        the group size, otherwise it defaults to 1 for single device execution.

        Returns:
            int: Number of devices in use. Returns 1 for standalone mode or the
                 group size in parallel mode.

        The method performs the following operations:
        - Checks if parallel mode is enabled
        - Retrieves group size when in parallel mode
        - Returns appropriate device count
        """
        device_num = 1
        if self.config.use_parallel:
            device_num = get_group_size()
        return device_num

    def _get_embedding_size(self) -> int:
        """Get the embedding size of the GPT model.

        This method retrieves the embedding dimension size from the LLM model's
        GPT transformer configuration. This value is typically used for monitoring
        and checkpoint health checks.

        Returns:
            int: The embedding size of the GPT model.
        """
        return self.llm_model.get_gpt_embedding_size()

    def _get_llm_transformer_config(self) -> TransformerConfig:
        """Get the GPT model's Transformer configuration information.

        This method retrieves the GPT Transformer configuration information from the LLM model instance.
        It is primarily used to obtain specific configuration parameters of the model, such as embedding dimension size.

        Returns:
            int: The GPT Transformer configuration information.
        """
        return self.llm_model.get_gpt_transformer_config()

    def _record_config_to_global_envs(self) -> None:
        """Record validated configuration to global environment.

        This method stores the validated and potentially reset configuration
        in the global configuration registry, making it accessible throughout
        the application.

        The method performs the following operations:
        - Records the current configuration dictionary to the global registry
        - Makes configuration accessible from anywhere in the application
        """
        MFGlobalConfig.record_config_dict(self.config)

    @staticmethod
    def _set_runner_seed(seed: int = None, is_train: bool = True) -> None:
        """Set random seed for reproducible training.

        This static method sets the random seed for both MindSpore and NumPy
        to ensure reproducible results across training runs. If no seed is
        provided, it defaults to a predefined value.

        Args:
            seed (int, optional): Random seed value to set. If None, defaults
                                  to 1314 for consistent initialization.
            is_train (bool): Whether the mode is training. Defaults to True.

        The method performs the following operations:
        - Sets MindSpore random seed using set_seed
        - Sets NumPy random seed for numpy operations
        - Uses default seed when none provided
        """
        if is_train and ms.context.get_auto_parallel_context("parallel_mode") \
                in ["semi_auto_parallel", "auto_parallel"]:
            logger.info("Not set seed for training, when parallel_mode is semi_auto_parallel or auto_parallel.")
            return

        if seed is not None:
            set_seed(seed)
            np.random.seed(seed)
        else:
            set_seed(1314)
            np.random.seed(1314)

    @staticmethod
    def _logging_host_info() -> None:
        """Log host information for debugging and tracking purposes.

        This static method retrieves and logs the hostname and IP address
        of the current machine. This information is useful for identifying
        the execution environment in distributed training scenarios.

        The method performs the following operations:
        - Executes hostname command to get machine name
        - Executes hostname -I command to get IP address
        - Logs the retrieved host information
        """
        host_name_output = subprocess.run(['hostname'], shell=False, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, encoding='utf-8', check=False)
        host_ip_output = subprocess.run(['hostname', '-I'], shell=False, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, encoding='utf-8', check=False)
        host_name = host_name_output.stdout.strip()
        host_ip = host_ip_output.stdout.strip().split(' ')[0]
        logger.info(f"Host information - Name: {host_name}, IP: {host_ip}")

    @staticmethod
    def _count_parameters(network: nn.Cell, run_mode: str = 'train', units: str = 'M') -> None:
        """Count and log the number of network parameters.

        This static method calculates and logs the total number of parameters
        in the provided neural network model, expressed in millions (M) for
        better readability.

        Args:
            network (nn.Cell): The neural network model whose parameters
                               are to be counted.
            run_mode (str): The running mode, options are 'train', 'finetune' and 'predict'.
                           Default is 'train'.
            units (str): The unit for returning the parameter count, options are 'M' and 'B'.
                         Default is 'M'.

        The method performs the following operations:
        - Counts total parameters and trainable parameters
        - Converts parameter count to specified units (M for millions, B for billions)
        - Logs parameter count with appropriate formatting based on run_mode
        - For train/finetune modes, logs both total and trainable parameters
        - For predict mode, logs only total parameters
        """
        trainable_params = [np.prod(param.shape) for param in network.trainable_params()]
        total_params = [np.prod(param.shape) for param in network.get_parameters()]
        if units == 'M':
            count_params_m = sum(total_params) / 1e6
            trainable_params_m = sum(trainable_params) / 1e6
            if run_mode in ['train', 'finetune']:
                logger.info(f"Network Parameters: {count_params_m:.0f} M.")
                logger.info(f"Network Trainable Parameters: {trainable_params_m:.0f} M.")
            else:
                logger.info(f"Network Parameters: {count_params_m:.0f} M.")
        elif units == 'B':
            count_params_m = sum(total_params) / 1e9
            trainable_params_m = sum(trainable_params) / 1e9
            if run_mode in ['train', 'finetune']:
                logger.info(f"Network Parameters: {count_params_m:.1f} B.")
                logger.info(f"Network Trainable Parameters: {trainable_params_m:.1f} B.")
            else:
                logger.info(f"Network Parameters: {count_params_m:.1f} B.")
        else:
            raise ValueError("Invalid units. Please use 'M' or 'B'.")

    @staticmethod
    def _get_pipeline_stages() -> int:
        """Get the number of pipeline stages for task trainer.

        This static method retrieves the pipeline stages configuration from
        the auto parallel context. This value determines how the model is
        split across pipeline stages for parallel execution.

        Returns:
            int: Number of pipeline stages configured for the task trainer.
        """
        pipeline_stages = ms.get_auto_parallel_context("pipeline_stages")
        return pipeline_stages

    @staticmethod
    def _sort_callbacks(default_callbacks: list) -> list:
        """Sort callbacks by execution order.

        This static method sorts the provided list of callbacks according to
        a predefined execution order defined in CALLBACK_HAS_SORT. Callbacks
        that match the predefined types are placed in their specified order,
        while others are appended to the end.

        Args:
            default_callbacks (list): List of callback instances to be sorted.

        Returns:
            list: Sorted list of callbacks according to execution priority.
        """
        callbacks = [None] * len(CALLBACK_HAS_SORT)

        for callback in default_callbacks:
            flag = False
            for index, callback_ in enumerate(CALLBACK_HAS_SORT):
                if isinstance(callback, callback_):
                    callbacks[index] = callback
                    flag = True
            if not flag:
                callbacks.append(callback)

        return list(filter(lambda x: x is not None, callbacks))

    @staticmethod
    def _get_load_checkpoint(checkpoint: str) -> Optional[str]:
        """Get a checkpoint path which will be loaded.

        This static method validates and returns the checkpoint path that
        will be used for loading model weights. It performs basic validation
        to ensure the checkpoint path is valid.

        Args:
            checkpoint (str): Checkpoint path to validate and return.

        Returns:
            Optional[str]: Validated checkpoint path, or None if input is falsy.

        Raises:
            TypeError: If checkpoint is not a string.
            ValueError: If checkpoint path does not exist.
        """
        if not checkpoint or checkpoint == NOT_LOAD_ANY_CHECKPOINT_MODE:
            return None

        if not isinstance(checkpoint, str):
            raise TypeError(f"checkpoint should be a str, but got {type(checkpoint)}")

        if os.path.exists(checkpoint):
            return checkpoint

        raise ValueError(f"{checkpoint} does not exist, please check load_checkpoint in yaml and set a correct value.")

    def _get_train_dataset_size(self) -> int:
        """Get the size of the training dataset.

        This static method retrieves the total number of samples in the
        training dataset, which is used for calculating training steps
        and epochs.

        Returns:
            int: Total number of samples in the training dataset.
        """
        if self.train_dataset is None:
            raise RuntimeError("Please set train_dataset in yaml.")
        return self.train_dataset.get_dataset_size()

    @staticmethod
    def _check_auto_parallel_mode_valid() -> bool:
        """Check if auto parallel mode is valid for training operations.

        This static method validates if the current parallel mode is one
        of the supported auto parallel modes (semi-auto or full auto) that
        can be used for training operations.

        Returns:
            bool: True if current parallel mode is valid for auto parallel
                  operations, False otherwise.
        """
        if (ms.get_auto_parallel_context("parallel_mode")
                in [ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL]):
            return True
        return False

    @staticmethod
    def _check_auto_parallel_mode_valid_for_training() -> None:
        """Check if auto parallel mode is valid for training.

        This static method validates that the current parallel mode is
        supported for training operations. It raises a RuntimeError if
        the mode is not supported.

        Raises:
            RuntimeError: If the current parallel mode is not supported
                          for training operations.
        """
        support_parallel_mode = [ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL,
                                 ParallelMode.DATA_PARALLEL]
        if ms.get_auto_parallel_context("parallel_mode") not in support_parallel_mode:
            raise RuntimeError(f"Training mode only supports {support_parallel_mode}.")

    @staticmethod
    def _check_parallel_mode_valid_for_predict() -> None:
        """Check if parallel mode is valid for prediction.

        This static method validates that the current parallel mode is
        supported for prediction/inference operations. Currently, only
        standalone mode is supported for prediction.

        Raises:
            RuntimeError: If the current parallel mode is not supported
                          for prediction operations.
        """
        support_parallel_mode = [ParallelMode.STAND_ALONE]
        if ms.get_auto_parallel_context("parallel_mode") not in support_parallel_mode:
            raise RuntimeError(f"Predict mode only supports {support_parallel_mode}.")

    def _check_load_checkpoint_valid(self) -> bool:
        """Check if checkpoint configuration is valid.

        This method validates the checkpoint loading configuration by
        checking if the specified checkpoint path exists and is accessible.
        It handles both boolean flags and string paths for checkpoint
        configuration.

        Returns:
            bool: True if checkpoint configuration is valid, False otherwise.

        Raises:
            ValueError: If checkpoint path is specified but does not exist.
        """
        load_checkpoint_path_or_file = self.config.checkpoint_config.load_checkpoint \
            if self.config.run_mode != "predict" else self.config.load_checkpoint
        if load_checkpoint_path_or_file and load_checkpoint_path_or_file != NOT_LOAD_ANY_CHECKPOINT_MODE:
            return True
        return False

    def _check_runner_wrapper_for_pipeline_parallel(self) -> bool:
        """Check if runner wrapper is needed for pipeline parallel.

        This method determines whether the runner wrapper should be used
        for pipeline parallel training by checking if auto parallel mode
        is valid and if pipeline model parallel size is greater than 1.

        Returns:
            bool: True if runner wrapper is needed for pipeline parallel,
                  False otherwise.
        """
        if (not self._check_auto_parallel_mode_valid() or
                self.config.distribute_parallel_config.pipeline_model_parallel_size <= 1):
            return False
        return True

    def _check_runner_wrapper_for_grad_accu(self) -> bool:
        """Check if runner wrapper is needed for gradient accumulation.

        This method determines whether the runner wrapper should be used
        for gradient accumulation training by checking if auto parallel
        mode is valid and if gradient accumulation steps are greater than 1.

        Returns:
            bool: True if runner wrapper is needed for gradient accumulation,
                  False otherwise.
        """
        gradient_accumulation_steps = self.config.training_args.gradient_accumulation_steps
        if not self._check_auto_parallel_mode_valid() or gradient_accumulation_steps <= 1:
            return False
        return True
