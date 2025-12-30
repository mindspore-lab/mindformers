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
"""Trainer for training models with MindFormers."""
import os
import enum
from typing import Union, Optional, Callable, List, Dict, Any

from mindspore.dataset import Dataset

# Import from mindformers
from mindformers.dataset import build_dataset
from mindformers.models import PreTrainedModel, build_network
from mindformers.checkpoint.checkpoint import load_checkpoint
from mindformers.trainer.training_args import TrainingArguments
from mindformers.tools.logger import logger
from mindformers.tools import MindFormerConfig
from mindformers.core import build_lr, build_optim, build_callback, build_metric
from mindformers.pet import get_pet_model
from mindformers.core.callback_pynative import CallbackHandler
from mindformers.trainer.optimizer_grouped_parameters import get_optimizer_grouped_parameters
from mindformers.trainer_pynative.train_state import TrainerState


class TrainMode(enum.Enum):
    """Training mode enumeration."""
    FINETUNE = "finetune"
    PRETRAIN = "pretrain"


class Trainer:
    """
    Trainer for training models in MindFormers.

    The Trainer class provides a unified interface for training models with support for:
    - Model training and evaluation
    - Checkpoint management
    - Callback system
    - Custom loss functions
    - Distributed training

    Args:
        model: Model instance or None. If None, will be built from config.
        config: Either a path to yaml config file or a TrainingArguments instance
        compute_loss_func: Optional custom loss function
        train_dataset: Training dataset instance or None
        eval_dataset: Evaluation dataset instance or None
        processing_class: Optional processor for data preprocessing
        optimizer: Optimizer instance or None
        lr_scheduler: Learning rate scheduler instance or None
        compute_metrics: Optional function to compute evaluation metrics
        callbacks: List of callback instances
    """

    def __init__(
        self,
        model: PreTrainedModel = None,
        config: Union[str, TrainingArguments] = None,
        compute_loss_func: Optional[Callable] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        processing_class: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        lr_scheduler: Optional[Any] = None,
        compute_metrics: Optional[Callable] = None,
        callbacks: Optional[List] = None
    ):
        """
        Initialize the Trainer.

        Args:
            model: Model instance
            config: YAML configuration file path (str) or TrainingArguments instance
            compute_loss_func: Custom loss function
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            processing_class: Data processor
            optimizer: Optimizer instance
            lr_scheduler: Learning rate scheduler
            compute_metrics: Metrics computation function
            callbacks: List of callbacks
        """
        # Initialize config
        self.config = self._init_config(config)

        # Verify instance validity when config is yaml file
        if isinstance(config, str):
            if any([model, train_dataset, eval_dataset, optimizer, lr_scheduler, callbacks]):
                logger.warning(
                    "When config is a yaml file, model/dataset/optimizer/lr_scheduler/callbacks "
                    "instances should not be provided. They will be built from config."
                )

        # Create model
        self.model = self._create_model(
            model,
            getattr(self.config, 'model', None)
        )

        # Create datasets
        self.train_dataset = self._create_dataset(
            train_dataset,
            getattr(self.config, 'train_dataset', None)
        )
        self.eval_dataset = self._create_dataset(
            eval_dataset,
            getattr(self.config, 'eval_dataset', None)
        )

        # Create optimizer and scheduler
        self.optimizer, self.lr_scheduler = self._create_optimizer_and_scheduler(
            optimizer,
            lr_scheduler,
            getattr(self.config, 'optimizer', None),
            getattr(self.config, 'lr_schedule', None)
        )

        # Create callback handler
        self.callback_handler = self._create_callback_handler(
            callbacks,
            self.config
        )

        # Create metrics
        self.compute_metrics = self._create_metrics(
            compute_metrics,
            getattr(self.config, 'metric', None)
        )

        # Store other parameters
        self.compute_loss_func = compute_loss_func
        self.processing_class = processing_class

        # Initialize training state
        self.state = None

    def _init_config(self, config: Union[str, TrainingArguments]) -> MindFormerConfig:
        """
        Initialize trainer config from yaml file or TrainingArguments instance.

        This method converts config inputs to MindFormerConfig:
        - yaml file path (str) -> MindFormerConfig
        - TrainingArguments -> MindFormerConfig

        Args:
            config: Either a yaml file path (str) or TrainingArguments instance

        Returns:
            MindFormerConfig instance

        Raises:
            ValueError: If config is None
            FileNotFoundError: If yaml file does not exist
        """
        if config is None:
            raise ValueError("config cannot be None. Please provide a yaml file path or TrainingArguments instance.")

        # If config is a string (yaml file path), load it as MindFormerConfig
        if isinstance(config, str):
            if not os.path.exists(config):
                raise FileNotFoundError(f"Config file not found: {config}")

            logger.info(f"Loading config from yaml file: {config}")
            return MindFormerConfig(config)

        # If config is TrainingArguments, convert to MindFormerConfig
        if isinstance(config, TrainingArguments):
            logger.info("Converting TrainingArguments to MindFormerConfig")
            # Convert TrainingArguments to dict first
            config_dict = {}
            for key in dir(config):
                if not key.startswith('_') and not callable(getattr(config, key)):
                    config_dict[key] = getattr(config, key)
            return MindFormerConfig(**config_dict)

        # Should not reach here due to type hints
        raise TypeError(f"config must be str or TrainingArguments, got {type(config)}")

    def _create_model(self, model, model_config: Optional[Dict]) -> Any:
        """
        Create or validate model instance.

        Args:
            model: User-provided model instance or None
            model_config: Model configuration from yaml

        Returns:
            Model instance
        """
        # If user provided model instance, use it directly
        if model is not None:
            logger.info("Using user-provided model instance.")
            return model

        # Build model from config
        if model_config is None:
            raise ValueError("Either model instance or model_config must be provided.")

        logger.info("Building model from config...")
        model = build_network(model_config)

        # Apply PET if provided (after base model is built)
        pet_config = getattr(self.config, 'pet_config', None)
        if pet_config is not None:
            logger.info("Applying PET configuration to model...")
            model = get_pet_model(model, pet_config)

        return model

    # pylint: disable=unused-argument
    def _wrapper_model(self, model, config: Dict) -> Any:
        """
        Wrap model for distributed training (HSDP).

        Args:
            model: Model to wrap
            config: Wrapper configuration (reserved for future use)

        Returns:
            Wrapped model
        """
        # Reserved interface: currently no wrapper logic required
        logger.info("Wrapper is a no-op. Returning model as-is.")
        return model

    def _create_dataset(
        self,
        dataset,
        dataset_config: Optional[Dict]
    ) -> Optional[Any]:
        """
        Create or validate dataset instance.

        Args:
            dataset: User-provided dataset instance or None
            dataset_config: Dataset configuration from yaml

        Returns:
            Dataset instance or None
        """
        # If user provided dataset instance, use it directly
        if dataset is not None:
            logger.info("Using user-provided dataset instance.")
            return dataset

        # If no config, return None
        if dataset_config is None:
            return None

        # Build dataset from config
        logger.info("Building dataset from config...")
        return build_dataset(dataset_config)

    def _create_optimizer_and_scheduler(
        self,
        optimizer,
        lr_scheduler,
        optimizer_config: Optional[Dict],
        lr_config: Optional[Dict]
    ) -> tuple:
        """
        Create optimizer and learning rate scheduler.

        Args:
            optimizer: User-provided optimizer instance or None
            lr_scheduler: User-provided LR scheduler instance or None
            optimizer_config: Optimizer configuration from yaml
            lr_config: LR scheduler configuration from yaml

        Returns:
            Tuple of (optimizer, lr_scheduler)
        """
        # If user provided instances, use them directly
        if optimizer is not None and lr_scheduler is not None:
            logger.info("Using user-provided optimizer and lr_scheduler instances.")
            return optimizer, lr_scheduler

        # Build from config
        if optimizer_config is None or lr_config is None:
            logger.warning("No optimizer or lr_scheduler config provided.")
            return None, None

        logger.info("Building optimizer and lr_scheduler from config...")

        # Build learning rate scheduler first
        lr = build_lr(lr_config)

        # Get grouped parameters using official utility
        weight_decay = getattr(optimizer_config, 'weight_decay', 0.0) if optimizer_config else 0.0
        grouped_params = get_optimizer_grouped_parameters(
            model=self.model,
            weight_decay=weight_decay,
            dynamic_lr_schedule=lr,
            layer_scale=False,
            layer_decay=1.0,
            optimizer_type=getattr(optimizer_config, 'type', 'AdamW') if optimizer_config else 'AdamW',
            model_params=None
        )

        # Build optimizer using default_args to inject params and lr
        default_args = {
            'params': grouped_params,
            'learning_rate': lr
        }
        optimizer = build_optim(optimizer_config, default_args=default_args)

        return optimizer, lr

    def _create_callback_handler(
        self,
        callbacks: Optional[List],
        config: Any
    ) -> CallbackHandler:
        """
        Create callback handler.

        Args:
            callbacks: User-provided callback list or None
            config: Configuration object

        Returns:
            CallbackHandler instance
        """
        # Prepare initial callback list
        callback_list: List = []
        if callbacks:
            callback_list.extend(callbacks)

        # Build callbacks from config and extend
        callback_config = getattr(config, 'callbacks', None)
        if callback_config is not None:
            cbs = build_callback(callback_config)
            if cbs:
                callback_list.extend(cbs)

        # Create handler with complete list
        cb_handler = CallbackHandler(
            callbacks=callback_list,
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler
        )

        return cb_handler

    def _create_metrics(
        self,
        compute_metrics: Optional[Callable],
        metric_config: Optional[Dict]
    ) -> Optional[Callable]:
        """
        Create or validate metrics function.

        Args:
            compute_metrics: User-provided metrics function or None
            metric_config: Metrics configuration from yaml

        Returns:
            Metrics function or None
        """
        # If user provided metrics function, use it directly
        if compute_metrics is not None:
            logger.info("Using user-provided compute_metrics function.")
            return compute_metrics

        # Build from config
        if metric_config is None:
            return None

        logger.info("Building metrics from config...")
        return build_metric(metric_config)

    def train(
        self,
        checkpoint_path: Optional[str] = None,
        mode: str = "pretrain",
        do_eval: bool = False
    ):
        """
        Execute the training loop.

        Args:
            checkpoint_path: Path to checkpoint file to load
            mode: Training mode, either "pretrain" or "finetune"
            do_eval: Whether to run evaluation

        Returns:
            Training output/results
        """
        # Validate mode
        if mode not in ["pretrain", "finetune"]:
            raise ValueError(f"mode must be 'pretrain' or 'finetune', got: {mode}")

        # Check checkpoint_path
        if mode == "finetune" and checkpoint_path is None:
            load_checkpoint_path = getattr(self.config, 'load_checkpoint', None)
            if load_checkpoint_path is None:
                raise ValueError(
                    "In finetune mode, checkpoint_path cannot be None. "
                    "Please provide checkpoint_path or set config.load_checkpoint"
                )
            checkpoint_path = load_checkpoint_path
        elif checkpoint_path is None and hasattr(self.config, 'load_checkpoint'):
            checkpoint_path = self.config.load_checkpoint

        # Initialize parallel config and wrappers
        self._init_parallel_config()

        # Load checkpoint
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path, mode)

        # Initialize training state
        self.state = TrainerState(
            max_steps=getattr(self.config, 'max_steps', 1000),
            eval_steps=getattr(self.config, 'eval_steps', 100),
            save_steps=getattr(self.config, 'save_steps', 100),
            global_batch_size=getattr(self.config, 'global_batch_size', 0),
        )

        # Calculate epoch step
        if self.train_dataset is not None:
            self.state.epoch_step = self._get_dataset_size(self.train_dataset)

        # Call train begin callback
        if self.callback_handler is not None:
            self.callback_handler.on_train_begin(self.config, self.state)

        # Execute training loop
        self._inner_train_loop(do_eval)

        # Call train end callback
        if self.callback_handler is not None:
            self.callback_handler.on_train_end(self.config, self.state)

    def _init_parallel_config(self):
        """Initialize parallel configuration."""
        # Initialize parallel configuration
        # 1) HSDP wrapper
        # 2) Pipeline parallel config
        # 3) Data parallel config
        logger.info("Initializing parallel config...")

    def _load_checkpoint(self, checkpoint_path: str, mode: str):
        """
        Load checkpoint from file.

        Args:
            checkpoint_path: Path to checkpoint file
            mode: Training mode ("pretrain" or "finetune")
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from: {checkpoint_path}")

        # Prepare global_step possibly adjusted by global_batch_size differences later
        global_step = getattr(self.state, 'global_step', None) if hasattr(self, 'state') else None

        # balanced_load flag from config if available
        balanced_load = getattr(self.config, 'balanced_load', False)

        # Use updated API signature per spec
        load_checkpoint(
            checkpoint=checkpoint_path,
            network=self.model,
            optimizer=self.optimizer if mode == "pretrain" else None,
            global_step=global_step,
            balanced_load=balanced_load,
        )

        # Implement: 需要将commoninfo的参数设置给lr模块、数据集模块等，请根据BaseTrainer的逻辑完善该部分逻辑的修改

    def _get_dataset_size(self, dataset) -> int:
        """
        Get the size of a dataset.

        Args:
            dataset: Dataset instance

        Returns:
            Number of batches in the dataset
        """
        if hasattr(dataset, '__len__'):
            return len(dataset)
        if hasattr(dataset, 'get_dataset_size'):
            return dataset.get_dataset_size()
        # Cannot determine dataset size; raise error per spec
        raise ValueError("Unable to determine dataset size from the provided dataset.")

    def _inner_train_loop(self, do_eval: bool = False):
        """
        Internal training loop.

        Args:
            do_eval: Whether to run evaluation
        """
        if self.train_dataset is None:
            raise ValueError("train_dataset is None, cannot train.")

        # Create dataset iterator
        dataset_iter = self._create_dataset_iterator(self.train_dataset)

        # Training loop
        step = self.state.global_step
        while step < self.state.max_steps:
            # Check epoch begin
            if step % self.state.epoch_step == 0 and step > 0:
                if self.callback_handler is not None:
                    self.callback_handler.on_epoch_begin(self.config, self.state)
                self.state.update_epoch()

            # Get batch data
            try:
                inputs = self.get_batch(dataset_iter)
            except StopIteration:
                # Recreate iterator if dataset exhausted
                dataset_iter = self._create_dataset_iterator(self.train_dataset)
                inputs = self.get_batch(dataset_iter)

            # Step begin callback
            if self.callback_handler is not None:
                self.callback_handler.on_step_begin(self.config, self.state)

            # Training step
            try:
                loss = self.training_step(self.model, inputs)
            except Exception as e:
                logger.error(f"Error in training step {step}: {e}")
                raise

            # Update state
            self.state.global_step += 1
            step = self.state.global_step

            # Step end callback (pass loss)
            if self.callback_handler is not None:
                self.callback_handler.on_step_end(
                    self.config,
                    self.state,
                    loss=loss
                )

            # Check epoch end
            if step % self.state.epoch_step == 0:
                if self.callback_handler is not None:
                    self.callback_handler.on_epoch_end(self.config, self.state)

            # Evaluation
            if do_eval and self.state.eval_steps > 0 and step % self.state.eval_steps == 0:
                self.evaluate()

    def _create_dataset_iterator(self, dataset):
        """
        Create an iterator for the dataset using MindSpore Dataset API.

        Args:
            dataset: mindspore.dataset.Dataset instance

        Returns:
            Dictionary iterator from MindSpore dataset
        """
        if hasattr(dataset, 'create_dict_iterator'):
            return dataset.create_dict_iterator()
        raise TypeError(f"Dataset type {type(dataset)} does not support create_dict_iterator()")

    def get_batch(self, dataset_iter) -> Dict[str, Any]:
        """
        Get a batch of data from the dataset.

        Modes:
        - Distributed dataset mode
        - Remove redundant load mode
        - Naive loading mode
        """
        use_distribute_dataset = getattr(self.config, 'use_distribute_dataset', False)
        use_remove_redundant_dataset = getattr(self.config, 'use_remove_redundant_dataset', False)

        if use_distribute_dataset:
            data = self._get_batch_distributed(dataset_iter)
        elif use_remove_redundant_dataset:
            data = self._get_batch_remove_redundant(dataset_iter)
        else:
            data = self._get_batch_naive(dataset_iter)

        # Ensure dict output
        if data is not None and not isinstance(data, dict):
            if isinstance(data, (tuple, list)):
                data = {"input_ids": data[0]} if len(data) > 0 else {}
        return data if data is not None else {}

    def _get_batch_distributed(self, dataset_iter):
        """Fetch next batch in distributed dataset mode (simplified)."""
        return next(dataset_iter)

    def _get_batch_remove_redundant(self, dataset_iter):
        """Fetch next batch in remove-redundant mode (simplified)."""
        return next(dataset_iter)

    def _get_batch_naive(self, dataset_iter):
        """Fetch next batch in naive loading mode (simplified)."""
        return next(dataset_iter)

    def compute_loss(
        self,
        model,
        inputs: Dict[str, Any]
    ):
        """
        Compute loss for the model.

        Args:
            model: The model
            inputs: Input data dictionary

        Returns:
            Loss value
        """
        # Forward pass
        outputs = model(**inputs)

        # Get labels from inputs
        labels = inputs.get('labels', None)

        # Compute loss
        if self.compute_loss_func is not None:
            # Use user-defined loss function
            loss = self.compute_loss_func(outputs, labels)
        else:
            # Extract loss from model output
            # We don't use .loss here since the model may return tuples instead of ModelOutput
            if isinstance(outputs, dict):
                loss = outputs["loss"]
            else:
                # Assume first element is loss
                loss = outputs[0]

        return loss

    def training_step(
        self,
        model,
        inputs: Dict[str, Any]
    ):
        """
        Perform a single training step.

        Args:
            model: The model
            inputs: Input data dictionary

        Returns:
            Loss value
        """
        # Forward and compute loss
        loss = self.compute_loss(model, inputs)

        # Backward pass
        # In real implementation with MindSpore:

        # Optimizer step

        return loss

    def evaluate(self):
        """Placeholder for evaluation; to be implemented."""
