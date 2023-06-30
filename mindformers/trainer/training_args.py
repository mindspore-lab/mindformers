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
# This file was refer to project:
# https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py
# ============================================================================
"""Default Training Arguments for Trainer."""

import os
from typing import Optional, Union
from dataclasses import dataclass, field

from mindformers.tools.register import MindFormerConfig
from mindformers.tools import logger
from .utils import LRType, OptimizerType, SaveIntervalStrategy


def _check_task_config(check_config):
    """check task config for adapting hugging-face."""
    if check_config is not None and isinstance(check_config, MindFormerConfig):
        return True
    return False


def _check_training_args(ori_value, new_value):
    """check training arguments for adapt MindFormers."""
    if new_value is not None:
        return new_value
    return ori_value


@dataclass
class TrainingArguments:
    """
    TrainingArguments is the subset of the arguments we use
    in our default config **which is relate to the training in MindSpore**.
    """
    output_dir: str = field(
        default=None,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    use_parallel: bool = field(
        default=None,
        metadata={"help": "The use_parallel is used to enable distribute parallel of the network."}
    )
    profile: bool = field(
        default=None,
        metadata={"help": "The profile is used to enable profiling of the network."}
    )
    sink_mode: bool = field(
        default=None,
        metadata={"help": "The sink_mode is used to enable data sink of the network."}
    )
    sink_size: bool = field(
        default=None,
        metadata={"help": "The sink_size is used to enable data sink number per step for training or evaluation."}
    )
    batch_size: int = field(
        default=None, metadata={"help": "Global batch size per GPU/TPU core/CPU for training and evaluation."}
    )
    per_device_train_batch_size: int = field(
        default=None, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=None, metadata={"help": "Batch size  m  per GPU/TPU core/CPU for evaluation."}
    )

    learning_rate: float = field(default=None, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=None, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=None, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=None, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=None, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=None, metadata={"help": "Max gradient norm."})

    num_train_epochs: float = field(default=None, metadata={"help": "Total number of training epochs to perform."})

    lr_scheduler_type: Union[LRType, str] = field(
        default=None,
        metadata={"help": "The lr scheduler type to use."},
    )
    optim: Union[OptimizerType, str] = field(
        default=None,
        metadata={"help": "The optimizer type to use."},
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})

    save_strategy: Union[SaveIntervalStrategy, str] = field(
        default=None,
        metadata={"help": "The checkpoint save strategy to use."},
    )
    integrated_save: bool = field(
        default=None, metadata={
            "help": (
                " Whether to merge and save the split Tensor in the automatic parallel scenario. "
                "Integrated save function is only supported in automatic parallel scene, not supported"
                "in manual parallel."
            )
        }
    )
    save_steps: int = field(default=None, metadata={"help": "Save checkpoint every X updates steps."})
    save_seconds: int = field(default=None, metadata={"help": "Save checkpoint every X updates seconds."})
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Limit the total amount of checkpoints. "
                "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
            )
        },
    )

    seed: int = field(default=None, metadata={"help": "Random seed that will be set at the beginning of training."})
    data_seed: Optional[int] = field(default=None, metadata={"help": "Random seed to be used with data samplers."})

    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your model."},
    )
    initial_epoch: int = field(
        default=None,
        metadata={"help": "The initial_epoch is used to resume training from the init epoch."}
    )

    device_num = int(os.getenv("RANK_SIZE", "1"))
    device_id = int(os.getenv("DEVICE_ID", "0"))
    rank_id = int(os.getenv("RANK_ID", "0"))

    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)
        if self.lr_scheduler_type is not None:
            self.lr_scheduler_type = LRType(self.lr_scheduler_type).value
        if self.optim is not None:
            self.optim = OptimizerType(self.optim).value
        if self.save_strategy is not None:
            self.save_strategy = SaveIntervalStrategy(self.save_strategy).value

    @property
    def get_device_num(self):
        """get device num for training."""
        return self.device_num

    @property
    def get_device_id(self):
        """get device id for training."""
        return self.device_id

    @property
    def get_rank_id(self):
        """get rank id for training."""
        return self.rank_id

    def convert_args_to_mindformers_config(self, task_config: MindFormerConfig = None):
        """convert training arguments to mindformer config type for adapting hugging-face."""
        if task_config is None:
            task_config = MindFormerConfig()

        self._adapt_dataset_config(task_config)
        self._adapt_runner_config(task_config)
        self._adapt_lr_schedule_config(task_config)
        self._adapt_optimizer_config(task_config)
        self._adapt_save_checkpoint_config(task_config)

        task_config.output_dir = _check_training_args(task_config.output_dir, self.output_dir)
        task_config.seed = _check_training_args(task_config.seed, self.seed)
        task_config.data_seed = _check_training_args(task_config.data_seed, self.data_seed)
        task_config.profile = _check_training_args(task_config.profile, self.profile)
        task_config.use_parallel = _check_training_args(task_config.use_parallel, self.use_parallel)
        task_config.load_checkpoint = _check_training_args(
            task_config.load_checkpoint, self.resume_from_checkpoint)

    def _adapt_dataset_config(self, task_config):
        """adapt dataset config."""
        if _check_task_config(task_config.train_dataset):
            task_config.train_dataset.batch_size = _check_training_args(
                task_config.train_dataset.batch_size, self.per_device_train_batch_size)
        else:
            logger.warning(
                "This task does not support training at the moment; "
                "it does not have a default training data configuration,"
                "so the per_device_train_batch_size setting will not take effect")
        if _check_task_config(task_config.eval_dataset):
            task_config.eval_dataset.batch_size = task_config.train_dataset.batch_size = _check_training_args(
                task_config.eval_dataset.batch_size, self.per_device_eval_batch_size)
        else:
            logger.warning(
                "This task does not support evaluate at the moment;"
                "it does not have a default evaluate data configuration,"
                "so the per_device_eval_batch_size setting will not take effect")

    def _adapt_runner_config(self, task_config):
        """adapt runner config."""
        if _check_task_config(task_config.runner_config):
            task_config.runner_config.epochs = _check_training_args(
                task_config.runner_config.epochs, self.num_train_epochs)
            task_config.runner_config.batch_size = _check_training_args(
                task_config.runner_config.batch_size, self.batch_size)
            task_config.runner_config.sink_size = _check_training_args(
                task_config.runner_config.sink_size, self.sink_size)
            task_config.runner_config.per_epoch_size = _check_training_args(
                task_config.runner_config.per_epoch_size, self.sink_size)
            task_config.runner_config.sink_mode = _check_training_args(
                task_config.runner_config.sink_mode, self.sink_mode)
            task_config.runner_config.initial_epoch = _check_training_args(
                task_config.runner_config.initial_epoch, self.initial_epoch)

    def _adapt_lr_schedule_config(self, task_config):
        """adapt lr schedule config."""
        if _check_task_config(task_config.lr_schedule):
            if task_config.lr_schedule.type is not None:
                task_config.lr_schedule.type = _check_training_args(
                    task_config.lr_schedule.type, self.lr_scheduler_type)
            if task_config.lr_schedule.learning_rate is not None:
                task_config.lr_schedule.learning_rate = _check_training_args(
                    task_config.lr_schedule.learning_rate, self.learning_rate)
            if task_config.lr_schedule.warmup_steps is not None:
                task_config.lr_schedule.warmup_steps = _check_training_args(
                    task_config.lr_schedule.warmup_steps, self.warmup_steps)

    def _adapt_optimizer_config(self, task_config):
        """adapt optimizer config."""
        if _check_task_config(task_config.optimizer):
            if task_config.optimizer.type is not None:
                task_config.optimizer.type = _check_training_args(
                    task_config.optimizer.type, self.optim)
            if task_config.optimizer.learning_rate is not None:
                task_config.optimizer.learning_rate = _check_training_args(
                    task_config.optimizer.learning_rate, self.learning_rate)
            if task_config.optimizer.weight_decay is not None:
                task_config.optimizer.weight_decay = _check_training_args(
                    task_config.optimizer.weight_decay, self.weight_decay)
            if task_config.optimizer.beta1 is not None:
                task_config.optimizer.beta1 = _check_training_args(
                    task_config.optimizer.beta1, self.adam_beta1)
            if task_config.optimizer.beta2 is not None:
                task_config.optimizer.beta2 = _check_training_args(
                    task_config.optimizer.beta2, self.adam_beta2)
            if task_config.optimizer.eps is not None:
                task_config.optimizer.eps = _check_training_args(
                    task_config.optimizer.eps, self.adam_epsilon)

    def _adapt_save_checkpoint_config(self, task_config):
        """adapt save checkpoint config."""
        if task_config.callbacks is not None and \
                isinstance(task_config.callbacks, list) and \
                self.save_strategy is not None:
            for i, callback in enumerate(task_config.callbacks):
                if isinstance(callback, dict) and callback['type'] == "CheckpointMointor":
                    if self.save_strategy == 'no':
                        task_config.callbacks.pop(i)
                        continue
                    if self.save_strategy == 'steps':
                        task_config.callbacks[i]['save_checkpoint_steps'] = _check_training_args(
                            task_config.callbacks[i]['save_checkpoint_steps'], self.save_steps)
                    elif self.save_strategy == 'seconds':
                        task_config.callbacks[i]['save_checkpoint_seconds'] = _check_training_args(
                            task_config.callbacks[i]['save_checkpoint_seconds'], self.save_seconds)
                    task_config.callbacks[i]['keep_checkpoint_max'] = self.save_total_limit \
                        if self.save_total_limit else 5
                    task_config.callbacks[i]['integrated_save'] = _check_training_args(
                        task_config.callbacks[i]['integrated_save'], self.integrated_save)
