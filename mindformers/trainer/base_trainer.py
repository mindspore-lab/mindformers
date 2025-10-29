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
import re
import subprocess
from pprint import pprint
from functools import partial
from typing import Optional, Union, List
from collections import OrderedDict
import numpy as np
from PIL.Image import Image
import mindspore as ms
from mindspore import Tensor
from mindspore import get_ckpt_path_with_strategy
from mindspore.communication import get_rank
from mindspore.communication.management import get_group_size
from mindspore.train.model import Model
from mindspore.train import Callback
from mindspore.dataset import GeneratorDataset
from mindspore.dataset.engine.datasets import Dataset
from mindspore.train.metrics import get_metrics
from mindspore.nn.wrap.cell_wrapper import _VirtualDatasetCell
from mindspore.nn import Optimizer, Cell, PipelineCell, MicroBatchInterleaved
from mindspore.nn.wrap.cell_wrapper import GradAccumulationCell
from mindspore.nn.utils import no_init_parameters
from mindformers.mindformer_book import MindFormerBook
from mindformers.core import build_lr, build_optim, build_callback, build_metric
from mindformers.core.context.build_context import is_legacy_model
from mindformers.core.parallel_config import build_parallel_config
from mindformers.dataset import build_dataset, check_dataset_config, check_dataset_iterable, BaseDataset
from mindformers.models import build_network, build_processor, build_tokenizer, \
    PreTrainedModel, PreTrainedTokenizerBase, BaseImageProcessor
from mindformers.pipeline import pipeline
from mindformers.wrapper import build_wrapper
from mindformers.wrapper.wrapper import GradAccumulationCellWithTwoOutput, GradAccumulationCellWithMultiOutputs, \
    PipelineCellWithTwoOutput, PipelineCellWithMultiOutputs
from mindformers.tools.register import MindFormerConfig
from mindformers.wrapper.wrapper import DataOrderWrapperCell
from mindformers.tools.logger import logger
from mindformers.utils.tensorboard import _set_tensorboard_writer, _unset_tensorboard_writer, \
    write_args_to_tensorboard, update_tensorboard_args
from mindformers.utils.resume_ckpt_utils import get_resume_checkpoint, load_resume_checkpoint
from mindformers.tools.utils import count_params, get_real_rank, get_real_group_size, FILE_PERMISSION
from mindformers.tools.check_rules import check_rules
from mindformers.core.callback.callback import (
    EvalCallBack,
    MFLossMonitor,
    TrainingStateMonitor,
    CheckpointMonitor,
    ColdHotExpertMonitor,
    TopkBiasBalanceCallback
)
from mindformers.modules.seq_pipe import SequenceSplit
from mindformers.utils.load_checkpoint_utils import get_load_path_after_hf_convert
from mindformers.checkpoint.checkpoint import load_checkpoint, CommonInfo
from mindformers.checkpoint.utils import compile_model
from ..core.config_args import ConfigArguments
from .training_args import TrainingArguments
from .utils import (
    check_runner_config,
    transform_and_load_checkpoint,
    load_resume_context_from_checkpoint,
    get_last_checkpoint,
    preload_ckpt
)
from .optimizer_grouped_parameters import get_optimizer_grouped_parameters
from .utils import set_seed, check_train_data_loader_type, \
    check_eval_data_loader_type, check_optimizer_and_lr_type, check_wrapper_config
from ..version_control import check_tft_valid, check_tre_valid, check_tsp_valid, check_is_reboot_node

# pylint: disable=import-outside-toplevel
SUPPORT_TASKS = MindFormerBook().get_trainer_support_task_list()
SUPPORT_MODEL_NAMES = MindFormerBook().get_model_name_support_list()
SUPPORT_PIPELINES = MindFormerBook().get_pipeline_support_task_list()
SUPPORT_PIPELINE_INPUT_DATA = MindFormerBook().get_pipeline_support_input_data_list()
CURRENT_PROJECT_PATH = MindFormerBook().get_project_path()
DEFAULT_CONFIG_DIR = 'configs'
NEED_MERGES_FILE_TOKENIZERS = ["Qwen2Tokenizer"]
CALLBACK_HAS_SORT = [
    MFLossMonitor, TrainingStateMonitor, ColdHotExpertMonitor, TopkBiasBalanceCallback, CheckpointMonitor
]


class BaseTrainer:
    r"""Base Task Trainer.
    Args:
        task (str): The task name supported.
        model_name (str): The model name of Task-Trainer. Default: None
    Raises:
        NotImplementedError: If train method or evaluate method or predict method not implemented.
    """

    def __init__(self, task: str = None, model_name: str = None):

        host_name_output = subprocess.run(['hostname'], shell=False, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, encoding='utf-8', check=True)
        host_ip_output = subprocess.run(['hostname', '-I'], shell=False, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, encoding='utf-8', check=True)
        host_name = host_name_output.stdout.strip()
        host_ip = host_ip_output.stdout.strip().split(' ')[0]
        logger.info(f"host_name: {host_name}, host_ip: {host_ip}")

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
        self.compute_metrics = None
        self.kwargs = None
        self.pipeline_task = None
        self.global_batch_size = None

        self.network_delay_inited = False
        self.optimizer_delay_inited = False

        self.lr_scheduler = None
        self.grouped_lr_scheduler = None

        if task not in SUPPORT_TASKS.keys():
            logger.warning("Input task name is not in the supported list or unspecified.")

        if task in SUPPORT_TASKS.keys() and model_name not in SUPPORT_TASKS.get(task).keys():
            model_name_support_list = list(MindFormerBook().get_model_name_support_list_for_task(task))
            model_name_support_list.sort()
            logger.warning("Input model name is not in the supported list or unspecified.")
            logger.warning("See the list of supported task and model name: %s", model_name_support_list)
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
        if os.environ.get("RUN_MODE") != "predict":
            self._check_grad_accumulation_steps()
            self._check_global_batch_size_for_auto_parallel()
            self._reset_wrapper()

        return self.config

    def setup_task_config(self):
        """Set up the default task config."""
        task_config = None
        if self.task in SUPPORT_TASKS.keys() and self.model_name in SUPPORT_TASKS.get(self.task).keys():
            task_config = MindFormerConfig(SUPPORT_TASKS.get(self.task).get(self.model_name))

        if isinstance(task_config, MindFormerConfig):
            self.default_task_config = task_config
        else:
            logger.warning("If the default config arguments is not specified,"
                           "you must define the required model, optimizer, and so on"
                           "in the train or evaluate or predict attribute function.")

    def _check_global_batch_size_for_auto_parallel(self):
        """Check global batch size in auto parallel mode."""
        batch_size = self.config.runner_config.batch_size
        self.config.runner_config.mini_batch_size = batch_size
        gradient_accumulation_steps = self.config.runner_config.gradient_accumulation_steps
        dp = self.config.parallel_config.data_parallel
        micro_batch_num = self.config.parallel_config.micro_batch_num
        micro_batch_interleave_num = self.config.micro_batch_interleave_num
        parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        full_batch = ms.get_auto_parallel_context("full_batch")
        ds_stra = ms.get_auto_parallel_context("dataset_strategy")
        pp = self.get_pipeline_stages()

        if parallel_mode in ["semi_auto_parallel", "auto_parallel"]:
            if pp == 1 and micro_batch_num > 1:
                logger.warning("When pipeline parallel is not enabled, "
                               "config.parallel_config.micro_batch_num does not take effect. Reset it to 1.")
                micro_batch_num = self.config.parallel_config.micro_batch_num = 1
            if full_batch:
                if ds_stra != 'full_batch':
                    logger.warning(f"full_batch=True only supports dataset_strategy='full_batch', "
                                   f"reset dataset_strategy {ds_stra} to 'full_batch'.")
                    ms.set_auto_parallel_context(dataset_strategy='full_batch')

                if pp > 1:
                    self.global_batch_size = batch_size * dp * micro_batch_num * micro_batch_interleave_num
                    logger.info("Pipeline parallel was opened: pipeline_stages = %s, full batch is True, "
                                "gradient_accumulation_steps will not take effect in pipeline parallel, "
                                "global batch size will be changed: "
                                "global_batch_size = "
                                "batch_size * data_parallel * micro_batch_num * micro_batch_interleave_num "
                                "= %s = %s * %s * %s * %s).",
                                pp, self.global_batch_size, batch_size, dp, micro_batch_num,
                                micro_batch_interleave_num)
                    self.config.runner_config.batch_size = self.global_batch_size
                    self._reset_wrapper_for_pipeline_parallel()
                else:
                    self.global_batch_size = batch_size * dp * micro_batch_interleave_num * gradient_accumulation_steps
                    logger.info("The current parallel mode is %s, full batch is True,"
                                "so global batch size will be changed: "
                                "global_batch_size = batch_size * data_parallel * micro_batch_interleave_num "
                                "* gradient_accumulation_steps = %s = %s * %s * %s * %s",
                                parallel_mode, self.global_batch_size, batch_size, dp, micro_batch_interleave_num,
                                gradient_accumulation_steps)
                    self.config.runner_config.batch_size = self.global_batch_size
            else:  # full_batch = False
                if not isinstance(ds_stra, (tuple, list)):
                    raise ValueError("If set full_batch=False, dataset_strategy must be set as 'tuple', "
                                     "such as ((dp, 1), ).")
                ds_stra_dp = ds_stra[0][0]
                if dp != ds_stra_dp:
                    raise ValueError(f"data_parallel {dp} should be equal to dataset_strategy[0][0] {ds_stra_dp}.")

                if pp > 1:
                    per_batch_size = batch_size * micro_batch_num * micro_batch_interleave_num
                    self.global_batch_size = per_batch_size * dp
                    logger.info("Pipeline parallel was opened: pipeline_stages = %s, full batch is False, "
                                "gradient_accumulation_steps will not take effect in pipeline parallel, "
                                "batch size per card will be changed: "
                                "per_batch_size = batch_size * micro_batch_num * micro_batch_interleave_num "
                                "= %s = %s * %s * %s).",
                                pp, per_batch_size, batch_size, micro_batch_num,
                                micro_batch_interleave_num)
                    logger.info("global_batch_size = per_batch_size * data_parallel = %s * %s = %s",
                                per_batch_size, dp, self.global_batch_size)
                    self.config.runner_config.batch_size = per_batch_size
                    self._reset_wrapper_for_pipeline_parallel()
                else:
                    per_batch_size = batch_size * micro_batch_interleave_num * gradient_accumulation_steps
                    self.global_batch_size = per_batch_size * dp
                    logger.info("The current parallel mode is %s, full batch is False, "
                                "batch size per card will be changed: "
                                "per_batch_size = batch_size * micro_batch_interleave_num * "
                                "gradient_accumulation_steps = %s = %s * %s * %s).",
                                parallel_mode, per_batch_size, batch_size, micro_batch_interleave_num,
                                gradient_accumulation_steps)
                    logger.info("global_batch_size = per_batch_size * data_parallel = %s * %s = %s",
                                per_batch_size, dp, self.global_batch_size)
                    self.config.runner_config.batch_size = per_batch_size
        else:
            logger.info("The current parallel mode is %s, batch size per card will not be changed: "
                        "batch_size_per_card = %s",
                        parallel_mode, batch_size)
            self.global_batch_size = batch_size * get_real_group_size() * gradient_accumulation_steps
            logger.info(
                "global_batch_size = batch_size_per_card * device_num * gradient_accumulation_steps "
                "= %s = %s * %s * %s",
                self.global_batch_size, batch_size, get_real_group_size(), gradient_accumulation_steps)
            self.config.runner_config.batch_size = batch_size * gradient_accumulation_steps
            self.config.parallel_config.data_parallel = 1
            self.config.parallel_config.model_parallel = 1
            self.config.parallel_config.context_parallel = 1
            self.config.parallel_config.expert_parallel = 1
            self.config.parallel_config.pipeline_stage = 1
            self.config.parallel_config.micro_batch_num = 1
            logger.info("parallel_config will be change to default config: %s.",
                        self.config.parallel_config)
        self.config.runner_config.global_batch_size = self.global_batch_size

    def _check_grad_accumulation_steps(self):
        """check the gradient accumulation steps."""
        if self.config.runner_config.gradient_accumulation_steps is None:
            self.config.runner_config.gradient_accumulation_steps = 1
        if not isinstance(self.config.runner_config.gradient_accumulation_steps, int) or \
            isinstance(self.config.runner_config.gradient_accumulation_steps, bool):
            raise ValueError("gradient_accumulation should be integer but got "
                             f"{type(self.config.runner_config.gradient_accumulation_steps)}")
        if not self.config.runner_config.gradient_accumulation_steps >= 1:
            raise ValueError("gradient_accumulation should be greater or equal than 1, "
                             f"but got {self.config.runner_config.gradient_accumulation_steps}")
        parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        pp = self.get_pipeline_stages()
        if parallel_mode in ["semi_auto_parallel", "auto_parallel"] and pp > 1 \
            and self.config.runner_config.gradient_accumulation_steps > 1:
            logger.warning("gradient_accumulation_steps will not take effect when using pipeline parallel, "
                           "reset the gradient_accumulation_steps from %s to 1.",
                           self.config.runner_config.gradient_accumulation_steps)
            self.config.runner_config.gradient_accumulation_steps = 1
        # grad accumulation not supported in data parallel/standalone mode for now
        if self.config.runner_config.gradient_accumulation_steps > 1 and \
            parallel_mode not in ["semi_auto_parallel", "auto_parallel"]:
            logger.warning("gradient_accumulation_steps currently need to be used in semi_auto_parallel mode, "
                           "but got %s mode, please check your runner config and parallel config. "
                           "Reset the gradient_accumulation_steps from %s to 1. ",
                           parallel_mode, self.config.runner_config.gradient_accumulation_steps)
            self.config.runner_config.gradient_accumulation_steps = 1
        if self.config.runner_config.gradient_accumulation_steps == 1 and pp == 1:
            logger.warning("The Lazy Inline compilation acceleration feature "
                           "only works with using gradient_accumulation_steps > 1 "
                           "when not in pipeline parallel mode (pipeline_stage = 1). "
                           "Current pipeline stage=1 but gradient_accumulation_steps=1, "
                           "the feature is disabled by default.")
            self.config.model.model_config.disable_lazy_inline = True

    @staticmethod
    def _check_training_network_no_use_past(network):
        if network is not None and hasattr(network.config, "use_past") and network.config.use_past:
            raise ValueError("In training process, network should be configured to use_past=False, "
                             f"but got use_past={network.config.use_past}")

    def _reset_wrapper(self):
        """reset wrapper in some special cases."""
        parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        gradient_accumulation_steps = self.config.runner_config.gradient_accumulation_steps
        # use gradient_accumulation, reset wrapper to MFPipelineWrapper
        if parallel_mode in ["semi_auto_parallel", "auto_parallel"] and gradient_accumulation_steps > 1:
            self._reset_wrapper_for_grad_accu(gradient_accumulation_steps)

    def _reset_wrapper_for_pipeline_parallel(self):
        """Reset wrapper when pipeline parallel."""
        if self.config.runner_wrapper is not None:
            self.config.runner_wrapper.type = "MFPipelineWithLossScaleCell" \
                if self.config.runner_wrapper.type != "MFPipelineWithLossScaleCell" else self.config.runner_wrapper.type
            self.config.runner_wrapper.micro_batch_num = self.config.parallel_config.micro_batch_num
            self.config.runner_wrapper.calculate_per_token_loss = self.config.calculate_per_token_loss
            logger.warning(
                "When using the pipeline parallel mode, "
                "the MFPipelineWithLossScaleCell class is used by default.")
        else:
            logger.info(
                "When using the pipeline parallel mode, "
                "because the wrapper class is not specified, "
                "MindSpore's built-in PipelineCell is used by default")
        logger.info("PipelineWrapper under evaluate or predict mode will not take effect.")

    def _reset_wrapper_for_grad_accu(self, gradient_accumulation_steps):
        """Reset wrapper when using grad accumulation."""
        if self.config.runner_wrapper is not None:
            self.config.runner_wrapper.type = "MFPipelineWithLossScaleCell" \
                if self.config.runner_wrapper.type != "MFPipelineWithLossScaleCell" else self.config.runner_wrapper.type
            # set micro_batch_num as gradient_accumulation_steps in MFPipelineWithLossScaleCell
            self.config.runner_wrapper.micro_batch_num = gradient_accumulation_steps
            self.config.runner_wrapper.calculate_per_token_loss = self.config.calculate_per_token_loss
            logger.warning(
                "When using the gradient_accumulation_steps in semi/auto parallel mode, "
                "the MFPipelineWithLossScaleCell class is used by default.")
        else:
            logger.info(
                "When using the gradient_accumulation_steps in semi/auto parallel mode, "
                "because the wrapper class is not specified, "
                "MindSpore's built-in GradAccumulationCell is used by default")

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

    def create_dataset(self, is_train: bool = True, default_args: dict = None):
        """Create the dataset for training or evaluate."""
        dataset = self.train_dataset if is_train else self.eval_dataset
        dataset_task = self.config.train_dataset_task if is_train else self.config.eval_dataset_task

        if isinstance(dataset, (BaseDataset, Dataset)):
            return dataset

        if dataset is None:
            dataset = build_dataset(dataset_task, default_args=default_args)
        elif check_dataset_iterable(dataset):
            dataset_task.type = 'GeneralDataset'
            default_args = {} if default_args is None else default_args
            default_args["dataset"] = dataset
            dataset = build_dataset(dataset_task, default_args=default_args)
        else:
            raise ValueError("Dataset should be Dataset, iterator, iterable class which has `__iter__`, "
                             f"iterable class which has  `__get_item__` and `__len__`, but get {dataset}.")
        return dataset

    def create_train_dataset(self, default_args: dict = None):
        """Create the train dataset for training."""
        logger.info(".........Build Dataset From Config..........")
        self._reset_dataset_batch_size()
        train_dataset = self.create_dataset(is_train=True, default_args=default_args)
        return train_dataset

    def create_eval_dataset(self, default_args: dict = None):
        """Create the eval dataset for evaluate."""
        logger.info(".........Build Dataset From Config..........")
        self._reset_dataset_batch_size()
        parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        pipeline_stages = ms.get_auto_parallel_context("pipeline_stages")
        if parallel_mode in ["semi_auto_parallel", "auto_parallel"] and pipeline_stages > 1:
            self.config.eval_dataset.batch_size = \
                self.config.eval_dataset.batch_size // self.config.parallel_config.micro_batch_num
        if parallel_mode in ["semi_auto_parallel", "auto_parallel"]:
            self.config.eval_dataset.batch_size = \
                self.config.eval_dataset.batch_size // self.config.micro_batch_interleave_num
        # reduce batch size for that gradient_accumulation_steps will not take effect on eval
        if self.config.runner_config.gradient_accumulation_steps > 1:
            self.config.eval_dataset.batch_size = \
                self.config.eval_dataset.batch_size // self.config.runner_config.gradient_accumulation_steps
        if self.config.eval_dataset.batch_size < 1:
            logger.warning("eval_dataset batch_size is less than 1 after bs calculate, reset batch_size to 1, "
                           "please check your configs about batch_size, micro_batch_num micro_batch_interleave_num "
                           "and gradient_accumulation_steps.")
            self.config.eval_dataset.batch_size = 1
        logger.info("For evaluate phase, batch size for eval dataset is %s, different from training, "
                    "not multiplied by micro_batch_num, micro_batch_interleave_num and gradient_accumulation_steps",
                    self.config.eval_dataset.batch_size)
        eval_dataset = self.create_dataset(is_train=False, default_args=default_args)
        return eval_dataset

    def create_network(self, default_args: dict = None):
        """Create the network for task trainer."""
        logger.info(".........Build Network From Config..........")
        if self.config.get("pretrained_model_dir", None):
            self.config.model.pretrained_model_dir = self.config.pretrained_model_dir
        if self.config.get("generation_config", None):
            self.config.model.generation_config = self.config.generation_config
        network = build_network(self.config.model, default_args=default_args)
        if hasattr(network, "check_pipeline_stage") and callable(network.check_pipeline_stage):
            network.check_pipeline_stage()
        return network

    def create_network_without_param_init(self, default_args: dict = None):
        """Create the network for task trainer without initialize parameters."""
        with no_init_parameters():
            network = self.create_network(default_args=default_args)
        logger.info("Parameters are not initialized during model initialization.")
        self.network_delay_inited = True
        return network

    @staticmethod
    def warp_data_order_with_tool_cells(network, construct_args_key):
        """For passing parameters in lexicographical order."""
        return DataOrderWrapperCell(construct_args_key, network)

    def _wrap_network_with_tool_cells_legacy(self, network):
        """For training process, warp the network with some tool cells."""
        micro_batch_interleave_num = self.config.micro_batch_interleave_num
        gradient_accumulation_steps = self.config.runner_config.gradient_accumulation_steps
        parallel_mode = ms.context.get_auto_parallel_context("parallel_mode")
        pp = self.get_pipeline_stages()
        if micro_batch_interleave_num > 1:
            logger.info("micro_batch_interleave_num > 1, the double copy parallel feature is turned on.")
            network = MicroBatchInterleaved(network, micro_batch_interleave_num)
        if gradient_accumulation_steps > 1 and not pp > 1:
            logger.info("gradient_accumulation_steps > 1, GradAccumulationCell is wrapped on network. "
                        "It is suggested to use `Lazy Inline` feature to save compiling time.")
            if self.config.runner_wrapper.calculate_per_token_loss:
                network = GradAccumulationCellWithTwoOutput(network, gradient_accumulation_steps)
            else:
                network = GradAccumulationCell(network, gradient_accumulation_steps)
        if pp > 1:
            micro_batch_num = self.config.parallel_config.micro_batch_num
            seq_split_num = self.config.parallel_config.seq_split_num
            if seq_split_num > 1:
                if self.config.recompute_config.recompute:
                    raise ValueError("When using seq pipe, cannot apply full recompute.")
                network = SequenceSplit(network, split_num=seq_split_num)
            if self.config.runner_wrapper.calculate_per_token_loss:
                if seq_split_num > 1:
                    raise ValueError("When using seq pipe, cannot apply calculate_per_token_loss.")
                network = PipelineCellWithTwoOutput(network, micro_size=micro_batch_num)
            else:
                network = PipelineCell(network, micro_size=micro_batch_num)
        if parallel_mode in ["semi_auto_parallel", "auto_parallel"] and ms.get_context('mode') == 0:
            network = _VirtualDatasetCell(network)
            ds_broadcast_level = ms.context.get_context("dataset_broadcast_opt_level")
            if ds_broadcast_level > 0:
                # pylint: disable=W0212
                network._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
        return network

    def _wrap_network_with_tool_cells(self, network):
        """For training process, warp the network with some tool cells."""
        micro_batch_interleave_num = self.config.micro_batch_interleave_num
        gradient_accumulation_steps = self.config.runner_config.gradient_accumulation_steps
        parallel_mode = ms.context.get_auto_parallel_context("parallel_mode")
        pp = self.get_pipeline_stages()
        if micro_batch_interleave_num > 1:
            raise ValueError("micro_batch_interleave_num > 1, is not supported when use_legacy=False.")

        if not pp > 1:
            if gradient_accumulation_steps > 1:
                logger.info("gradient_accumulation_steps > 1, GradAccumulationCell is wrapped on network. "
                            "It is suggested to use `Lazy Inline` feature to save compiling time.")
                network = GradAccumulationCellWithMultiOutputs(network, gradient_accumulation_steps)
        else:
            micro_batch_num = self.config.parallel_config.micro_batch_num
            seq_split_num = self.config.parallel_config.seq_split_num
            if seq_split_num > 1:
                if self.config.recompute_config.recompute:
                    raise ValueError("When using seq pipe, cannot apply full recompute.")
                raise ValueError("seq_split_num > 1, is not supported when use_legacy=False.")
            if self.config.runner_wrapper.calculate_per_token_loss:
                if seq_split_num > 1:
                    raise ValueError("When using seq pipe, cannot apply calculate_per_token_loss.")
            network = PipelineCellWithMultiOutputs(network, micro_size=micro_batch_num)
        if parallel_mode in ["semi_auto_parallel", "auto_parallel"] and ms.get_context('mode') == 0:
            network = _VirtualDatasetCell(network)
            ds_broadcast_level = ms.context.get_context("dataset_broadcast_opt_level")
            if ds_broadcast_level > 0:
                # pylint: disable=W0212
                network._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
        return network

    def wrap_network_with_tool_cells(self, network):
        """For training process, warp the network with some tool cells."""
        if is_legacy_model():
            return self._wrap_network_with_tool_cells_legacy(network)
        return self._wrap_network_with_tool_cells(network)

    @staticmethod
    def wrap_eval_network_with_tool_cells(network):
        """For evaluate in training process, warp the network with some tool cells."""
        parallel_mode = ms.context.get_auto_parallel_context("parallel_mode")
        if parallel_mode in ["semi_auto_parallel", "auto_parallel"] and ms.get_context('mode') == 0:
            network = _VirtualDatasetCell(network)
            ds_broadcast_level = ms.context.get_context("dataset_broadcast_opt_level")
            if ds_broadcast_level > 0:
                # pylint: disable=W0212
                network._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
        return network

    def create_image_processor(self, default_args: dict = None):
        """Create the image processor for predict."""
        logger.info(".........Build Image Processor From Config..........")
        self.image_processor = build_processor(
            self.config.processor.image_processor, default_args=default_args)
        return self.image_processor

    def _create_grouped_lr_scheduler(self, learning_scale, scale_factor):
        """
        Create learning rate (LR) schedulers for both default and parameter-grouped configurations.
        """
        default_config = self.config.grouped_lr_schedule.default
        default_lr_scheduler = self.create_lr_scheduler(default_config, learning_scale, scale_factor)

        grouped_lr_scheduler = []
        grouped_config = self.config.grouped_lr_schedule.grouped

        # Iterate over each grouped LR configuration
        for lr_config in grouped_config:
            params = lr_config.pop('params', None)
            if not params:
                raise ValueError("If using grouped_lr_schedule, 'params' must be set correctly in `grouped`.")

            lr_config = MindFormerConfig(**lr_config)
            lr_scheduler = self.create_lr_scheduler(lr_config, learning_scale, scale_factor)
            grouped_lr_scheduler.append({
                'params': params,
                'lr_scheduler': lr_scheduler,
                'lr_config': lr_config
            })

        return default_lr_scheduler, grouped_lr_scheduler

    def create_optimizer_scheduler(self, network, model_params: set):
        """Create the optimizer for training."""
        logger.info("..........Build Optimizer From Config..........")
        # Get learning rate scaling settings
        learning_scale = self.config.lr_scale
        scale_factor = self.config.lr_scale_factor

        # Build the learning rate scheduler
        if self.config.grouped_lr_schedule is not None:
            # Create grouped LR scheduler if configured (e.g., different LRs for different parameter groups)
            self.lr_scheduler, self.grouped_lr_scheduler = self._create_grouped_lr_scheduler(
                learning_scale, scale_factor)
        else:
            # Create standard LR scheduler
            self.lr_scheduler = self.create_lr_scheduler(
                self.config.lr_schedule, learning_scale, scale_factor)

        # Build optimizer parameter groups (apply weight decay, LR groups, etc.)
        group_params = get_optimizer_grouped_parameters(
            model=network,
            weight_decay=self.config.optimizer.weight_decay,
            dynamic_lr_schedule=self.lr_scheduler,
            grouped_lr_schedule=self.grouped_lr_scheduler,
            model_params=model_params,
            optimizer_type=self.config.optimizer.type,
            layer_scale=self.config.layer_scale,
            layer_decay=self.config.layer_decay
        )

        # Build optimizer with dynamic lr_scheduler if available
        if self.lr_scheduler is not None:
            self.optimizer = build_optim(
                self.config.optimizer,
                default_args={"params": group_params, "learning_rate": self.lr_scheduler})
        else:
            # Otherwise, create lr_scheduler with static learning rate from config
            if self.config.optimizer.learning_rate is None:
                raise ValueError("")

            learning_rate = self.config.optimizer.learning_rate
            if learning_scale and scale_factor is not None:
                # reset learning_rate in optimizer with scale_factor if learning_scale is True
                self.config.optimizer.learning_rate = self.learning_rate_scale(learning_rate, scale_factor)

            # Build optimizer with fixed learning rate
            self.optimizer = build_optim(
                self.config.optimizer, default_args={"params": group_params})

        return self.optimizer

    def create_optimizer_scheduler_without_param_init(self, network, model_params: set):
        """Create the optimizer for training without initialize parameters."""
        with no_init_parameters():
            optimizer = self.create_optimizer_scheduler(network=network, model_params=model_params)
        logger.info("Parameters are not initialized during optimizer initialization.")
        self.optimizer_delay_inited = True
        return optimizer

    def create_lr_scheduler(self, lr_config: dict, learning_scale: bool = False, scale_factor: int = 256):
        """Create the learning rate scheduler."""
        logger.info("..........Build LR Schedule From Config..........")
        train_data_size = self.get_train_data_size()

        warmup_lr_init = None
        if lr_config:
            warmup_epochs = lr_config.warmup_epochs
            warmup_ratio = lr_config.warmup_ratio
            warmup_lr_init = lr_config.warmup_lr_init

            # Calculate total training steps depending on sink_mode
            if self.config.runner_config.sink_mode:
                total_steps = int(self.config.runner_config.epochs * self.config.runner_config.sink_size)
            else:
                total_steps = int(self.config.runner_config.epochs * train_data_size)

            # Set total_steps in `lr_schedule` if not explicitly defined
            if lr_config.total_steps is None or lr_config.total_steps == -1:
                lr_config.total_steps = total_steps
            else:
                lr_config.total_steps = int(lr_config.total_steps)

            # Compute warmup steps if warmup is enabled
            if warmup_epochs:
                if not isinstance(warmup_epochs, int) or warmup_epochs < 0:
                    raise ValueError("`warmup_epochs` must be a non-negative integer.")
                if not warmup_ratio:
                    raise ValueError("`warmup_ratio` must be specified when warmup is enabled.")

                warmup_steps = int(warmup_epochs * train_data_size)
                lr_config.warmup_steps = warmup_steps
                logger.info(
                    f"Get warmup_steps({warmup_steps}) = "
                    f"int(warmup_epochs({warmup_epochs}) * train_data_size({train_data_size}))"
                )

            if learning_scale and scale_factor is not None:
                lr_config.learning_rate = self.learning_rate_scale(lr_config.learning_rate, scale_factor)

        # Build learning rate scheduler from configuration
        lr_scheduler = build_lr(lr_config)

        if lr_scheduler and hasattr(lr_scheduler, "warmup_lr_init") and warmup_lr_init is None:
            logger.info(f"warmup_lr_init is not set. The default value {lr_scheduler.warmup_lr_init} will be applied.")
        return lr_scheduler

    def create_model_wrapper(self, network, optimizer):
        """Create the model wrapper for training."""
        logger.info(".........Build Model Wrapper for Train From Config..........")
        calculate_per_token_loss = getattr(self.config, "calculate_per_token_loss", False)
        use_skip_data_by_global_norm = getattr(self.config, "use_skip_data_by_global_norm", False)
        print_separate_loss = getattr(self.config, "print_separate_loss", True)
        global_norm_spike_threshold = 1.0
        if self.config.get('monitor_config') is not None:
            global_norm_spike_threshold = getattr(self.config.monitor_config, "global_norm_spike_threshold", 1.0)
        model_wrapper = build_wrapper(self.config.runner_wrapper,
                                      default_args={"network": network,
                                                    "optimizer": optimizer,
                                                    "parallel_config": self.config.parallel_config,
                                                    "calculate_per_token_loss": calculate_per_token_loss,
                                                    "global_norm_spike_threshold": global_norm_spike_threshold,
                                                    "print_separate_loss": print_separate_loss,
                                                    "use_skip_data_by_global_norm": use_skip_data_by_global_norm,
                                                    "lr_scheduler": self.lr_scheduler,
                                                    "grouped_lr_scheduler": self.grouped_lr_scheduler})
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
        extend_eval_callbacks = build_callback(self.config.eval_callbacks, default_args=default_args)
        if extend_eval_callbacks:
            self.eval_callbacks.extend(extend_eval_callbacks)
        return self.eval_callbacks

    def create_metrics(self, metric_name: str = None):
        """Create Metrics For Evaluate or Fit."""
        if self.compute_metrics is not None:
            self.compute_metrics = get_metrics(self.compute_metrics)
            return self.compute_metrics
        if self.config.metric is None:
            raise ValueError("When `do_eval` is True and `compute_metrics` is None, \
                              the config of metric must not be None.")
        if isinstance(self.config.metric, dict):
            self.config.metric = [self.config.metric]

        self.compute_metrics = {}
        for metric_config in self.config.metric:
            if "type" not in metric_config:
                raise ValueError("The type of metric is not found!")
            metric = build_metric(metric_config)
            if metric_name is None:
                metric_name = metric.__class__.__name__
            self.compute_metrics[metric_name] = metric

        return self.compute_metrics

    def count_parameters(self):
        """Count network parameters number."""
        if self.network is not None:
            logger.info("Network Parameters: %s M.", str(count_params(self.network)))
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
        if isinstance(network, (Cell, PreTrainedModel)):
            network.set_train(is_train)
        self.network = network

    def get_train_data_size(self):
        """Get train dataset size."""
        if self.train_dataset is None:
            raise NotImplementedError("train dataset is None")
        return self.train_dataset.get_dataset_size()

    def get_eval_data_size(self):
        """Get eval dataset size."""
        if self.eval_dataset is None:
            raise NotImplementedError("eval dataset is None")
        return self.eval_dataset.get_dataset_size()

    @staticmethod
    def get_pipeline_stages():
        """Get pipeline stages for task trainer."""
        pipeline_stages = ms.get_auto_parallel_context("pipeline_stages")
        return pipeline_stages

    def learning_rate_scale(self, base_learning_rate: float = 0., scale_factor: Optional[Union[float, int]] = 256.):
        """Scale learning rate for training."""
        if not isinstance(base_learning_rate, float):
            raise ValueError(f"learning rate must be float type, but get {type(base_learning_rate)}")
        if not isinstance(scale_factor, (float, int)):
            raise ValueError(f"scale_factor must be float or int type, but get {type(scale_factor)}")

        device_num = get_real_group_size()
        per_device_batch_size = self.config.train_dataset.batch_size
        learning_rate = (base_learning_rate * device_num * per_device_batch_size) / scale_factor
        return learning_rate

    def _process_megatron_dataset(self, dataset, config):
        """Dataset processing for Megatron Dataset."""
        if ms.context.get_context("dataset_broadcast_opt_level") < 3:
            raise ValueError("If using `BlendedMegatronDatasetDataLoader`, please set "
                             "`dataset_broadcast_opt_level: 3` in the `parallel_speed_up.json` file.")

        dataset_info = config.train_dataset.data_loader
        # reset dataset size to remove redundant data
        dataset = dataset.take(int(dataset_info.sizes[0]) // self.global_batch_size)
        logger.info(f"Use BlendedMegatronDatasetDataLoader, reset dataset size to {dataset.get_dataset_size()}.")

        # Sync assign eod compression arguments
        if self.config.train_dataset.data_loader.config.create_compressed_eod_mask:
            self.config.model.model_config.use_eod_attn_mask_compression = True
        return dataset, config

    @staticmethod
    def _check_sink_mode_with_ds_broadcast(config):
        """Check sink_mode with dataset_broadcast_opt_level."""
        ds_broadcast_level = ms.context.get_context("dataset_broadcast_opt_level")
        if ds_broadcast_level > 0 and not config.runner_config.sink_mode:
            raise ValueError(
                f"Current `dataset_broadcast_opt_level` is {ds_broadcast_level}, "
                f"if set larger than 0, then sink_mode should be set to be 'True'")

    @staticmethod
    def _check_input_sliced_sig(config, usage_info='special'):
        """Check input_sliced_sig in model config."""
        input_sliced_sig = config.model.model_config.get("input_sliced_sig")
        if not input_sliced_sig and is_legacy_model():
            raise ValueError(
                f"In this {usage_info} configuration, input_sliced_sig in model_config should be set 'True'")

    @staticmethod
    def _get_columns_with_strategy(config, dataset_config, dataloader_type):
        """
        Determine dataset sharding strategy and column names based on configuration.
        """
        dp = config.parallel_config.data_parallel

        # Default sharding strategy and column names
        dataset_strategy = 'full_batch'
        column_names = None

        create_compressed_eod_mask = dataset_config.get('create_compressed_eod_mask', False)
        create_attention_mask = dataset_config.get('create_attention_mask', False)

        if create_compressed_eod_mask:
            dataset_strategy = [[dp, 1], [dp, 1], [dp, 1], [dp, 1], [dp, 1]]
            column_names = ['input_ids', 'labels', 'loss_mask', 'position_ids', 'actual_seq_len']
            config.model.model_config.use_eod_attn_mask_compression = True

        elif create_attention_mask:
            dataset_strategy = [[dp, 1], [dp, 1], [dp, 1], [dp, 1], [dp, 1, 1, 1]]
            column_names = ['input_ids', 'labels', 'loss_mask', 'position_ids', 'attention_mask']

        elif dataloader_type == 'BlendedMegatronDatasetDataLoader':
            dataset_strategy = [[dp, 1], [dp, 1], [dp, 1], [dp, 1]]
            column_names = ['input_ids', 'labels', 'loss_mask', 'position_ids']

        # Allow overriding defaults from config
        dataset_strategy = config.parallel.get('dataset_strategy', dataset_strategy)
        column_names = config.train_dataset.get('input_columns', column_names)
        construct_args_key = config.train_dataset.get('construct_args_key', column_names)

        return config, dataset_strategy, column_names, construct_args_key

    @staticmethod
    def _set_full_batch_with_strategy(config, dataset_strategy):
        """
        Convert dataset strategy format and set auto parallel context.
        """
        # If full_batch is explicitly set in parallel config, log a warning
        full_batch = config.parallel.get('full_batch')
        if full_batch is not None:
            logger.warning("`full_batch` will be deprecated in future interfaces, use `dataset_strategy` instead.")

        # Convert strategy to correct format and set full_batch flag
        if dataset_strategy != 'full_batch':
            full_batch = False
            if isinstance(dataset_strategy, list):
                dataset_strategy = tuple(tuple(ds_item) for ds_item in dataset_strategy)
            elif not isinstance(dataset_strategy, tuple):
                raise ValueError('`dataset_strategy` should be list or tuple.')

        # Apply parallel context settings
        ms.set_auto_parallel_context(**{
            'full_batch': full_batch,
            'dataset_strategy': dataset_strategy
        })

        return full_batch, dataset_strategy

    @staticmethod
    def _set_dataset_broadcast_opt_level(config):
        """Set the dataset broadcast optimization level."""
        ds_broadcast_level = ms.context.get_context("dataset_broadcast_opt_level")
        if ds_broadcast_level == 0:
            logger.info("Set ds_broadcast_level=3 by default while enabling broadcast dataset.")
            ds_broadcast_level = 3

        # Override with the value from config if provided, otherwise keep the existing one
        ds_broadcast_level = config.context.get('dataset_broadcast_opt_level', ds_broadcast_level)
        ms.set_context(dataset_broadcast_opt_level=ds_broadcast_level)

    def _train_dataset_postprocess(self, dataset, config):
        """
        Postprocess the training dataset after construction.
        Mainly used to adjust dataset size for special dataloaders.
        """
        dataloader_config = config.train_dataset.get('data_loader', {})
        dataloader_type = dataloader_config.get('type')

        # Special handling for BlendedMegatronDatasetDataLoader
        if dataloader_type == "BlendedMegatronDatasetDataLoader":
            dataset_info = config.train_dataset.data_loader
            # Resize dataset to ensure it's divisible by global batch size (remove redundant data)
            dataset = dataset.take(int(dataset_info.sizes[0]) // self.global_batch_size)
            logger.info(
                f"Use BlendedMegatronDatasetDataLoader, reset dataset size to {dataset.get_dataset_size()}."
            )

        return dataset

    def _train_dataset_preprocess(self, config):
        """
        Preprocess training dataset configuration before building the dataloader.
        This includes validating broadcast options, setting dataset strategies,
        and assigning dataset column names depending on the dataloader type.
        """
        # Check dataset sink mode with dataset broadcast optimization level
        self._check_sink_mode_with_ds_broadcast(config)

        dataloader_config = config.train_dataset.get('data_loader', {})
        dataloader_type = dataloader_config.get('type')

        # Case 1: BlendedMegatronDatasetDataLoader
        if dataloader_type == 'BlendedMegatronDatasetDataLoader':
            # Validate that input slicing is enabled
            self._check_input_sliced_sig(config, dataloader_type)
            self._set_dataset_broadcast_opt_level(config)
            dataloader_config = dataloader_config.get('config')

        # Case 2: HFDataLoader or CommonDataLoader
        if dataloader_type in ['HFDataLoader', 'CommonDataLoader']:
            # Collect handler types if present
            handler = []
            if dataloader_config.handler is not None:
                handler = [sub_handler.get('type') for sub_handler in dataloader_config.handler]

            # Packing or attention mask requires extended dataset strategy
            if 'PackingHandler' in handler:
                self._check_input_sliced_sig(config, f"{dataloader_type} with packing")
                config.train_dataset.data_loader.create_attention_mask = True

            # Must use broadcast opt level > 0 if broadcast is enabled
            if dataloader_config.get('use_broadcast_data', True):
                self._set_dataset_broadcast_opt_level(config)

        config, dataset_strategy, column_names, construct_args_key = self._get_columns_with_strategy(
            config, dataloader_config, dataloader_type)
        logger.info(f"Got dataset config: "
                    f"dataset_strategy: {dataset_strategy}, "
                    f"column_names: {column_names}, "
                    f"construct_args_key: {construct_args_key}")

        full_batch, dataset_strategy = self._set_full_batch_with_strategy(config, dataset_strategy)

        # Update config with resolved column names and construct keys
        config.train_dataset.input_columns = column_names
        config.train_dataset.construct_args_key = construct_args_key
        config.train_dataset_task.dataset_config = config.train_dataset

        config.parallel.full_batch = full_batch
        config.parallel.dataset_strategy = dataset_strategy

        return config

    @staticmethod
    def resume_ckpt_path_with_strategy(config):
        """Get resume checkpoint path with strategy.

        This method finds the appropriate checkpoint path for the current rank when resuming training
        with a distributed strategy.

        Args:
            config (ConfigArguments): The configuration containing checkpoint and strategy settings.

        Returns:
            str: Path to the checkpoint file for the current rank after applying strategy.
                 Returns None if no valid checkpoint is found.

        Raises:
            ValueError: If strategy file for current rank does not exist.
        """

        cur_rank = get_rank()
        src_strategy_files = sorted(list(os.listdir(config.src_strategy_path_or_dir)))
        if len(src_strategy_files) - 1 < cur_rank:
            raise ValueError(f" rank {cur_rank} src_strategy is not exist")
        src_strategy_file = os.path.join(config.src_strategy_path_or_dir, src_strategy_files[cur_rank])
        if os.path.isfile(config.load_checkpoint):
            return get_ckpt_path_with_strategy(config.load_checkpoint, src_strategy_file)

        if os.path.isdir(config.load_checkpoint):
            device_nums = get_group_size()
            max_ckpt_path = ""
            max_time = 0
            for i in range(device_nums):
                cur_rank_ckpt_dir = os.path.join(config.load_checkpoint, f"rank_{i}")
                last_ckpt = get_last_checkpoint(cur_rank_ckpt_dir, config.load_ckpt_format)
                if last_ckpt is None:
                    continue
                cur_time = os.path.getmtime(last_ckpt)
                if cur_time > max_time:
                    max_time = cur_time
                    max_ckpt_path = last_ckpt
            pattern = fr'rank_\d+(?:_(\d+))?-(\d+)_(\d+)\.{config.load_ckpt_format}$'
            match = re.search(pattern, max_ckpt_path)
            if match:
                repeat_num = int(match.group(1)) if match.group(1) else 0
                if repeat_num == 0:
                    return get_ckpt_path_with_strategy(max_ckpt_path, src_strategy_file)
                for i in range(repeat_num, 0, -1):
                    # Replace the current repeat number in max_ckpt_path with i
                    cur_ckpt_path = re.sub(r'rank_\d+_\d+', f'rank_{cur_rank}_{i}', max_ckpt_path)
                    load_ckpt_new = get_ckpt_path_with_strategy(cur_ckpt_path, src_strategy_file)
                    if load_ckpt_new:
                        return load_ckpt_new
                # Try without repeat number (i=0 case)
                cur_ckpt_path = re.sub(r'rank_\d+_\d+', f'rank_{cur_rank}', max_ckpt_path)
                return get_ckpt_path_with_strategy(cur_ckpt_path, src_strategy_file)

        return None

    # pylint: disable=C0330
    def training_process(
            self,
            config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
            network: Optional[Union[Cell, PreTrainedModel]] = None,
            dataset: Optional[Union[BaseDataset, GeneratorDataset]] = None,
            optimizer: Optional[Optimizer] = None,
            callbacks: Optional[Union[Callback, List[Callback]]] = None,
            compute_metrics: Optional[Union[dict, set]] = None,
            **kwargs):
        """Train or Fine-tune for BaseTrainer in MindFormers."""
        self.kwargs = kwargs

        self.eval_dataset = kwargs.get('eval_dataset', None)
        if dataset:
            self.train_dataset = dataset
        if compute_metrics:
            self.compute_metrics = compute_metrics

        # preprocess config for train dataset
        config = self._train_dataset_preprocess(config)
        construct_args_key = config.train_dataset.pop("construct_args_key", None)
        is_full_config = kwargs.get("is_full_config", False)
        config = self.set_config(config, is_full_config)

        # build dataset
        logger.info(".........Build Dataset For Train..........")
        dataset = self.create_train_dataset()
        # postprocess dataset
        dataset = self._train_dataset_postprocess(dataset, config)
        logger.info("Create train dataset finish, dataset size:%d", dataset.get_dataset_size())

        append_info = None
        if not config.ckpt_use_legacy_format:
            if config.resume_training and config.load_checkpoint:
                logger.info(".............Start load resume context from common.json..................")
                common_file = os.path.join(config.load_checkpoint, 'common.json')
                if not os.path.exists(common_file):
                    raise FileNotFoundError(f"No common.json found in directory '{config.load_checkpoint}'.")
                common_info = CommonInfo.load_common(common_file)
                step_scale = common_info.global_batch_size / config.runner_config.global_batch_size
                config.runner_config.initial_step = int(common_info.step_num * step_scale)
                if config.runner_config.sink_mode:
                    config.runner_config.initial_epoch = int(common_info.epoch_num * step_scale)
                else:
                    data_size = dataset.get_dataset_size()
                    not_last_step_in_epoch = int(config.runner_config.initial_step % data_size != 0)
                    config.runner_config.initial_epoch = int(common_info.epoch_num) - not_last_step_in_epoch

                resume_dict = {
                    "step_num": config.runner_config.initial_step,
                    "epoch_num": config.runner_config.initial_epoch,
                    "loss_scale": 1
                }
                if config.runner_wrapper.scale_sense is not None:
                    if hasattr(config.runner_wrapper.scale_sense, "loss_scale_value"):
                        config.runner_wrapper.scale_sense.loss_scale_value = common_info.loss_scale
                        resume_dict["loss_scale"] = config.runner_wrapper.scale_sense.loss_scale_value
                    else:
                        config.runner_wrapper.scale_sense = common_info.loss_scale
                        resume_dict["loss_scale"] = config.runner_wrapper.scale_sense
                append_info = [resume_dict]
            else:
                config.runner_config.initial_epoch = 0
                config.runner_config.initial_step = 0
        else:
            if config.resume_training and config.load_checkpoint:
                logger.info(".............Start load resume context from checkpoint..................")
                if check_tft_valid() and not config.remove_redundancy:
                    logger.info("..............Start resume checkpoint path from strategy..............")
                    resume_ckpt_path = self.resume_ckpt_path_with_strategy(config)
                    if resume_ckpt_path is None:
                        raise ValueError(f"Try to resume from checkpoints with strategy in directory "
                                         f"'{config.load_checkpoint}' failed, please specify load_checkpoint to "
                                         f"specific checkpoint file to resume training.")
                    config.load_checkpoint = resume_ckpt_path
                load_resume_context_from_checkpoint(config, dataset)
                resume_dict = {
                    "step_num": config.runner_config.initial_step,
                    "epoch_num": config.runner_config.initial_epoch,
                }
                if config.runner_wrapper.scale_sense is not None:
                    if hasattr(config.runner_wrapper.scale_sense, 'loss_scale_value'):
                        resume_dict["loss_scale"] = config.runner_wrapper.scale_sense.loss_scale_value
                    else:
                        resume_dict["loss_scale"] = config.runner_wrapper.scale_sense
                else:
                    resume_dict["loss_scale"] = 1
                append_info = [resume_dict]
            else:
                config.runner_config.initial_epoch = 0
                config.runner_config.initial_step = 0

        # preload ckpt format file via MindIO
        if config.load_checkpoint and config.load_ckpt_format == "ckpt":
            preload_ckpt(config)

        # check if skip datasets
        if config.data_skip_steps or config.resume_training:
            if not config.ignore_data_skip:
                data_skip_steps = config.data_skip_steps if config.data_skip_steps \
                    else config.runner_config.initial_step
                if data_skip_steps > 0:
                    dataset.set_init_step(data_skip_steps)
                    logger.info("dataset skip %d steps.", data_skip_steps)
            else:
                logger.info("ignore dataset skip.")

        self.set_train_dataset(dataset)
        check_runner_config(config, dataset)

        # check rules
        check_rules(config, mode='train', network=network, dataset=dataset)

        # It is necessary to enable save strategy online before the model compilation phase,
        # if save checkpoint with megatron format.
        save_checkpoint_with_legacy_format = True
        for callback in self.config.callbacks:
            if "type" in callback and callback["type"] == "CheckpointMonitor":
                save_checkpoint_with_legacy_format = callback.get("use_legacy_format", True)
        if not save_checkpoint_with_legacy_format:
            try:
                from mindspore.parallel.strategy import enable_save_strategy_online
                enable_save_strategy_online()
            except ImportError:
                logger.warning("If you want to save checkpoint with new format,"
                               "please ensure your version of mindspore > 2.7.1, "
                               "to support 'enable_save_strategy_online'.")

        # build network
        logger.info(".........Build Net For Train..........")
        if network is None and self.network is None:
            calculate_per_token_loss = getattr(config, "calculate_per_token_loss", False)
            if config.load_checkpoint:
                network = self.create_network_without_param_init(
                    default_args={"parallel_config": config.parallel_config,
                                  "moe_config": config.moe_config,
                                  "dataset_config": config.train_dataset,
                                  "calculate_per_token_loss": calculate_per_token_loss})
            else:
                network = self.create_network(
                    default_args={"parallel_config": config.parallel_config,
                                  "moe_config": config.moe_config,
                                  "dataset_config": config.train_dataset,
                                  "calculate_per_token_loss": calculate_per_token_loss,
                                  "batch_size": self.config.runner_config.mini_batch_size})
        elif network is None and self.network is not None:
            logger.info(".........Using The Existing Network For Train:: %s", self.network.__class__.__name__)
            network = self.network

        model_params = set()
        if self.config.optimizer.type in ("PmaAdamW", "FusedPmaAdamW"):
            if hasattr(network, "get_model_parameters"):
                model_params.update(network.get_model_parameters())
            else:
                raise NotImplementedError(f"The {type(network)} has not implemented the interface: "
                                          f"get_model_parameters.")

        is_moe_model = False
        is_mtp_model = False
        if not is_legacy_model():
            is_moe_model = network.is_moe_model()
            is_mtp_model = network.is_mtp_model()

        config.load_checkpoint = get_load_path_after_hf_convert(config, network)
        self._check_training_network_no_use_past(network)

        eval_network = None
        if network is not None:
            eval_network = network
            if construct_args_key is not None:
                eval_network = self.warp_data_order_with_tool_cells(network, construct_args_key)
            # warp network for training
            network = self.wrap_network_with_tool_cells(eval_network)
            eval_network = self.wrap_eval_network_with_tool_cells(eval_network)
            self.set_network(network, is_train=True)

        self.count_parameters()

        # build optimizer
        logger.info(".........Build Optimizer For Train..........")
        if optimizer is None:
            if config.load_checkpoint:
                optimizer = self.create_optimizer_scheduler_without_param_init(network, model_params)
            else:
                optimizer = self.create_optimizer_scheduler(network, model_params)

        # build model wrapper
        logger.info(".........Build Running Wrapper From Config For Train..........")
        wrapper = self.create_model_wrapper(network, optimizer)

        # initial tensorboard
        if (hasattr(config, 'tensorboard') and hasattr(config.tensorboard, 'tensorboard_dir') and
                config.tensorboard.tensorboard_dir):
            rank_id = get_real_rank()
            if isinstance(config.tensorboard.tensorboard_dir, str):
                logger.info(f"Set tensorboard path to '{config.tensorboard.tensorboard_dir}'")
                config.tensorboard.tensorboard_dir = os.path.join(config.tensorboard.tensorboard_dir, f"rank_{rank_id}")
                _set_tensorboard_writer(config.tensorboard)
                update_tensorboard_args(config.tensorboard)
                write_args_to_tensorboard(config)
            else:
                logger.warning("Since tensorboard_dir is not a string, tensorboard configuration will not take effect.")

        # build callback
        logger.info(".........Build Callbacks For Train..........")
        default_callbacks = []
        if self.config.profile:
            default_callbacks.append(self.config.profile_cb)
        if isinstance(self.config.runner_config.stop_step, int) and self.config.runner_config.stop_step > 0:
            stop_step_dict = OrderedDict()
            stop_step_dict['type'] = "TrainCallBack"
            stop_step_dict['stop_step'] = self.config.runner_config.stop_step
        ckpt_config = None
        hidden_size = config.model.model_config.hidden_size
        vocab_size = config.model.model_config.vocab_size
        embedding_size = None
        use_checkpoint_health_monitor = getattr(config, "use_checkpoint_health_monitor", False)
        if use_checkpoint_health_monitor:
            if hidden_size is None:
                raise ValueError("You should set the hidden_size while use the checkpoint health monitor function.")
            if vocab_size is None:
                raise ValueError("You should set the vocab_size while use the checkpoint health monitor function.")
            embedding_size = hidden_size * vocab_size

        for callback in self.config.callbacks:
            default_args = None
            if "type" in callback and callback["type"] == "MFLossMonitor":
                default_args = {
                    "learning_rate": optimizer.learning_rate if optimizer else wrapper.optimizer.learning_rate,
                    "origin_epochs": config.runner_config.origin_epochs,
                    "dataset_size": config.data_size,
                    "micro_batch_interleave_num": config.micro_batch_interleave_num,
                    "micro_batch_num": config.parallel_config.micro_batch_num,
                    "initial_epoch": config.runner_config.initial_epoch,
                    "initial_step": config.runner_config.initial_step,
                    "global_batch_size": self.global_batch_size,
                    "gradient_accumulation_steps": self.config.runner_config.gradient_accumulation_steps,
                    "calculate_per_token_loss": getattr(config, "calculate_per_token_loss", False),
                    "print_separate_loss": getattr(config, "print_separate_loss", True),
                    "is_moe_model": is_moe_model,
                    "is_mtp_model": is_mtp_model
                }
            if "type" in callback and callback["type"] == "TrainingStateMonitor":
                default_args = {
                    "origin_epochs": config.runner_config.origin_epochs,
                    "dataset_size": config.data_size,
                    "initial_epoch": config.runner_config.initial_epoch,
                    "initial_step": config.runner_config.initial_step,
                    "global_batch_size": self.global_batch_size,
                    "check_for_nan_in_loss_and_grad": getattr(config, "check_for_nan_in_loss_and_grad", False),
                    "use_skip_data_by_global_norm": getattr(config, "use_skip_data_by_global_norm", False),
                    "embedding_size": embedding_size,
                    "use_local_norm": getattr(config.runner_wrapper, "local_norm", False)
                }
            elif "type" in callback and callback["type"] == "CheckpointMonitor":
                logger.info("Recommend using weights in the safetensors format.")
                embedding_local_norm_threshold = callback.get("embedding_local_norm_threshold", 1.0)
                # load params into net
                default_args = {"append_info": append_info,
                                "global_batch_size": self.global_batch_size,
                                "remove_redundancy": callback.get("remove_redundancy", False),
                                "checkpoint_format": callback.get("checkpoint_format", "ckpt"),
                                "embedding_size": embedding_size,
                                "embedding_local_norm_threshold": embedding_local_norm_threshold,
                                "use_checkpoint_health_monitor": use_checkpoint_health_monitor,
                                "health_ckpts_record_dir": config.output_dir
                                }
                if not config.get("use_legacy", True) and default_args.get("checkpoint_format") == "ckpt":
                    logger.warning(
                        "When use_legacy=False, it's not supported to save 'ckpt' files and switch to 'safetensors'.")
                    default_args["checkpoint_format"] = "safetensors"
                if default_args.get("remove_redundancy") and default_args.get("checkpoint_format") == "ckpt":
                    raise ValueError("The format of checkpoint is ckpt which is not support remove redundancy.")
                ckpt_config = [callback, default_args]
            elif "type" in callback and callback["type"] == "TrainFaultTolerance":
                continue
            elif "type" in callback and callback["type"] == "StressDetectCallBack":
                default_args = {
                    "dataset_size": config.data_size
                }
            elif "type" in callback and callback["type"] == "SDCMonitor":
                default_args = {"initial_step": config.runner_config.initial_step}
            default_callbacks.append(build_callback(callback, default_args=default_args))

        if check_tft_valid() or check_tre_valid() or check_tsp_valid():
            if ckpt_config is None:
                raise ValueError("You must set CheckpointMonitor callback for TFT training!")
            ckpt_config[1]["async_save"] = False
            ckpt_cb_obj = build_callback(ckpt_config[0], default_args=ckpt_config[1])

            # pylint:disable=W0640,W0212
            def ckpt_save_func(cb_params, append_dict, prefix=None):
                ckpt_cb_obj._append_dict.update(append_dict)
                ckpt_cb_obj.save_network_params = False
                ckpt_cb_obj.save_trainable_params = False
                if prefix is not None:
                    ckpt_cb_obj._prefix = prefix + "_" + ckpt_cb_obj._prefix
                ckpt_cb_obj._save_ckpt(cb_params, True)

            def ckpt_load_func():
                logger.info('Begin to load ckpt')
                ckpt_file = get_resume_checkpoint(f'{self.config.output_dir}/checkpoint', True,
                                                  self.config.load_ckpt_format)
                remove_redundancy = False if config.remove_redundancy is None else config.remove_redundancy
                param_dict = load_resume_checkpoint(ckpt_file, remove_redundancy, self.config.load_ckpt_format)
                logger.info(f'End to load ckpt, param_dict.size={len(param_dict)}')
                return param_dict, remove_redundancy

            default_args = {
                "ckpt_save_fn": ckpt_save_func,
                "initial_epoch": config.runner_config.initial_epoch,
                "initial_step": config.runner_config.initial_step,
                "ckpt_load_fn": ckpt_load_func
            }
            default_callbacks.append(build_callback({"type": "TrainFaultTolerance"}, default_args=default_args))

        if callbacks is not None:
            if isinstance(callbacks, list):
                default_callbacks.extend(callbacks)
            if isinstance(callbacks, Callback):
                default_callbacks.append(callbacks)
        callbacks = self.sort_callbacks(default_callbacks)

        # define compute metrics for evaluate in training
        if config.do_eval:
            compute_metrics = self.create_metrics()

        # define Model and begin training
        logger.info(".........Starting Init Train Model..........")
        if wrapper is not None:
            if config.train_dataset.dynamic_batch:
                if self.config.parallel_config.seq_split_num > 1:
                    raise ValueError("Cannot apply sequence pipe in dynamic shape.")
                from mindspore import Symbol
                divisor = config.train_dataset.divisor if config.train_dataset.divisor else 2
                remainder = config.train_dataset.remainder if config.train_dataset.remainder else 1
                s = Symbol(divisor=divisor, remainder=remainder)
                dyn_inputs = []
                for _ in config.train_dataset.input_columns:
                    dyn_inputs.append(Tensor(shape=(config.train_dataset.batch_size, s), dtype=ms.int32))
                wrapper.set_inputs(*dyn_inputs)
            model = Model(wrapper, metrics=compute_metrics, eval_network=eval_network)
        else:
            model = Model(network, optimizer=optimizer, metrics=compute_metrics, eval_network=eval_network)

        # resume checkpoint
        if not config.ckpt_use_legacy_format and config.load_checkpoint:
            if config.use_parallel:
                compile_model(model, dataset, mode=config.context.mode, sink_mode=config.runner_config.sink_mode,
                              epoch=config.runner_config.epochs, sink_size=config.runner_config.sink_size)

            if config.resume_training:
                logger.info(".............Start resume training from checkpoint..................")
                global_step = common_info.global_step
                if common_info.global_batch_size != self.global_batch_size:
                    global_step = \
                        int(common_info.global_step * (common_info.global_batch_size / self.global_batch_size))
                    logger.info(
                        f"Scaled global step: {common_info.global_step} → {global_step} "
                        f"(batch size changed from {common_info.global_batch_size} to {self.global_batch_size})"
                    )
                load_checkpoint(
                    checkpoint=config.load_checkpoint, network=network, optimizer=optimizer, global_step=global_step,
                    balanced_load=config.balanced_load
                )
            else:
                load_checkpoint(checkpoint=config.load_checkpoint, network=network, balanced_load=config.balanced_load)
        elif (config.load_checkpoint or config.only_save_strategy) and not check_is_reboot_node():
            if config.resume_training:
                logger.info(".............Start resume training from checkpoint..................")
                transform_and_load_checkpoint(config, model, network, dataset, optimizer=optimizer)
            else:
                transform_and_load_checkpoint(config, model, network, dataset)

        # build evaluate in training
        if config.do_eval:
            logger.info(".........Build Evaluate in Training Callback..........")
            eval_dataset = self.create_eval_dataset()
            logger.info("Create evaluate dataset finish, dataset size:%d", eval_dataset.get_dataset_size())

            eval_callback = EvalCallBack(
                partial(
                    self._evaluate_in_training,
                    model=model,
                    eval_dataset=eval_dataset,
                ),
                step_interval=config.eval_step_interval if config.eval_step_interval else 100,
                epoch_interval=config.eval_epoch_interval if config.eval_epoch_interval else -1,
            )
            callbacks.append(eval_callback)

        if config.moe_config.enable_cold_hot_expert:
            save_checkpoint_steps = -1
            for callback in config.callbacks:
                if callback['type'] == 'CheckpointMonitor':
                    save_checkpoint_steps = callback['save_checkpoint_steps']
            cold_hot_monitor = ColdHotExpertMonitor(
                moe_config=config.moe_config,
                hidden_size=config.model.model_config.hidden_size,
                ffn_hidden_size=config.model.model_config.ffn_hidden_size,
                expert_parallel=config.parallel_config.expert_parallel,
                model_parallel=config.parallel_config.model_parallel,
                save_checkpoint_steps=save_checkpoint_steps)
            # ColdHotExpertMonitor needs to be placed before CheckpointMonitor
            callbacks.insert(1, cold_hot_monitor)

        if get_real_rank() % 8 == 0:
            pprint(config)
        logger.info(".........Model Compiling, Please Wait a Moment...........")
        if self.network_delay_inited:
            logger.info(".........train network delay initialize..........")
            network.init_parameters_data()
        if self.optimizer_delay_inited:
            logger.info(".........optimizer delay initialize..........")
            optimizer.init_parameters_data()
        logger.info(".........Starting Training Model..........")
        model.train(config.runner_config.epochs, dataset,
                    callbacks=callbacks,
                    dataset_sink_mode=config.runner_config.sink_mode,
                    sink_size=config.runner_config.sink_size,
                    initial_epoch=config.runner_config.initial_epoch)

        # close tensorboard
        _unset_tensorboard_writer()
        logger.info(".........Training Over!.............")

    def evaluate_process(
            self,
            config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
            network: Optional[Union[Cell, PreTrainedModel]] = None,
            dataset: Optional[Union[BaseDataset, GeneratorDataset]] = None,
            callbacks: Optional[Union[Callback, List[Callback]]] = None,
            compute_metrics: Optional[Union[dict, set]] = None,
            **kwargs):
        """Evaluate for BaseTrainer in MindFormers."""
        self.eval_dataset = dataset if dataset else self.eval_dataset
        metric_name = kwargs.get("metric_name")
        is_full_config = kwargs.get("is_full_config", False)
        config = self.set_config(config, is_full_config)
        construct_args_key = config.eval_dataset.pop("construct_args_key", None)

        # build dataset
        logger.info(".........Build Dataset For Evaluate..........")
        dataset = self.create_eval_dataset()
        self.set_eval_dataset(dataset)
        logger.info("Create evaluate dataset finish, dataset size:%d", dataset.get_dataset_size())

        # check rules
        check_rules(config, mode='eval', network=network, dataset=dataset, task=self.task)

        # build network
        if network is None and self.network is None:
            network = self.create_network_without_param_init(
                default_args={"parallel_config": config.parallel_config,
                              "moe_config": config.moe_config})
        elif network is None and self.network is not None:
            logger.info(".........Using The Existing Network For Evaluate: %s", self.network.__class__.__name__)
            network = self.network
        config.load_checkpoint = get_load_path_after_hf_convert(config, network)
        if network is not None and construct_args_key is not None:
            network = self.warp_data_order_with_tool_cells(network, construct_args_key)

        self.set_network(network, is_train=False)
        self.count_parameters()

        # build metric
        logger.info(".........Build Compute Metrics For Evaluate..........")
        if compute_metrics is None:
            compute_metrics = self.create_metrics(metric_name=metric_name)

        # build callback
        logger.info(".........Build Callbacks For Evaluate..........")
        default_callbacks = self.create_eval_callbacks()
        if callbacks is not None:
            if isinstance(callbacks, list):
                default_callbacks.extend(callbacks)
            if isinstance(callbacks, Callback):
                default_callbacks.append(callbacks)
        callbacks = default_callbacks

        logger.info(".........Starting Init Evaluate Model..........")
        model = Model(network, metrics=compute_metrics, eval_network=network)

        if config.load_checkpoint or config.only_save_strategy:
            logger.info(".............Start load checkpoint for eval..................")
            transform_and_load_checkpoint(config, model, network, next(dataset.create_tuple_iterator()), do_eval=True)

        if self.network_delay_inited:
            logger.info(".........eval network delay initialize..........")
            network.init_parameters_data()

        logger.info(".........Starting Evaluate Model..........")
        if get_real_rank() % 8 == 0:
            pprint(config)
        output = model.eval(dataset,
                            callbacks=callbacks,
                            dataset_sink_mode=config.runner_config.sink_mode)
        logger.info('%s = %s', metric_name, str(output))
        logger.info(".........Evaluate Over!.............")

    def predict_process(self,
                        config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
                        input_data: Optional[Union[GeneratorDataset, Tensor, np.ndarray, Image, str, list]] = None,
                        task: str = None,
                        network: Optional[Union[Cell, PreTrainedModel]] = None,
                        tokenizer: Optional[PreTrainedTokenizerBase] = None,
                        image_processor: Optional[BaseImageProcessor] = None,
                        audio_processor: Optional[BaseImageProcessor] = None, **kwargs):
        """Predict for BaseTrainer in MindFormers."""
        if not self.pipeline_task:
            is_full_config = kwargs.get("is_full_config", False)
            config = self.set_config(config, is_full_config)

            # check rules
            check_rules(config, mode='predict', network=network, task=self.task)

            if ms.context.get_auto_parallel_context('parallel_mode') in \
                    ['semi_auto_parallel', 'auto_parallel', 'hybrid_parallel']:
                if task not in ["translation", "text_generation", "multi_modal_to_text_generation"]:
                    raise ValueError("Currently distributed predict only support translation and text_generation. "
                                     "Process exit!")
                if config.parallel_config.pipeline_stage > 1:
                    raise ValueError("Currently distributed predict dose not support pipeline parallel. "
                                     "Process exit!")

            # build network
            if network is None:
                if config.load_checkpoint:
                    network = self.create_network_without_param_init(
                        default_args={"parallel_config": config.parallel_config,
                                      "moe_config": config.moe_config})
                else:
                    logger.info('Weights are not loaded. Hence Delay Initialization is disabled.')
                    network = self.create_network(
                        default_args={"parallel_config": config.parallel_config,
                                      "moe_config": config.moe_config})

            self.set_network(network, is_train=False)
            self.count_parameters()
            config.load_checkpoint = get_load_path_after_hf_convert(config, network)

            processor_config = config.get("processor", {})
            tokenizer_config = processor_config.get("tokenizer", None)
            image_processor_config = processor_config.get("image_processor", None)
            audio_processor_config = processor_config.get("audio_processor", None)

            if tokenizer is None:
                pretrained_model_dir = config.get("pretrained_model_dir", None)
                trust_remote_code = config.get("trust_remote_code", False)
                tokenizer = build_tokenizer(tokenizer_config,
                                            use_legacy=config.get("use_legacy", True),
                                            pretrained_model_dir=pretrained_model_dir,
                                            trust_remote_code=trust_remote_code)

            if image_processor is None and image_processor_config:
                image_processor = build_processor(image_processor_config)

            if audio_processor is None and audio_processor_config:
                audio_processor = build_processor(audio_processor_config)

            model = Model(network)

            if not config.use_legacy and config.load_checkpoint:
                if self.config.load_ckpt_format == 'safetensors':
                    network.load_weights(config.load_checkpoint)
                else:
                    raise ValueError('The process of MCore does not support the weights of ckpt.')
            else:
                if config.load_checkpoint or config.only_save_strategy:
                    if ms.context.get_auto_parallel_context('parallel_mode') in \
                            ['semi_auto_parallel', 'auto_parallel', 'hybrid_parallel']:
                        if network.config:
                            batch_size = network.config.batch_size
                            seq_length = network.config.seq_length
                        else:
                            batch_size = config.model.model_config.batch_size
                            seq_length = config.model.model_config.seq_length
                        input_ids = np.ones(shape=tuple([batch_size, seq_length]))
                        infer_data = network.prepare_inputs_for_predict_layout(input_ids)
                        transform_and_load_checkpoint(config, model, network, infer_data, do_predict=True)
                    else:
                        transform_and_load_checkpoint(config, model, network, None, do_predict=True)

            if self.network_delay_inited:
                logger.info(".........predict network delay initialize..........")
                network.init_parameters_data()

            self.pipeline_task = pipeline(
                task=task,
                model=network,
                tokenizer=tokenizer,
                image_processor=image_processor,
                audio_processor=audio_processor,
                **kwargs
            )

        top_k = kwargs.pop("top_k", None)
        if top_k is None:
            if config.top_k is not None:
                top_k = config.top_k
            elif config.model and config.model.model_config.top_k:
                top_k = config.model.model_config.top_k
            else:
                top_k = 1

        save_file = kwargs.pop("save_file", None)
        if save_file is None:
            if config.save_file is not None:
                save_file = config.save_file
            else:
                save_file = f"{task}_result.txt"

        if get_real_rank() % 8 == 0:
            pprint(config)
        output_results = self.pipeline_task(input_data, top_k=top_k)

        output_info = []

        for one_output in output_results:
            if isinstance(one_output, dict) and "info" in one_output:
                output_info.append(one_output["info"])
            else:
                output_info.append(one_output)
        flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        with os.fdopen(os.open(save_file, flags_, FILE_PERMISSION), 'w') as file:
            file.write(str(output_info))

        logger.info("output result is: %s", str(output_info))
        logger.info("output result is saved at: %s", save_file)
        logger.info(".........Predict Over!.............")
        return output_results

    def _evaluate_in_training(self, model: Model, eval_dataset: BaseDataset):
        origin_phase = model.eval_network.phase
        model.eval_network.set_train(False)
        output = model.eval(
            eval_dataset, dataset_sink_mode=self.config.runner_config.sink_mode
        )
        model.eval_network.set_train(origin_phase)
        return output

    @staticmethod
    def sort_callbacks(default_callbacks):
        """Sort by execution order."""
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
