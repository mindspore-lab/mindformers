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
from typing import List, Optional, Union

import yaml
import numpy as np
from PIL.Image import Image

from mindspore import Tensor
from mindspore.common import set_seed
from mindspore.nn import TrainOneStepCell, Optimizer
from mindspore.train import Callback
from mindspore import load_param_into_net, load_checkpoint

from mindformers.mindformer_book import MindFormerBook
from mindformers.tools.register import MindFormerConfig, MindFormerRegister
from mindformers.models import build_model, build_tokenizer, build_feature_extractor, \
    BaseModel, BaseFeatureExtractor, BaseTokenizer
from mindformers.dataset import build_dataset, build_dataset_loader, \
    check_dataset_config, BaseDataset
from mindformers.pipeline import pipeline
from mindformers.wrapper import build_wrapper
from mindformers.common.optim import build_optim
from mindformers.common.lr import build_lr
from mindformers.common.callback import build_callback
from mindformers.common.parallel_config import build_parallel_config
from mindformers.tools.cloud_adapter import CFTS
from mindformers.tools.logger import logger
from mindformers.tools.utils import count_params
from mindformers.tools.image_tools import load_image
from .build_trainer import build_trainer
from .config_args import ConfigArguments
from .utils import check_train_data_loader_type, check_eval_data_loader_type, \
    check_optimizer_and_lr_type, check_wrapper_config


__all__ = ['Trainer']


SUPPORT_TASKS = MindFormerBook().get_trainer_support_task_list()
SUPPORT_MODEL_NAMES = MindFormerBook().get_model_name_support_list()
SUPPORT_PIPELINE_INPUT_DATA = MindFormerBook().get_pipeline_support_input_data_list()
DEFAULT_CHECKPOINT_DIR = 'checkpoint'
DEFAULT_CONFIG_DIR = 'configs'


class Trainer:
    """Trainer API."""
    def __init__(self,
                 config: Optional[Union[str, dict, ConfigArguments]] = None,
                 task_name: Optional[str] = 'general',
                 model: Optional[Union[str, BaseModel]] = None,
                 train_dataset: Optional[Union[str, BaseDataset]] = None,
                 eval_dataset: Optional[Union[str, BaseDataset]] = None,
                 tokenizer: Optional[BaseTokenizer] = None,
                 feature_extractor: Optional[BaseFeatureExtractor] = None,
                 optimizers: Optional[Optimizer] = None,
                 wrapper: Optional[TrainOneStepCell] = None,
                 callbacks: Optional[Union[Callback, List[Callback]]] = None,
                 compute_metrics: Optional[Union[dict, set]] = None,
                 **kwargs):

        self.task_name = task_name
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.optimizers = optimizers
        self.wrapper = wrapper
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.callbacks = callbacks
        self.compute_metrics = compute_metrics
        self.kwargs = kwargs

        if wrapper is not None:
            if model is not None:
                logger.warning(
                    'wrapper has existed, input model invalid, it should be include in wrapper.')
            if optimizers is not None:
                logger.warning(
                    'wrapper has existed, input optimizers invalid, it should be include in wrapper.')

        assert task_name in SUPPORT_TASKS.keys(), \
            f"task name must be in {SUPPORT_TASKS.keys()}, but get {task_name}."
        if isinstance(model, str):
            assert model in SUPPORT_MODEL_NAMES, \
                f"model must be in {SUPPORT_MODEL_NAMES} when model's type is string, but get {model}."
            self.model_name = model
            self.model = None
        else:
            self.model_name = "common"

        task_config = MindFormerConfig(SUPPORT_TASKS.get(self.task_name).get(self.model_name))

        if self.model_name == "common":
            if self.model is not None:
                task_config.trainer.model_name = self.model.__class__.__name__
            if self.wrapper is not None:
                task_config.trainer.model_name = self.wrapper.network.__class__.__name__

        if config is None:
            self.config = task_config
        else:
            if isinstance(config, dict):
                task_config.merge_from_dict(config)
            elif isinstance(config, str):
                assert os.path.realpath(config) and os.path.exists(config), \
                    f"config path must be exist, but get {config}."
                assert config.endswith(('.yaml', '.yml')), \
                    f"config file must be end with .yaml or .yml, but get {config}"
                task_config = MindFormerConfig(config)
            elif isinstance(config, ConfigArguments):
                if hasattr(config, 'train_dataset'):
                    check_train_data_loader_type(config, task_config)
                if hasattr(config, 'eval_dataset'):
                    check_eval_data_loader_type(config, task_config)
                if hasattr(config, 'optimizer'):
                    check_optimizer_and_lr_type(config, task_config)
                if hasattr(config, 'runner_wrapper'):
                    check_wrapper_config(config, task_config)
                task_config.merge_from_dict(config.__dict__)

            self.config = task_config

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
        self.rank_id = int(os.getenv("RANK_ID", "0"))
        self.context_config = self.config.context
        self.parallel_config = self.config.parallel
        build_parallel_config(self.config)

        # set cloud file transform for ModelArts.
        cfts = CFTS(**self.config.aicc_config)
        MindFormerRegister.register_cls(cfts, alias='cfts')

        # set seed
        set_seed(self.config.seed)
        np.random.seed(self.config.seed)

        # set output directory
        os.environ.setdefault("LOCAL_DEFAULT_PATH", self.config.output_dir)

        # self.save_config_to_yaml()
        # logger.info("save running config success of {}.".format(task_config.trainer.model_name.lower()))

    def train(self, resume_from_checkpoint: Optional[Union[str, bool]] = None,
              initial_epoch: int = 0, do_eval: bool = False, **kwargs):
        """train."""
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        if self.train_dataset is None:
            self.train_dataset = build_dataset(self.config.train_dataset_task)

        if do_eval:
            if self.eval_dataset is None:
                self.eval_dataset = build_dataset(self.config.eval_dataset_task)

        if self.model is None and self.wrapper is None:
            self.model = build_model(self.config.model)

        if self.optimizers is None and self.wrapper is None:
            self.optimizers = self.create_optimizer_and_scheduler()

        if self.wrapper is None:
            self.wrapper = self.create_train_one_step_wrapper()

        if self.callbacks is None:
            self.callbacks = self.create_callbacks()

        self.load_checkpoint(resume_from_checkpoint)
        if initial_epoch != 0:
            self.config.runner_config.initial_epoch = initial_epoch

        trainer = build_trainer(self.config.trainer)
        trainer.train(
            config=self.config, network=self.model,
            dataset=self.train_dataset, optimizer=self.optimizers,
            eval_dataset=self.eval_dataset if do_eval else None,
            wrapper=self.wrapper,
            callbacks=self.callbacks, **kwargs)

    def evaluate(self, eval_checkpoint: Optional[Union[str, bool]] = None, **kwargs):
        """eval."""
        if eval_checkpoint is False:
            eval_checkpoint = None

        if self.eval_dataset is None:
            self.eval_dataset = build_dataset(self.config.eval_dataset_task)

        if self.model is None:
            self.model = build_model(self.config.model)

        if self.callbacks is None:
            self.callbacks = self.create_callbacks()

        filter_prefix = ["adam_v", "adam_m", "epoch_num", "step_num", "global_step"]
        self.load_checkpoint(eval_checkpoint, filter_prefix=filter_prefix, is_train=False)

        trainer = build_trainer(self.config.trainer)
        trainer.evaluate(
            config=self.config, network=self.model,
            dataset=self.eval_dataset, callbacks=self.callbacks, **kwargs)

    def predict(self,
                predict_checkpoint: Optional[Union[str, bool]] = None,
                input_data: Optional[Union[Tensor, np.ndarray, Image, str, list]] = None, **kwargs):
        """predict."""
        if predict_checkpoint is False:
            predict_checkpoint = None

        if input_data is None:
            input_data = load_image(SUPPORT_PIPELINE_INPUT_DATA.get(self.task_name))
        assert isinstance(input_data, (Tensor, np.ndarray, Image, str, list)), \
            "Input data's type must be one of [str, ms.Tensor, np.ndarray, PIL.Image.Image]"

        if self.model is None:
            self.model = build_model(self.config.model)

        if self.tokenizer is None:
            self.tokenizer = build_tokenizer(self.config.processor.tokenizer)

        if self.feature_extractor is None:
            self.feature_extractor = build_feature_extractor(self.config.processor.feature_extractor)

        filter_prefix = ["adam_v", "adam_m", "epoch_num", "step_num", "global_step"]
        self.load_checkpoint(predict_checkpoint, filter_prefix=filter_prefix, is_train=False)

        predictor = pipeline(
            self.task_name, model=self.model,
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer, **kwargs)
        return predictor(input_data)

    def create_optimizer_and_scheduler(self):
        """create_optimizer_and_scheduler."""
        lr_schedule = self.create_scheduler()
        params = self.model.trainable_params()
        return self.create_optimizer(lr_schedule, params)

    def create_scheduler(self):
        """create_scheduler."""
        return build_lr(self.config.lr_schedule)

    def create_optimizer(self, lr_schedule, params):
        """create_optimizer."""
        if lr_schedule is not None:
            return build_optim(self.config.optimizer, default_args={"params": params,
                                                                    "learning_rate": lr_schedule})
        assert self.config.optimizer.learning_rate, "learning_rate must be input"
        return build_optim(self.config.optimizer, default_args={"params": params})

    def create_train_one_step_wrapper(self):
        """create_train_one_step_wrapper."""
        if self.model is not None and self.optimizers is not None:
            return build_wrapper(
                self.config.wrapper,
                default_args={"network": self.model, "optimizer": self.optimizers})
        return None

    def create_callbacks(self):
        """create_callbacks."""
        return build_callback(self.config.callbacks)

    def set_parallel_config(
            self, data_parallel=1, model_parallel=1, expert_parallel=1, pipeline_stage=1,
            micro_batch_num=1, optimizer_shard=False, gradient_aggregation_group=4, vocab_emb_dp=True):
        """set_parallel_config."""
        self.config.parallel_config.data_parallel = data_parallel
        self.config.parallel_config.model_parallel = model_parallel
        self.config.parallel_config.expert_parallel = expert_parallel
        self.config.parallel_config.pipeline_stage = pipeline_stage
        self.config.parallel_config.optimizer_shard = optimizer_shard
        self.config.parallel_config.micro_batch_num = micro_batch_num
        self.config.parallel_config.vocab_emb_dp = vocab_emb_dp
        self.config.parallel_config.gradient_aggregation_group = gradient_aggregation_group

    def set_recompute_config(self, recompute=False, parallel_optimizer_comm_recompute=False,
                             mp_comm_recompute=True, recompute_slice_activation=False):
        """set_recompute_config."""
        self.config.recompute_config.recompute = recompute
        self.config.recompute_config.parallel_optimizer_comm_recompute = parallel_optimizer_comm_recompute
        self.config.recompute_config.mp_comm_recompute = mp_comm_recompute
        self.config.recompute_config.recompute_slice_activation = recompute_slice_activation

    def set_moe_config(self, expert_num=1, capacity_factor=1.1, aux_loss_factor=0.05, num_experts_chosen=1):
        """set_moe_config."""
        self.config.moe_config.expert_num = expert_num
        self.config.moe_config.capacity_factor = capacity_factor
        self.config.moe_config.aux_loss_factor = aux_loss_factor
        self.config.moe_config.num_experts_chosen = num_experts_chosen

    def get_train_dataloader(self):
        """get_train_dataloader."""
        return build_dataset_loader(self.config.train_dataset.data_loader)

    def get_eval_dataloader(self):
        """get_eval_dataloader."""
        return build_dataset_loader(self.config.eval_dataset.data_loader)

    def compute_loss(self):
        """compute_loss."""

    def count_parameter(self):
        """count_parameter."""
        logger.info("%s parameter is: %s M",
                    self.config.trainer.model_name, str(count_params(self.model)))

    def load_checkpoint(self, model_checkpoint, filter_prefix=None, is_train=True):
        """Load Checkpoint."""
        if model_checkpoint is not None:
            if isinstance(model_checkpoint, bool):
                last_checkpoint = load_checkpoint(
                    self.get_last_checkpoint(), filter_prefix=filter_prefix)
                not_load_net_params = load_param_into_net(self.model, last_checkpoint)
                logger.info("not_load_net_params: %s", str(not_load_net_params))
                if is_train:
                    not_load_optim_params = load_param_into_net(self.optimizers, last_checkpoint)
                    logger.info("not_load_optim_params: %s", str(not_load_optim_params))
            elif isinstance(model_checkpoint, str):
                assert os.path.realpath(model_checkpoint) and os.path.exists(model_checkpoint), \
                    f"predict checkpoint must be correct and exist path, but get {model_checkpoint}"
                checkpoint = load_checkpoint(model_checkpoint, filter_prefix=filter_prefix)
                not_load_net_params = load_param_into_net(self.model, checkpoint)
                logger.info("not_load_net_params: %s", str(not_load_net_params))
                if is_train:
                    not_load_optim_params = load_param_into_net(self.optimizers, checkpoint)
                    logger.info("not_load_optim_params: %s", str(not_load_optim_params))
            else:
                raise KeyError("resume_from_checkpoint input type should be in [string(checkpoint path), bool],"
                               f"but get {model_checkpoint}")

    def get_last_checkpoint(self):
        """get last checkpoint for resuming."""
        output_folder = self.config.output_dir
        checkpoint_dir = os.path.join(
            output_folder, 'rank_{}'.format(self.rank_id), DEFAULT_CHECKPOINT_DIR)
        output_checkpoint_path = [
            checkpoint for checkpoint in os.listdir(checkpoint_dir)
            if checkpoint.endswith('.ckpt')
        ]
        if not output_checkpoint_path:
            return None
        output_checkpoint_path = sorted(output_checkpoint_path,
                                        key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
        return os.path.join(checkpoint_dir, output_checkpoint_path[-1])

    def save_config_to_yaml(self):
        """save now config file to yaml file."""
        config_dir = os.path.join(
            self.config.output_dir, DEFAULT_CONFIG_DIR, self.config.trainer.model_name.lower())
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        config_path = os.path.join(config_dir, self.config.trainer.model_name.lower() + '.yaml')
        with open(config_path, 'w') as file_pointer:
            file_pointer.write(yaml.dump(self.config))
