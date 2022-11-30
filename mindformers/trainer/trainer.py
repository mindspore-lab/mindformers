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
from typing import Callable, List, Optional, Union

from mindformers.mindformer_book import MindFormerBook
from mindformers.tools.register import MindFormerConfig, MindFormerRegister
from mindformers.models import build_model, build_tokenizer
from mindformers.dataset import build_dataset
from mindformers.trainer import build_trainer
from mindformers.common.optim import build_optim
from mindformers.common.lr import build_lr
from mindformers.common.callback import build_callback
from mindformers.common.context import init_context
from mindformers.common.parallel_config import build_parallel_config
from mindformers.tools.cloud_adapter import CFTS


SUPPORT_TASKS = MindFormerBook().get_trainer_support_task_list()


class Trainer:
    """Trainer API."""
    def __init__(self,
                 config: dict = None,
                 task_name: str = None,
                 model: Optional[Union[str, Callable]] = None,
                 train_dataset: Callable = None,
                 eval_dataset: Callable = None,
                 optimizers: Callable = None,
                 tokenizer: Callable = None,
                 callbacks: List[Callable] = None,
                 compute_metrics: str = None, **kwargs):
        self.task_name = task_name
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.optimizers = optimizers
        self.tokenizer = tokenizer
        self.callbacks = callbacks
        self.compute_metrics = compute_metrics
        self.kwargs = kwargs

        if isinstance(model, str):
            # check model name
            self.model_name = model
            self.model = None
        else:
            self.model_name = "common"

        task_config = MindFormerConfig(SUPPORT_TASKS.get(self.task_name).get(self.model_name))

        if self.model_name == "common":
            task_config.trainer.model_name = "Your Self-Define Model"

        if config is None:
            self.config = task_config
        else:
            task_config.merge_from_dict(config)
            self.config = task_config

        init_context(seed=self.config.seed, use_parallel=self.config.use_parallel,
                     context_config=self.config.context, parallel_config=self.config.parallel)

        self.context_config = self.config.context
        self.parallel_config = self.config.parallel

        build_parallel_config(self.config)

        cfts = CFTS(**self.config.aicc_config)
        MindFormerRegister.register_cls(cfts, alias='cfts')

    def train(self, resume_from_checkpoint: Optional[Union[str, bool]] = None, **kwargs):
        """train."""
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        if self.train_dataset is None:
            self.train_dataset = build_dataset(self.config.train_dataset_task)

        if self.model is None:
            self.model = build_model(self.config.model)

        if self.optimizers is None:
            self.optimizers = self.create_optimizer_and_scheduler()

        if resume_from_checkpoint:
            # 待补充
            pass

        if self.tokenizer is None:
            self.tokenizer = build_tokenizer(self.config.tokenizer)

        if self.callbacks is None:
            self.callbacks = self.create_callbacks()

        trainer = build_trainer(self.config.trainer)
        trainer.train(
            config=self.config, network=self.model,
            dataset=self.train_dataset, optimizer=self.optimizers,
            tokenizer=self.tokenizer, callbacks=self.callbacks, **kwargs)

    def evaluate(self, eval_checkpoint: str = None, **kwargs):
        """eval."""
        if self.eval_dataset is None:
            self.eval_dataset = build_dataset(self.config.eval_dataset_task)

        if self.model is None:
            self.model = build_model(self.config.model)

        if eval_checkpoint:
            # 待补充
            pass

        if self.tokenizer is None:
            self.tokenizer = build_tokenizer(self.config.tokenizer)  # 待补充

        if self.callbacks is None:
            self.callbacks = self.create_callbacks()

        trainer = build_trainer(self.config.trainer)
        trainer.evaluate(
            config=self.config, network=self.model,
            dataset=self.eval_dataset, tokenizer=self.tokenizer,
            callbacks=self.callbacks, **kwargs)

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
        return build_optim(self.config.optimizer, default_args={"params": params,
                                                                "learning_rate": lr_schedule})

    def create_callbacks(self):
        """create_callbacks."""
        return build_callback(self.config.callbacks)

    def set_context(self, seed=0, use_parallel=False, device_id=0, device_target="Ascend", parallel_model=0):
        """set_context."""
        self.context_config.device_id = device_id
        self.context_config.device_target = device_target
        self.parallel_config.parallel_mode = parallel_model
        init_context(seed, use_parallel, self.context_config, self.parallel_config)

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

    def get_eval_dataloader(self):
        """get_eval_dataloader."""

    def get_train_dataloader(self):
        """get_train_dataloader."""

    def compute_loss(self):
        """compute_loss."""

    def count_parameter(self):
        """count_parameter."""

    def save_metrics(self):
        """save_metrics."""

    def save_model(self):
        """save_model."""
