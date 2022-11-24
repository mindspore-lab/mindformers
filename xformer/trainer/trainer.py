from typing import Callable, List, Optional, Union

from xformer.xformer_book import XFormerBook
from xformer.tools.register import XFormerConfig
from xformer.models import build_model, build_tokenizer
from xformer.dataset import build_dataset
from xformer.trainer import build_trainer
from xformer.common.optim import build_optim
from xformer.common.lr import build_lr
from xformer.common.callback import build_callback
from xformer.common.context import init_context
from xformer.common.parallel_config import build_parallel_config


SUPPORT_TASKS = XFormerBook().get_trainer_support_task_list()


class Trainer:
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

        task_config = XFormerConfig(SUPPORT_TASKS.get(self.task_name).get(self.model_name))

        if self.model_name == "common":
            task_config.trainer.model_name = "Your Self-Define Model"

        if config is None:
            self.config = task_config
        else:
            task_config.merge_from_dict(config)
            self.config = task_config

            # context init 待补充, 包含并行配置初始化
        init_context(seed=self.config.seed, use_parallel=self.config.use_parallel,
                     context_config=self.config.context, parallel_config=self.config.parallel)

        self.context_config = self.config.context
        self.parallel_config = self.config.parallel

        build_parallel_config(self.config)

    def train(self, resume_from_checkpoint: Optional[Union[str, bool]] = None, **kwargs):
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
            self.tokenizer = build_tokenizer(self.config.tokenizer)  # 待补充

        if self.callbacks is None:
            self.callbacks = self.create_callbacks()

        task = build_trainer(self.config.task)
        task.train(
            config=self.config, network=self.model,
            dataset=self.train_dataset, optimizer=self.optimizers,
            tokenizer=self.tokenizer, callbacks=self.callbacks, **kwargs)

    def evaluate(self, eval_checkpoint: str = None, **kwargs):
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

        task = build_trainer(self.config.task)
        task.evaluate(
            config=self.config, network=self.model,
            dataset=self.eval_dataset, tokenizer=self.tokenizer,
            callbacks=self.callbacks, **kwargs)

    def create_optimizer_and_scheduler(self):
        lr_schedule = self.create_scheduler()
        params = self.model.trainable_params()
        return self.create_optimizer(lr_schedule, params)

    def create_scheduler(self):
        return build_lr(self.config.lr_schedule)

    def create_optimizer(self, lr_schedule, params):
        return build_optim(self.config.optimizer, default_args={"params": params,
                                                                "learning_rate": lr_schedule})

    def create_callbacks(self):
        return build_callback(self.config.callbacks)

    def set_context(self, seed=0, use_parallel=False, device_id=0, device_target="Ascend", parallel_model=0):
        self.context_config.device_id = device_id
        self.context_config.device_target = device_target
        self.parallel_config.parallel_mode = parallel_model
        init_context(seed, use_parallel, self.context_config, self.parallel_config)

    def set_parallel_config(
            self, data_parallel=1, model_parallel=1, expert_parallel=1, pipeline_stage=1,
            micro_batch_num=1, optimizer_shard=False, gradient_aggregation_group=4, vocab_emb_dp=True):
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
        self.config.recompute_config.recompute = recompute
        self.config.recompute_config.parallel_optimizer_comm_recompute = parallel_optimizer_comm_recompute
        self.config.recompute_config.mp_comm_recompute = mp_comm_recompute
        self.config.recompute_config.recompute_slice_activation = recompute_slice_activation

    def set_moe_config(self, expert_num=1, capacity_factor=1.1, aux_loss_factor=0.05, num_experts_chosen=1):
        self.config.moe_config.expert_num = expert_num
        self.config.moe_config.capacity_factor = capacity_factor
        self.config.moe_config.aux_loss_factor = aux_loss_factor
        self.config.moe_config.num_experts_chosen = num_experts_chosen

    def get_eval_dataloader(self):
        pass

    def get_train_dataloader(self):
        pass

    def compute_loss(self):
        pass

    def count_parameter(self):
        pass

    def save_metrics(self):
        pass

    def save_model(self):
        pass
