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
from pprint import pprint
from typing import List, Optional, Union

import numpy as np
from PIL.Image import Image

from mindspore import Tensor
from mindspore.common import set_seed
from mindspore.nn import TrainOneStepCell, Optimizer
from mindspore.train import Callback
from mindspore.dataset import GeneratorDataset

from mindformers.common.parallel_config import build_parallel_config
from mindformers.dataset import build_dataset, build_dataset_loader, \
    check_dataset_config, BaseDataset
from mindformers.mindformer_book import MindFormerBook
from mindformers.models import BaseModel, BaseImageProcessor, BaseTokenizer, BaseAudioProcessor
from mindformers.tools.cloud_adapter import CFTS
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerConfig, MindFormerRegister
from mindformers.tools.register.config import ordered_yaml_dump
from .build_trainer import build_trainer
from .config_args import ConfigArguments
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
    Trainer package to train\evaluate\predict class.

    The trainer interface is used to quickly start training, evaluation and predict
    for integrated tasks. It also allows users to customize the model, optimizer, dataset,
    tokenizer, processor, train_one_step, callback, and metric.

    Args:
        config (Optional[Union[str, dict, ConfigArguments]]): The task config which is used to
            configure the dataset, the hyper-parameter, optimizer, etc. It support yaml path or
            config dict or ConfigArguments class.
            Default: None.
        task (str): The task name supported. Please refer to *****.
            Default: 'general'.
        model (Optional[Union[str, BaseModel]]): The network for trainer. It support model name supported
            or BaseModel class. Supported model name can refer to ****.
            Default: None.
        train_dataset (Optional[Union[str, BaseDataset]]): The training dataset. It support real dataset path or
            BaseDateset class or MindSpore Dataset class.
            Default: None.
        eval_dataset (Optional[Union[str, BaseDataset]]): The evaluate dataset. It support real dataset path or
            BaseDateset class or MindSpore Dataset class.
            Default: None.
        tokenizer (Optional[BaseTokenizer]): The tokenizer for text preprocessing. It support BaseTokenizer class.
            Default: None.
        image_processor (Optional[BaseImageProcessor]): The processor for image preprocessing.
            It support BaseImageProcessor class.
            Default: None.
        audio_processor (Optional[BaseAudioProcessor]): The processor for audio preprocessing.
            It support BaseAudioProcessor class.
            Default: None.
        optimizers (Optional[Optimizer]): The training network's optimizer. It support Optimizer class of MindSpore.
            Default: None.
        wrapper (Optional[TrainOneStepCell]): Wraps the `network` with the `optimizer`.
            It support TrainOneStepCell class of MindSpore.
            Default: None.
        callbacks (Optional[Union[Callback, List[Callback]]]): The training callback function.
            It support CallBack or CallBack List of MindSpore.
            Default: None.
        eval_callbacks (Optional[Union[Callback, List[Callback]]]): The evaluate callback function.
            It support CallBack or CallBack List of MindSpore.
            Default: None.
        compute_metrics (Optional[Union[dict, set]]): The metric of evaluating.
            It support dict or set in MindSpore's Metric class.
            Default: None.
        save_config (bool): Save current the config of task. Default: False.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Raises:
        KeyError: If 'task' or 'model' not in supported trainer.

    Examples:
        >>> from mindformers import Trainer
        >>> import numpy as np
        >>> from mindspore.dataset import GeneratorDataset
        >>> class MyDataLoader:
        ...    def __init__(self):
        ...        self._data = [np.zeros((3, 224, 224), np.float32) for _ in range(64)]
        ...
        ...    def __getitem__(self, index):
        ...        return self._data[index]
        ...
        ...    def __len__(self):
        ...        return len(self._data)
        >>> #1) input task name and model name to init trainer
        >>> task_trainer = Trainer(task='image_classification',
        ...                        model='vit_base_p16',
        ...                        train_dataset='data/imagenet/train')
        >>> #2) input config to init trainer
        >>> from mindformers.trainer.config_args import ConfigArguments, OptimizerConfig, \
        ...     RunnerConfig, LRConfig, WrapperConfig
        >>> from mindspore.nn import AdamWeightDecay, WarmUpLR, \
        ...     DynamicLossScaleUpdateCell, TrainOneStepWithLossScaleCell
        >>> from mindspore.train.callback import LossMonitor
        >>> runner_config = RunnerConfig(epochs=10, batch_size=2, image_size=224)
        >>> lr_schedule_config = LRConfig(lr_type='WarmUpLR', learning_rate=0.001, warmup_steps=10)
        >>> optim_config = OptimizerConfig(optim_type='Adam', beta1=0.009, learning_rate=lr_schedule_config)
        >>> loss_scale = DynamicLossScaleUpdateCell(loss_scale_value=2**12, scale_factor=2, scale_window=1000)
        >>> wrapper_config = WrapperConfig(wrapper_type='TrainOneStepWithLossScaleCell', scale_sense=loss_scale)
        >>> dataset = GeneratorDataset(source=MyDataLoader(), column_names='image')
        >>> dataset = dataset.batch(batch_size=2)
        >>> config = ConfigArguments(seed=2022, runner_config=runner_config,
        ...                          optimizer=optim_config, runner_wrapper=wrapper_config)
        >>> task_trainer = Trainer(task='image_classification',
        ...                        model='vit_base_p16',
        ...                        config=config, train_dataset=dataset)
        >>> #3) input instance to init trainer
        >>> from mindformers.models import VitModel
        >>> vit_model_with_loss = VitModel()
        >>> lr_schedule = WarmUpLR(learning_rate=0.001, warmup_steps=100)
        >>> optimizer = AdamWeightDecay(beta1=0.009, beta2=0.999,
        ...                             learning_rate=lr_schedule,
        ...                             params=vit_model_with_loss.trainable_params())
        >>> loss_cb = LossMonitor(per_print_times=2)
        >>> callbacks = [loss_cb]
        >>> task_trainer = Trainer(task='image_classification',
        ...                        model=vit_model_with_loss,
        ...                        config=config,
        ...                        optimizers=optimizer,
        ...                        train_dataset=dataset,
        ...                        callbacks=callbacks)
    """

    def __init__(self,
                 config: Optional[Union[str, dict, ConfigArguments]] = None,
                 task: Optional[str] = 'general',
                 model: Optional[Union[str, BaseModel]] = None,
                 train_dataset: Optional[Union[str, BaseDataset]] = None,
                 eval_dataset: Optional[Union[str, BaseDataset]] = None,
                 tokenizer: Optional[BaseTokenizer] = None,
                 image_processor: Optional[BaseImageProcessor] = None,
                 audio_processor: Optional[BaseAudioProcessor] = None,
                 optimizers: Optional[Optimizer] = None,
                 wrapper: Optional[TrainOneStepCell] = None,
                 callbacks: Optional[Union[Callback, List[Callback]]] = None,
                 eval_callbacks: Optional[Union[Callback, List[Callback]]] = None,
                 compute_metrics: Optional[Union[dict, set]] = None,
                 save_config: bool = False,
                 **kwargs):

        self.task = task
        self.model = model
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
        self.kwargs = kwargs

        if not os.path.exists(os.path.join('.', DEFAULT_CONFIG_DIR)):
            configs_directory = os.path.join('.', DEFAULT_CONFIG_DIR)
            if os.path.exists(os.path.join(CURRENT_PROJECT_PATH, DEFAULT_CONFIG_DIR)):
                mindformers_configs_directory = os.path.join(CURRENT_PROJECT_PATH, DEFAULT_CONFIG_DIR)
                shutil.copytree(mindformers_configs_directory, configs_directory)

        if wrapper is not None:
            if model is not None:
                logger.warning(
                    'wrapper has existed, input model invalid, it should be include in wrapper.')
            if optimizers is not None:
                logger.warning(
                    'wrapper has existed, input optimizers invalid, it should be include in wrapper.')

        assert task in SUPPORT_TASKS.keys(), \
            f"task name must be in {SUPPORT_TASKS.keys()}, but get {task}."
        if isinstance(model, str):
            assert model in SUPPORT_MODEL_NAMES, \
                f"model must be in {SUPPORT_MODEL_NAMES} when model's type is string, but get {model}."
            self.model_name = model
            self.model = None
        else:
            self.model_name = "common"

        task_config = MindFormerConfig(SUPPORT_TASKS.get(self.task).get(self.model_name))

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

        if save_config:
            self.save_config_to_yaml(self.config)
            logger.info("save running config success of %s_new.", task_config.trainer.model_name.lower())

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

        # pprint last config
        pprint(self.config)

    def train(self, resume_or_finetune_from_checkpoint: Optional[Union[str, bool]] = False,
              initial_epoch: int = 0, do_eval: bool = False, do_finetune: bool = False, **kwargs):
        r"""Train task for Trainer.
        This function is used to train or fine-tune the network.

        Args:
            resume_or_finetune_from_checkpoint (Optional[Union[str, bool]]):
                Used to restore training or fine-tune the weight of the network.
                It support real checkpoint path or valid model name of mindformers or bool value.
                if it's true, the last checkpoint file saved from the previous training round is automatically used.
                if do_finetune is true, this checkpoint will be used to finetune the network.
                Default: False.
            initial_epoch (int): Epoch at which to start train, it used for resuming a previous training run.
                Default: 0.
            do_eval (bool): Whether evaluations are performed during training. Default: False.
            do_finetune: Whether to finetune network. When it's true, resume_or_finetune_from_checkpoint must be input.
                Default: False.

        Raises:
            TypeError: if resume_or_finetune_from_checkpoint is not bool or str type.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> from mindformers import Trainer
            >>> task_trainer = Trainer(task='image_classification',
            ...                        model='vit_base_p16',
            ...                        train_dataset='data/imagenet/train',
            ...                        eval_dataset='data/imagenet/train')
            >>> # 1) default train task to reproduce model.
            >>> task_trainer.train()
            >>> # 2) eval network when train task to reproduce model.
            >>> task_trainer.train(do_eval=True)
            >>> # 3) resume train task to auto load the last checkpoint, if training break after 10 epochs.
            >>> task_trainer.train(resume_or_finetune_from_checkpoint=True, initial_epoch=10)
            >>> # 4) resume train task according to checkpoint path, if training break after 10 epochs.
            >>> task_trainer.train(
            ...     resume_or_finetune_from_checkpoint='./output/rank_0/checkpoint/mindformers.ckpt',
            ...     initial_epoch=10)
            >>> # 5) finetune train task according to resume_or_finetune_from_checkpoint.
            >>> task_trainer.train(resume_or_finetune_from_checkpoint='mae_vit_base_p16', do_finetune=True)
        """
        if resume_or_finetune_from_checkpoint is not None and \
                not isinstance(resume_or_finetune_from_checkpoint, (bool, str)):
            raise TypeError(f"resume_or_finetune_from_checkpoint must be one of [None, string, bool], "
                            f"but get {resume_or_finetune_from_checkpoint}")
        if resume_or_finetune_from_checkpoint is False:
            resume_or_finetune_from_checkpoint = None

        if do_finetune and resume_or_finetune_from_checkpoint is None:
            logger.warning("if do_finetuen is true, "
                           "resume_or_finetune_from_checkpoint must be input and valid, "
                           "but it's None.")

        if do_eval:
            if self.eval_dataset is None:
                self.eval_dataset = build_dataset(self.config.eval_dataset_task)

        if resume_or_finetune_from_checkpoint is True:
            self.config.model.model_config.checkpoint_name_or_path = None
            self.config.resume_checkpoint_path = resume_or_finetune_from_checkpoint
        elif isinstance(resume_or_finetune_from_checkpoint, str):
            if do_finetune:
                self.config.model.model_config.checkpoint_name_or_path = resume_or_finetune_from_checkpoint
            else:
                self.config.model.model_config.checkpoint_name_or_path = None
                self.config.resume_checkpoint_path = resume_or_finetune_from_checkpoint
        else:
            self.default_checkpoint_name_or_path = self.config.model.model_config.checkpoint_name_or_path
            self.config.model.model_config.checkpoint_name_or_path = None

        if initial_epoch != 0:
            self.config.runner_config.initial_epoch = initial_epoch

        trainer = build_trainer(self.config.trainer)
        trainer.train(
            config=self.config, network=self.model,
            dataset=self.train_dataset, optimizer=self.optimizers,
            eval_dataset=self.eval_dataset if do_eval else None,
            wrapper=self.wrapper,
            callbacks=self.callbacks, **kwargs)

    def evaluate(self, eval_checkpoint: Optional[Union[str, bool]] = False, **kwargs):
        r"""Evaluate task for Trainer.
        This function is used to evaluate the network.

        Args:
            eval_checkpoint (Optional[Union[str, bool]]):
                Used to evaluate the weight of the network.
                It support real checkpoint path or valid model name of mindformers or bool value.
                if it's true, the last checkpoint file saved from the previous training round is automatically used.
                Default: False.

        Raises:
            TypeError: if eval_checkpoint is not bool or str type.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> from mindformers import Trainer
            >>> task_trainer = Trainer(task='image_classification',
            ...                        model='vit_base_p16',
            ...                        train_dataset='data/imagenet/train',
            ...                        eval_dataset='data/imagenet/train')
            >>> # 1) default evaluate task to test model.
            >>> task_trainer.evaluate()
            >>> # 2) evaluate task to auto load the last checkpoint.
            >>> task_trainer.evaluate(eval_checkpoint=True)
            >>> # 3) evaluate task according to checkpoint path.
            >>> task_trainer.evaluate(eval_checkpoint='./output/rank_0/checkpoint/mindformers.ckpt')
        """
        if eval_checkpoint is not None and not isinstance(eval_checkpoint, (bool, str)):
            raise TypeError(f"eval_checkpoint must be one of [None, string, bool], "
                            f"but get {eval_checkpoint}")

        if eval_checkpoint is False:
            eval_checkpoint = None

        self._check_checkpoint_config(eval_checkpoint)

        trainer = build_trainer(self.config.trainer)
        trainer.evaluate(
            config=self.config, network=self.model,
            dataset=self.eval_dataset, callbacks=self.callbacks, **kwargs)

    def predict(self,
                predict_checkpoint: Optional[Union[str, bool]] = None,
                input_data: Optional[Union[GeneratorDataset,
                                           Tensor, np.ndarray, Image, str, list]] = None, **kwargs):
        r"""Predict task for Trainer.
        This function is used to evaluate the network.

        Args:
            predict_checkpoint (Optional[Union[str, bool]]):
                Used to predict the weight of the network.
                It support real checkpoint path or valid model name of mindformers or bool value.
                if it's true, the last checkpoint file saved from the previous training round is automatically used.
                Default: False.
            input_data (Optional[Union[Tensor, np.ndarray, Image, str, list]]): The predict data. Default: None.

        Return:
            predict result (dict).

        Raises:
            TypeError: if predict_checkpoint is not bool or str type.
            TypeError: if input_data is not Tensor or np.ndarray or Image or str or list.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> from mindformers import Trainer
            >>> task_trainer = Trainer(task='image_classification',
            ...                        model='vit_base_p16',
            ...                        train_dataset='data/imagenet/train',
            ...                        eval_dataset='data/imagenet/train')
            >>> # 1) default predict task to test model.
            >>> task_trainer.predict()
            >>> # 2) predict task to auto load the last checkpoint.
            >>> task_trainer.evaluate(predict_checkpoint=True)
            >>> # 3) predict task according to checkpoint path.
            >>> task_trainer.evaluate(predict_checkpoint='./output/rank_0/checkpoint/mindformers.ckpt')
            >>> # 4) predict task according to input data.
            >>> input_data = "./sunflower.png"
            >>> task_trainer.predict(input_data=input_data)
        """
        if predict_checkpoint is not None and not isinstance(predict_checkpoint, (bool, str)):
            raise TypeError(f"predict_checkpoint must be one of [None, string, bool], "
                            f"but get {predict_checkpoint}")

        if self.task not in SUPPORT_PIPELINES.keys():
            raise NotImplementedError(f"The {self.task} not support predict, "
                                      f"now this tasks {SUPPORT_PIPELINES.keys()} is support predict.")

        if predict_checkpoint is False:
            predict_checkpoint = None

        if input_data is None:
            input_data = build_dataset_loader(self.config.eval_dataset.data_loader)
            logger.info("dataset by config is used as input_data.")
        assert isinstance(input_data, (GeneratorDataset, Tensor, np.ndarray, Image, str, list)), \
            "Input data's type must be one of [GeneratorDataset," \
            " str, ms.Tensor, np.ndarray, PIL.Image.Image]"

        self._check_checkpoint_config(predict_checkpoint)

        trainer = build_trainer(self.config.trainer)
        output_result = trainer.predict(
            config=self.config, input_data=input_data,
            network=self.model, image_processor=self.image_processor,
            audio_processor=self.audio_processor,
            tokenizer=self.tokenizer, **kwargs)
        return output_result

    def set_parallel_config(
            self, data_parallel=1, model_parallel=1, expert_parallel=1, pipeline_stage=1,
            micro_batch_num=1, optimizer_shard=False, gradient_aggregation_group=4, vocab_emb_dp=True):
        r"""
        set_parallel_config for the setting global data parallel, model parallel and fusion group.
        The parallel configure setting for Trainer.

        Args:
            data_parallel (int): The data parallel way. The input data will be sliced into n parts for each layer
                according to the data parallel way. Default: 1.
            model_parallel (int): The model parallel way. The parameters of dense layers in MultiheadAttention and
                FeedForward layer will be sliced according to the model parallel way. Default: 1.
            expert_parallel (int): The expert parallel way. This is effective only when MoE (Mixture of Experts)
                is applied. This value specifies the number of partitions to split the experts into.
            pipeline_stage (int): The number of the pipeline stage. Should be a positive value. Default: 1.
            micro_batch_num (int): The micro size of the batches for the pipeline training. Default: 1.
            optimizer_shard (bool): Whether to enable optimizer shard. Default False.
            gradient_aggregation_group (int): The fusion group size of the optimizer state sharding. Default: 4.
            vocab_emb_dp (bool): Shard embedding in model parallel or data parallel. Default: True.

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> from mindformers.trainer import Trainer
            >>> task_trainer = Trainer(task='image_classification',
            ...                        model='vit_base_p16',
            ...                        train_dataset='data/imagenet/train',
            ...                        eval_dataset='data/imagenet/train')
            >>> task_trainer.set_parallel_config(data_parallel=2, model_parallel=2)
        """
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
        r"""Set recompute config.
        TransformerRecomputeConfig for the setting recompute attributes for encoder/decoder layers.

        Args:
            recompute (bool): Enable recomputation of the transformer block or not. Default: False.
            parallel_optimizer_comm_recompute (bool): Specifies whether the communication operator allgathers
                introduced by optimizer shard are recomputed in auto parallel or semi auto parallel mode.
                Default: False.
            mp_comm_recompute (bool): Specifies whether the model parallel communication operators
                in the cell are recomputed in auto parallel or semi auto parallel mode. Default: True.
            recompute_slice_activation (bool): Slice the cell output which would remains in memory. Default: False.

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> from mindformers.trainer import Trainer
            >>> task_trainer = Trainer(task='image_classification',
            ...                        model='vit_base_p16',
            ...                        train_dataset='data/imagenet/train',
            ...                        eval_dataset='data/imagenet/train')
            >>> task_trainer.set_recompute_config(recompute=True)
        """
        self.config.recompute_config.recompute = recompute
        self.config.recompute_config.parallel_optimizer_comm_recompute = parallel_optimizer_comm_recompute
        self.config.recompute_config.mp_comm_recompute = mp_comm_recompute
        self.config.recompute_config.recompute_slice_activation = recompute_slice_activation

    def set_moe_config(self,
                       expert_num=1,
                       capacity_factor=1.1,
                       aux_loss_factor=0.05,
                       num_experts_chosen=1,
                       expert_group_size=None,
                       group_wise_a2a=False,
                       comp_comm_parallel=False,
                       comp_comm_parallel_degree=2):
        r"""The configuration of MoE (Mixture of Expert).

        Args:
            expert_num (int): The number of experts employed. Default: 1
            capacity_factor (float): The factor is used to indicate how much to expand expert capacity,
                which is >=1.0. Default: 1.1.
            aux_loss_factor (float): The factor is used to indicate how much the load balance loss (produced by the
                router) to be added to the entire model loss, which is < 1.0. Default: 0.05.
            num_experts_chosen (int): The number of experts is chosen by each token and it should not be larger
                than expert_num. Default: 1.
            expert_group_size (int): The number of tokens in each data parallel group. Default: None. This parameter is
                effective only when in AUTO_PARALLEL mode, and NOT SHARDING_PROPAGATION.
            group_wise_a2a (bool): Whether to enable group-wise alltoall communication, which can reduce communication
                time by converting part of inter communication into intra communication. Default: False. This parameter
                is effective only when model parallel > 1 and data_parallel equal to expert parallel.
            comp_comm_parallel (bool): Whether to enable ffn compute and communication parallel, which can reduce pure
                communicattion time by splitting and overlapping compute and communication. Default: False.
            comp_comm_parallel_degree (int): The split number of compute and communication. The larger the numbers,
                the more overlap there will be but will consume more memory. Default: 2. This parameter is effective
                only when comp_comm_parallel enable.

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> from mindformers.trainer import Trainer
            >>> task_trainer = Trainer(task='image_classification',
            ...                        model='vit_base_p16',
            ...                        train_dataset='data/imagenet/train',
            ...                        eval_dataset='data/imagenet/train')
            >>> task_trainer.set_moe_config(expert_num=2, capacity_factor=1.2, aux_loss_factor=0.001)
        """
        self.config.moe_config.expert_num = expert_num
        self.config.moe_config.capacity_factor = capacity_factor
        self.config.moe_config.aux_loss_factor = aux_loss_factor
        self.config.moe_config.num_experts_chosen = num_experts_chosen
        self.config.moe_config.expert_group_size = expert_group_size
        self.config.moe_config.group_wise_a2a = group_wise_a2a
        self.config.moe_config.comp_comm_parallel = comp_comm_parallel
        self.config.moe_config.comp_comm_parallel_degree = comp_comm_parallel_degree

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

    def save_config_to_yaml(self, config: dict = None):
        """save now config file to yaml file."""
        if config is None:
            config = self.config
        model_name = self.config.trainer.model_name
        config_dict = _reset_config_for_save(config, model_name)
        config_dir = os.path.join(
            self.configs_directory, model_name.lower() + '_new')
        if not os.path.exists(config_dir):
            os.makedirs(config_dir, exist_ok=True)

        model_config_dir = os.path.join(config_dir, 'model_config')
        task_config_dir = os.path.join(config_dir, 'task_config')
        if not os.path.exists(model_config_dir):
            os.makedirs(model_config_dir, exist_ok=True)

        if not os.path.exists(task_config_dir):
            os.makedirs(task_config_dir, exist_ok=True)

        model_config_yaml_path = os.path.join(
            model_config_dir, '{}.yaml'.format(model_name.lower()))
        dataset_config_yaml_path = os.path.join(
            task_config_dir, '{}_dataset.yaml'.format(model_name.lower()))
        runner_yaml_path = os.path.join(task_config_dir, 'runner.yaml')
        context_yaml_path = os.path.join(task_config_dir, 'context.yaml')
        run_yaml_path = os.path.join(config_dir, 'run_{}.yaml'.format(model_name.lower()))

        _save_config_to_yaml(model_config_yaml_path, config_dict.get('model_config'))
        _save_config_to_yaml(dataset_config_yaml_path, config_dict.get('dataset_config'))
        _save_config_to_yaml(runner_yaml_path, config_dict.get('runner_config'))
        _save_config_to_yaml(context_yaml_path, config_dict.get('context_config'))
        _save_config_to_yaml(run_yaml_path, config_dict.get('run_config'))

    def _check_checkpoint_config(self, checkpoint: Optional[Union[str, bool]] = None):
        """check checkpoint config."""
        if checkpoint is True:
            self.config.model.model_config.checkpoint_name_or_path = self.get_last_checkpoint()
        elif isinstance(checkpoint, str):
            self.config.model.model_config.checkpoint_name_or_path = checkpoint
        else:
            if self.default_checkpoint_name_or_path is not None:
                self.config.model.model_config.checkpoint_name_or_path = self.default_checkpoint_name_or_path


def _save_config_to_yaml(save_file_path: str = None, save_config: dict = None):
    r"""Save Config to Yaml File.

    Args:
        save_file_path (str): The real path to save yaml file. Default: None.
        save_config (dict): The task config. Default: None.
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


def _reset_config_for_save(config: dict = None, model_name: str = 'common'):
    r"""Reset Config According to Yaml File Number.
    Args:
        config (dict): The task config. Default: None.
        model_name (str): The model name to save. Default: 'common'.
    """
    if config is None:
        config = {}
    config = config.copy()

    config_dict = {
        "model_config": OrderedDict(),
        "dataset_config": OrderedDict(),
        "runner_config": OrderedDict(),
        "context_config": OrderedDict(),
        "run_config": OrderedDict()
    }

    if config.get('model') is not None:
        model_config = config2dict(config.pop('model'))
        config_dict["model_config"].setdefault('model', model_config)

    if config.get('processor') is not None:
        processor_config = config2dict(config.config.pop('processor'))
        config_dict["model_config"].setdefault('processor', processor_config)

    if config.get('train_dataset_task') is not None and config.get('train_dataset') is not None:
        train_dataset_config = config2dict(config.pop('train_dataset'))
        train_dataset_task_config = config2dict(config.pop('train_dataset_task'))
        config_dict["dataset_config"].setdefault('train_dataset', train_dataset_config)
        config_dict["dataset_config"].setdefault('train_dataset_task', train_dataset_task_config)

    if config.get('eval_dataset_task') is not None and config.get('eval_dataset') is not None:
        eval_dataset_config = config2dict(config.pop('eval_dataset'))
        eval_dataset_task_config = config2dict(config.pop('eval_dataset_task'))
        config_dict["dataset_config"].setdefault('train_dataset', eval_dataset_config)
        config_dict["dataset_config"].setdefault('train_dataset_task', eval_dataset_task_config)

    if config.get('context') is not None:
        context_config = config2dict(config.pop('context'))
        parallel_context_config = config2dict(config.pop('parallel'))
        moe_conifg = config2dict(config.pop('moe_config'))
        recompute_config = config2dict(config.pop('recompute_config'))
        parallel_config = config2dict(config.pop('parallel_config'))
        config_dict['context_config'].setdefault('context', context_config)
        config_dict['context_config'].setdefault('parallel', parallel_context_config)
        config_dict['context_config'].setdefault('moe_conifg', moe_conifg)
        config_dict['context_config'].setdefault('recompute_config', recompute_config)
        config_dict['context_config'].setdefault('parallel_config', parallel_config)

    if config.get('runner_config') is not None:
        runner_config = config2dict(config.pop('runner_config'))
        config_dict['runner_config'].setdefault('runner_config', runner_config)

    if config.get('runner_wrapper') is not None:
        wrapper_config = config2dict(config.pop('runner_wrapper'))
        config_dict['runner_config'].setdefault('runner_wrapper', wrapper_config)

    if config.get('optimizer') is not None:
        optim_config = config2dict(config.pop('optimizer'))
        config_dict['runner_config'].setdefault('optimizer', optim_config)

    if config.get('lr_schedule') is not None:
        lr_config = config2dict(config.pop('lr_schedule'))
        config_dict['runner_config'].setdefault('lr_schedule', lr_config)

    if config.get('callbacks') is not None:
        cb_config = config2dict(config.pop('callbacks'))
        config_dict['runner_config'].setdefault('callbacks', cb_config)

    config_dict['run_config'].setdefault('base_config', [
        './task_config/context.yaml',
        './task_config/runner.yaml',
        './task_config/{}_dataset.yaml'.format(model_name.lower()),
        './model_config/{}.yaml'.format(model_name.lower())])

    run_config = config2dict(config)
    for key, value in run_config.items():
        config_dict['run_config'].setdefault(key, value)

    return config_dict
