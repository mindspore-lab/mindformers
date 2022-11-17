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
"""
Trainer Class for quick training.
"""

import argparse
import os
import json
from dataclasses import dataclass

import mindspore
from mindspore import context, DynamicLossScaleManager
from mindspore import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.train.model import Model
from mindspore.train.callback import TimeMonitor, ModelCheckpoint, CheckpointConfig
from mindspore.nn.transformer import TransformerRecomputeConfig, MoEConfig, TransformerOpParallelConfig
from mindspore.nn.wrap.loss_scale import TrainOneStepWithLossScaleCell
from mindspore.nn.wrap.cell_wrapper import MicroBatchInterleaved
from mindspore.nn.wrap.loss_scale import TrainOneStepCell
import mindspore.communication.management as D
import mindspore.common.dtype as mstype

from mindtransformer.auto_class import AutoClass
from mindtransformer.optim.optimizer import build_optimizer
from mindtransformer.utils import print_model_size, get_newest_ckpt, download_data
from mindtransformer.trainer.grad_accu_model import AccModel
from mindtransformer.learning_rate import LearningRate
from mindtransformer.modules import override_attention
from mindtransformer.callback import LossCallBack
from mindtransformer.generate import generate_words
from mindtransformer.logger import get_logger
from mindtransformer.utils import get_acc
from mindtransformer.utils import _mapper_string_to_bool
from mindtransformer.trainer.grad_accu_trainer import TrainAccuStepsWithLossScaleCell


@dataclass
class TrainingConfig:
    """
        The training configuration for the Trainer. This configuration controls the setting of the following:
        mindspore.context、mindspore.context.auto_parallel_context and the training configurations.

        Args:
            is_training(bool): Set the is_training of the net.
                For the user want to train the network, the is_training should be True. Otherwise it should be False.
                Default True.
            auto_model(str): The net name to initialize the network. If provided, the trainer will initialize
                the net with the predefined configuration. Options: ['gpt', 'bert', 'vit']. Default "".
            micro_batch_size(int): The batch size of each data parallel way. Default 4.
            global_batch_size(int): The batch size of the global batch size. Default 4.
            dropout_rate(float): The dropout rate of the net. Default 0.1.
            seed(int): The seed of the weight initialization. The seed will be called by mindspore.common.seed in
                Trainer.train. Default None.
            device_target(str): The device target of the programs. This determines which hardware is running on.
                Options: ["Ascend", "GPU", "CPU"]. Default "GPU".
            save_graphs(bool): Whether to save graphs. If True, the output graphs will be saved under the path where
                program starts. Default False.
            mode(int): The running mode of the mindspore context. It determines whether the net is running under the
                graph mode or pynative mode. Options: [mindspore.context.GRAPH_MODE(0),
                mindspore.context.PYNATIVE_MODE(1)]. Default mindspore.context.GRAPH_MODE(0).
            graph_kernel_flags(str): Optimization options of graph kernel fusion, and the priority is higher when it
                conflicts with enable_graph_kernel. Only for experienced users. Please see the document of the
                graph_kernel_flags in MindSpore.
                Default "--disable_expand_ops=Softmax,Dropout --enable_parallel_fusion=true \
                         --reduce_fuse_depth=8 --enable_auto_tensor_inplace=true".
            enable_graph_kernel(bool): Whether to enable graph kernel fusion to optimize network execution performance.
                Indicates whether to enable graph kernel computation to optimize network execution performance.
                If enable_graph_kernel is set to True, acceleration can be enabled. Note this will be
                effective only on GPU. Default True.
            optimizer(str): The optimizer to train the network. It only supports the "adam" now.
                The user can override the trainer.build_optimizer to return the custom optimizer. Default "adam".
            acc_step(int): The gradient accumulation step when do gradient accumulation. If the set value is larger
                than 1, it will enable the gradient accumulation. Default 1.
            full_batch(bool): If enabled, the data read from the dataset will be viewed as the full data. This is
                effective only when the user runs on SEMI/AUTO parallel model. Default True.
            train_data_path(str): The train data path. It is used in Trainer.build_dataset. It will load the mindrecords
                from the given path. If user wants to do custom operations about the dataset, the user can override
                the class method of Trainer.build_datset. Default "".
            epoch_size(int): The training epochs of the Trainer.train(). Default 1.
            start_lr(float): The start learning rate. Default 1e-4.
            end_lr(float): The end learning rate. Default 1e-5.
            warmup_step(int): The warmup steps to train the model. The learning rate will increase from 0 and reach at
                the start_lr. Default 0.
            opt_offload(bool): Enabele the offload training. The parameter, optimizer state and update will be
                on the host. Default False.
            sink_size(int): This number controls how many steps to return to the host when training on the Ascend.
                Please see the document of sink_size in MindSpore's API. Default 10.
            init_loss_scale_value(float): The inilization of the loss scale. Note most of the networks in this
                repo enable mixed precision training as the default setting, so just keep this value as the default.
                Default 4294967296.
            scale_factor(int): The scale factor of the loss scale when overflow happens. Note most of the networks in
                this repo enable mixed precision training as the default setting, so just keep this value as the
                default. Default 2.
            scale_window(int): The scale window of the loss scale when no overflow happens. Note most of the networks in
                this repo enable mixed precision training as the default setting, so just keep this value as the
                default. Default 1000.
            eval(bool): If the mode is running on the evaluation. Default False.
            eval_batch_size(int): The evaluation batch size. Default 1.
            eval_data_path(str): The evaluation data path of the network. Default "".
            dataset_format: str = "mindrecord"

            load_checkpoint_path(str): The restored checkpoint path. If set, the net will try to restore the
                checkpoints from the given path. Default "".
            save_checkpoint(bool): Enable the checkpoint saving. If set, the net will save the
                checkpoints. Default True.
            save_checkpoint_path(str): The saved checkpoint path. If set, the net will save the
                checkpoints from the given path. Default "".
            checkpoint_prefix(str): The prefix of the checkpoint name. Default "tmp".
            compute_dtype(mindspore.dtype): The computation dtype of the matmul of Transformer Layer.
                Default mstype.float16.
            layernorm_dtype(mindspore.dtype): The computation dtype of the layernorm of Transformer Layer.
                Default mstype.float32.
            softmax_dtype(mindspore.dtype): The computation dtype of the softmax of Transformer Layer.
                Default mstype.float16.
            dataset_drop_remainder(bool): Drop the remaining data that is less than a batch. Default True.
            dataset_do_shuffle(bool): If do the shuffle for the dataset. Default True.
            dataset_schema_file_path: str = ""
            dataset_device_num: int = 1
            dataset_rank: int = 0
            dataset_schema_dir: str = ""
            dataset_bucket_list: str = None
            micro_batch_interleaved_num: int = 1
            flatten_weights(bool): Reset data for weight parameters so that they are using contiguous memory
                chunks grouped by data type. Default False.
            expert_num(int):The number of experts employed. Default: 1.
            capacity_factor(float): The factor is used to indicate how much to expand expert capacity,
                which is >=1.0. Default: 1.1.
            aux_loss_factor(float): The factor is used to indicate how much the load balance loss
                (produced by the router) to be added to the entire model loss, which is < 1.0. Default: 0.05.
            num_experts_chosen(int): The number of experts is chosen by each token and it should not be larger
                than expert_num. Default: 1.
            recompute(bool): Enable the recomputation of each transformer layer. Default True.
            parallel_optimizer_comm_recompute(bool): Enable the recomputation of AllGather introduced by optimizer
                shard. Default False.
            mp_comm_recompute(bool): Enable the recomputation of AllReduce introduced by model parallel, such as matmul.
                Default False.
            recompute_slice_activation(bool): Slice the cell output which would remains in memory. Default: False.
            parallel_mode(str): The parallel mode of the running mode. Options: ["stand_alone", "auto_parallel",
                "semi_auto_parallel"]. Default "stand_alone".
            data_parallel(int): The data parallel way. The input data will be sliced into n parts for each layer
                according to the data parallel way. Default: 1.
            model_parallel(int):  The model parallel way. The parameters of dense layers in MultiheadAttention and
                FeedForward layer will be sliced according to the model parallel way. Default: 1.
            pipeline_stage(int):  The number of the pipeline stage. Should be a positive value. Default: 1.
            micro_batch_num(int):  The micro size of the batches for the pipeline training. Default: 1.
            expert_parallel(int): The expert parallel way. This is effective only when MoE (Mixture of Experts) is
                applied. This value specifies the number of partitions to split the experts into.Default 1.
            vocab_emb_dp(bool): Shard embedding in model parallel or data parallel. Default: True.
            optimizer_shard(bool): To enable the optimizer shard, known as ZeRO-2 when user runs on SEMI/AUTO parallel.
                Default False.
            gradient_aggregation_group(int): The number of fusion groups. Default 6.

        Examples:
            >>> import numpy as np
            >>> from mindtransformer.trainer import Trainer, TrainingConfig
            >>> from mindspore.dataset import GeneratorDataset
            >>> class GPTTrainer(Trainer):
            >>>     def build_model(self, model_config):
            >>>         from mindtransformer.models.gpt import GPTWithLoss
            >>>         my_net = GPTWithLoss(model_config)
            >>>         return my_net
            >>>
            >>>     def build_model_config(self):
            >>>         from mindtransformer.models.gpt import GPTConfig
            >>>         return GPTConfig(num_layers=1, hidden_size=8, num_heads=1, seq_length=14)
            >>>
            >>>     def build_dataset(self):
            >>>         def generator():
            >>>             data = np.random.randint(low=0, high=15, size=(15,)).astype(np.int32)
            >>>             for _ in range(10):
            >>>                 yield data
            >>>
            >>>         ds = GeneratorDataset(generator, column_names=["text"])
            >>>         ds = ds.batch(2)
            >>>         return ds
            >>>
            >>>     def build_lr(self):
            >>>         return 0.01
            >>> config = TrainingConfig(device_target='CPU', epoch_size=2, sink_size=2)
            >>> gpt_trainer = GPTTrainer()
            >>> gpt_trainer.train()

    """
    is_training: bool = True
    auto_model: str = ""
    micro_batch_size: int = 4
    global_batch_size: int = 4
    expand_ratio: int = 4
    dropout_rate: float = 0.1
    seed: int = None
    device_target: str = "GPU"
    save_graphs: bool = False
    mode: int = 0
    graph_kernel_flags: str = "--disable_expand_ops=Softmax,Dropout " \
                              "--enable_parallel_fusion=true --reduce_fuse_depth=8 --enable_auto_tensor_inplace=true"
    enable_graph_kernel: bool = False
    optimizer: str = "adam"
    acc_step: int = 1
    full_batch: bool = True
    train_data_path: str = ""
    epoch_size: int = 1
    start_lr: float = 1e-4
    end_lr: float = 1e-5
    warmup_step: int = 0
    opt_offload: bool = False
    sink_size: int = 10
    init_loss_scale_value: float = 4294967296
    scale_factor: int = 2
    scale_window: int = 1000
    eval: bool = False
    rank_id: int = 0
    device_num: int = 1
    get_eval_dataset: bool = False
    eval_batch_size: int = 1
    eval_data_path: str = ""
    dataset_format: str = "mindrecord"
    load_checkpoint_path: str = ""
    save_checkpoint: bool = True
    save_checkpoint_path: str = ""
    checkpoint_prefix: str = "tmp"

    compute_dtype: mstype = mstype.float16
    layernorm_dtype: mstype = mstype.float32
    softmax_dtype: mstype = mstype.float16

    # dataset
    dataset_drop_remainder: bool = True
    dataset_do_shuffle: bool = True
    dataset_schema_file_path: str = ""
    dataset_device_num: int = 1
    dataset_rank: int = 0
    dataset_schema_dir: str = ""
    dataset_bucket_list: str = None

    # speed_up:
    micro_batch_interleaved_num: int = 1
    flatten_weights: bool = False
    fused_kernel: bool = False

    # moe_config
    expert_num: int = 1
    capacity_factor: float = 1.05
    aux_loss_factor: float = 0.05
    num_experts_chosen: int = 1

    # recompute_config
    recompute: bool = False
    parallel_optimizer_comm_recompute: bool = False
    mp_comm_recompute: bool = False
    recompute_slice_activation: bool = False

    # parallel_config
    parallel_mode: str = "stand_alone"
    data_parallel: int = 1
    model_parallel: int = 1
    pipeline_stage: int = 1
    micro_batch_num: int = 1
    expert_parallel: int = 1
    vocab_emb_dp: bool = False
    optimizer_shard: bool = False
    gradient_aggregation_group: int = 6


class Trainer:
    """
    Trainer is a general procedural wrapper for training process.Generally, for a new network training, developer
    only needs to overwritten 'build_model_config', 'build_model' and 'build_dataset'.
    Parameters:
        config(TrainingConfig):
            The training config for network training
    """

    def __init__(self, config):
        self.config = config
        self.logger = get_logger()
        self.config.logger = self.logger
        self.set_context_env()
        self.set_auto_parallel_context_env()
        self.set_fused_kernel()
        self.input_args = None
        self.input_kwargs = None

    def set_context_env(self):
        """
        Set mindspore context according to training config,
        default only set device_target,save_graphs,enable_graph_kernel and graph_kernel_flags.
        To set more mindspore context flag, developer needs to overwritten this function.
        """
        if self.config.device_target != "GPU":
            self.config.enable_graph_kernel = False
            self.logger.info("Disable graph kernel.")
        context.set_context(mode=self.config.mode,
                            device_target=self.config.device_target,
                            save_graphs=self.config.save_graphs,
                            enable_graph_kernel=self.config.enable_graph_kernel,
                            graph_kernel_flags=self.config.graph_kernel_flags)

    def check_args(self, device_num):
        """Validate the dp and mp"""
        dp = self.config.data_parallel
        mp = self.config.model_parallel
        if mp < 1:
            raise ValueError("The model parallel must be equal or larger than 1. "
                             "You can fix this by setting --model_parallel=1, for example.")
        if mp > device_num:
            raise ValueError("The model parallel must be less or equal to the device_num %d. "
                             "You can fix this by setting --model_parallel=1, for example" % device_num)
        if self.config.parallel_mode in (
                ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL) and dp * mp != device_num:
            self.logger.info("The data_parallel * model_parallel must be equal to the %d. "
                             "You can remove this warning by setting --data_parallel=%d. "
                             "Now the full_batch will be set False.", device_num, device_num // mp)
            self.config.full_batch = False

        # If the user runs the data_parallel and set full_batch to be true
        if self.config.parallel_mode in (ParallelMode.DATA_PARALLEL,) and self.config.full_batch:
            raise ValueError(
                "full_batch doesn't support DATA_PARALLEL mode, you can fix it by setting --full_batch=False")

    def set_auto_parallel_context_env(self):
        """
        Set the auto parallel context according to training config, default only set
        parallel_mode,full_batch,device_num and grad_accumulation_step
        """
        if self.config.parallel_mode != context.ParallelMode.STAND_ALONE:
            self.logger.info(
                "Enabling the parallel mode: %s for multi card training.", self.config.parallel_mode)
            D.init()
            device_num = D.get_group_size()
            rank_id = D.get_rank()
            context.reset_auto_parallel_context()
            self.check_args(device_num)
            context.set_auto_parallel_context(parallel_mode=self.config.parallel_mode, gradients_mean=True,
                                              full_batch=self.config.full_batch,
                                              device_num=device_num, grad_accumulation_step=self.config.acc_step)

        else:
            self.logger.info(
                "Enabling the parallel mode: %s for stand alone training.", self.config.parallel_mode)
            rank_id = 0
            device_num = 1
        if self.config.full_batch:
            self.logger.info("Enabling the full batch import.")
        self.config.rank_id = rank_id
        self.config.device_num = device_num

    def build_parallel_config(self):
        """build parallel config"""
        recompute_config = TransformerRecomputeConfig(
            recompute=self.config.recompute,
            parallel_optimizer_comm_recompute=self.config.parallel_optimizer_comm_recompute,
            mp_comm_recompute=self.config.mp_comm_recompute,
            recompute_slice_activation=self.config.recompute_slice_activation)
        parallel_config = TransformerOpParallelConfig(
            data_parallel=self.config.data_parallel,
            model_parallel=self.config.model_parallel,
            pipeline_stage=self.config.pipeline_stage,
            micro_batch_num=self.config.micro_batch_num,
            expert_parallel=self.config.expert_parallel,
            vocab_emb_dp=self.config.vocab_emb_dp,
            optimizer_shard=self.config.optimizer_shard,
            gradient_aggregation_group=self.config.gradient_aggregation_group,
            recompute=recompute_config)
        parallel_config.moe_config = MoEConfig(
            expert_num=self.config.expert_num,
            capacity_factor=self.config.capacity_factor,
            aux_loss_factor=self.config.aux_loss_factor,
            num_experts_chosen=self.config.num_experts_chosen)
        return parallel_config

    def set_fused_kernel(self):
        """set fused kernel"""
        if self.config.fused_kernel:
            pwd = os.path.dirname(os.path.abspath(__file__))
            softmax_kernel_path = os.path.join(pwd, 'modules/fused_kernel/aot_scale_masked_softmax.cu')
            self.logger.info("Detect the fused_kernel True, "
                             "start to compile the cuda code. Cuda code path %s. "
                             "The attention in the mindspore will be replaced with softmax fused attention.",
                             softmax_kernel_path)

            override_attention(softmax_kernel_path)

    def load_checkpoint(self, net_with_loss):
        """
        Load model parameters from a given checkpoint.

        Args:
            net_with_loss (Cell): The network where the parameters will be loaded.

        Raises:
            TypeError: Argument is not a Cell.
        """
        if self.config.load_checkpoint_path == "" and self.config.save_checkpoint_path != "" \
                and self.config.checkpoint_prefix != "":
            self.config.load_checkpoint_path = get_newest_ckpt(self.config.save_checkpoint_path,
                                                               self.config.checkpoint_prefix)

        if self.config.load_checkpoint_path != "":
            if self.config.load_checkpoint_path.endswith('.ckpt'):
                self.logger.info("Start to load the ckpt from %s", self.config.load_checkpoint_path)
            else:
                self.config.load_checkpoint_path = get_newest_ckpt(self.config.load_checkpoint_path,
                                                                   self.config.checkpoint_prefix)
            ckpt = load_checkpoint(self.config.load_checkpoint_path)
            load_param_into_net(net_with_loss, ckpt)
        else:
            self.logger.info("training from scratch")

    def optimize_net_for_traning(self, net_with_loss):
        """optimize net"""
        micro_batch_interleaved_num = self.config.micro_batch_interleaved_num
        flatten_weights = self.config.flatten_weights
        if micro_batch_interleaved_num > 1:
            net_with_loss = MicroBatchInterleaved(net_with_loss, micro_batch_interleaved_num)
            self.logger.info(
                "Enabling the micro batch interleaved, the batch num is : %d.", micro_batch_interleaved_num)
        if flatten_weights:
            net_with_loss.flatten_weights()
            self.logger.info("Enabling the flatten_weights.")
        return net_with_loss

    def build_callback(self):
        """
        Build training callback including time monitor, loss call back, checkpoint call back, and so on.

        Output:
            callback: List, the callback functions
        """
        callback = [TimeMonitor(self.config.step_per_epoch), LossCallBack(self.config.step_per_epoch)]
        if not self.config.save_checkpoint:
            return callback
        self.logger.info(
            "Enable the checkpoint saving each %d steps. Integrated Save is False", self.config.step_per_epoch)
        config_ck = CheckpointConfig(save_checkpoint_steps=self.config.step_per_epoch,
                                     integrated_save=False,
                                     keep_checkpoint_max=1)
        ckpt_prefix = self.config.checkpoint_prefix if self.config.checkpoint_prefix \
                                                       is not None else self.config.auto_model
        ckpoint_cb = ModelCheckpoint(prefix=ckpt_prefix,
                                     directory=self.config.save_checkpoint_path + './ckpt_%d' % self.config.rank_id,
                                     config=config_ck)
        callback.append(ckpoint_cb)
        return callback

    def build_training_net(self, net, optim):
        """build training net"""
        # CPU doest not support overflow check, so should use fixed loss scale.
        update_cell = None
        if mindspore.get_context('device_target').lower() != 'cpu':
            loss_scale_manager = DynamicLossScaleManager(init_loss_scale=self.config.init_loss_scale_value,
                                                         scale_factor=self.config.scale_factor,
                                                         scale_window=self.config.scale_window)
            update_cell = loss_scale_manager.get_update_cell()

        if context.get_context("device_target") == "CPU":
            self.logger.info("For training on cpu, the loss scale will always be 1.")
            step_cell = TrainOneStepCell(net, optim)
            return step_cell

        if self.config.acc_step > 1:
            step_cell = TrainAccuStepsWithLossScaleCell(net, optim, update_cell)
        else:
            step_cell = TrainOneStepWithLossScaleCell(net, optim, update_cell)

        return step_cell

    def download_dataset(self):
        """get dataset from local or obs"""
        url = self.config.train_data_path if not self.config.get_eval_dataset else self.config.eval_data_path
        if url.startswith == "s3://":
            # copy data from the cloud to the /cache/Data
            cache_url = '/cache/Data/'
            self.logger.info("Find the data url %s startswith s3. Start to cache the data_path "
                             "to the local path %s.", url, cache_url)
            download_data(src_data_path=url, tgt_data_path=cache_url, rank=self.config.rank_id)
            self.logger.info("Data cache the finished.")
        else:
            cache_url = url
        return cache_url

    def build_dataset(self):
        """
        build dataset

        Raise:
            ValueError: self.config.auto_model is incorrect.
        """
        create_dataset = AutoClass.get_create_dataset_func(self.config.auto_model)
        if create_dataset is not None:
            return create_dataset(self.config)
        raise ValueError("invalid auto_model %s." % self.config.auto_model)

    def download_and_build_dataset(self):
        """
            Download and build the dataset. Download the data and build the dataset from config.

            Outputs:
                ds. The dataset class built from the config.

        """
        self.config.data_path = self.download_dataset()
        device_num = self.config.device_num
        rank_id = self.config.rank_id
        if context.get_auto_parallel_context('full_batch'):
            self.logger.info("Detect the full_batch import is true, modify the shard_num and shard_id to be 1 and 0."
                             "So each card will receive the same input data with "
                             "batch size: %d", self.config.global_batch_size)
            device_num = 1
            rank_id = 0

        self.config.dataset_device_num = device_num
        self.config.dataset_rank = rank_id
        self.config.dataset_batch_size = self.config.global_batch_size
        self.config.dataset_path = self.config.data_path
        self.config.dataset_schema_dir = None
        self.config.dataset_bucket_list = None

        ds = self.build_dataset()
        return ds

    def build_model_config(self):
        """
            Build the model configuration. Build the model config from the key "auto_model", which is set in "config".

            Outputs:
                model_config. The configuration of the model. It contains "GPTConfig", "BertConfig",
                              "TransformerConfig", "VitConfig" and "OPTConfig" now.

            Raises:
                ValueError: `auto_model` is an invalid key in config map.

        """
        model_config = AutoClass.get_config_class(self.config.auto_model)
        if model_config is not None:
            return model_config()
        raise ValueError("invalid auto_model %s." % self.config.auto_model)

    def check_and_build_model_config(self):
        """check and build model config"""
        if self.config.global_batch_size % self.config.micro_batch_interleaved_num != 0:
            raise ValueError(
                "global_batch_size:{} must be a multiple of micro_batch_interleaved:%d."
                % self.config.global_batch_size, self.config.micro_batch_interleaved)
        data_dp = 1
        if self.config.parallel_mode in (ParallelMode.AUTO_PARALLEL, ParallelMode.SEMI_AUTO_PARALLEL) and \
                not context.get_auto_parallel_context('full_batch'):
            data_dp = context.get_auto_parallel_context("device_num")
        model_config = self.build_model_config()

        for k, v in self.config.__dict__.items():
            if hasattr(model_config, k):
                if isinstance(v, str):
                    setattr(model_config, k, type(getattr(model_config, k))(v))
                else:
                    setattr(model_config, k, v)
        model_config.compute_dtype = self.config.compute_dtype
        model_config.batch_size = data_dp * self.config.global_batch_size // self.config.micro_batch_interleaved_num
        print("Model config are as follows:")
        print(json.dumps({k: str(v) for k, v in model_config.__dict__.items()}, indent=4))

        parallel_config = self.build_parallel_config()
        model_config.parallel_config = parallel_config

        return model_config

    def build_model(self, model_config):
        """
            Build the model with loss function. Build the model and its loss function from "model_config" class.

            Inputs:
                - **model_config** (class) - The configuration of the model. It contains "GPTConfig", "BertConfig",
                  "TransformerConfig", "VitConfig" and "OPTConfig" now.

            Outputs:
                network_with_loss. The network and the loss function from the config.

            Raises:
                ValueError: `auto_model` is an invalid key in NETWORK_WITH_LOSS_MAPPING.

        """
        network_with_loss = AutoClass.get_network_with_loss_class(self.config.auto_model)
        if network_with_loss is not None:
            return network_with_loss(model_config)
        raise ValueError("invalid auto_model %s." % self.config.auto_model)

    def build_lr(self):
        r"""
            Build learning rate with warm up and decay.
            Support cosine learning rate decay and polynomial learning rate decay.

            Args:
                epoch_size(int): The number of total training epochs.
                step_per_epoch(int): The number of total steps per epoch.
                warmup_step(int): The number of the warmup steps.
                start_lr(float): The learning rate at the beginning of the training.
                end_lr(float): The learning rate at the end of the training.

            Outputs:
                Tensor. The learning rate value of the current step.
        """
        total_steps = int(self.config.epoch_size * self.config.step_per_epoch)
        warmup_step = self.config.warmup_step if self.config.warmup_step > 0 else int(0.1 * total_steps)
        lr = LearningRate(learning_rate=float(self.config.start_lr),
                          end_learning_rate=float(self.config.end_lr),
                          warmup_steps=warmup_step,
                          decay_steps=total_steps)
        return lr

    def build_optimizer(self, net_with_loss):
        r"""
            Build the optimizer for training.

            Args:
                net(cell): the network model used in the training.
                lr: the built learning rate output by build_lr().
                optimizer_name(str): the name of the optimizer.
                args: the configuration of the optimizer.
                stage_num(int): the number of stages in the pipeline parallelization.
                fused(bool): whether to use operator fution.
                opt_offload(bool): whether to offload the optimizer to CPU.
                flatten_weights(bool): whether to fuse the optimizers. This is usually used together with opt_offload
                                       to reduce the number of the optimizers.

            Inputs:
                - **net_with_loss** (cell): the network model which contains the loss layer used in the training.

            Outputs:
                The built optimizer.
        """
        return build_optimizer(net=net_with_loss,
                               lr=self.build_lr(),
                               optimizer_name=self.config.optimizer,
                               args=None,
                               stage_num=1,
                               fused=True,
                               opt_offload=self.config.opt_offload,
                               flatten_weights=self.config.flatten_weights)

    def model_train(self, train_net, ds, callback):
        """model train"""
        self.logger.info("Start to compile the net and run.")
        if self.config.acc_step > 1:
            self.logger.info("Start to run gradient accumulation.")
            model = AccModel(train_net)
            # Note: If accumulation is enabled, it only supports dataset sink mode
            model.train(self.config.actual_epoch_num, ds, callbacks=callback, dataset_sink_mode=True)
        else:
            model = Model(train_net)
            model.train(self.config.actual_epoch_num, ds, callbacks=callback, sink_size=self.config.callback_step)

    def train(self):
        """Main training process"""
        self.set_context_env()
        self.set_auto_parallel_context_env()

        # This should be called before any cell construction
        self.set_fused_kernel()

        if self.config.seed:
            self.logger.info("Set seed:%s", self.config.seed)
            set_seed(self.config.seed)
        # Build the model with loss
        self.logger.info("Start to build model")
        model_config = self.check_and_build_model_config()
        model_config.is_training = True
        net_with_loss = self.build_model(model_config)
        self.logger.info("Build model finished")

        # load checkpoint
        self.load_checkpoint(net_with_loss)

        # optimize net
        net_with_loss = self.optimize_net_for_traning(net_with_loss)
        print_model_size(net_with_loss, self.logger)

        # download and build dataset
        self.logger.info("Start to build the dataset.")
        ds = self.download_and_build_dataset()

        self.config.step_per_epoch = ds.get_dataset_size()
        self.logger.info("Build dataset finished. The total dataset size is %s.", self.config.step_per_epoch)

        self.config.callback_step = self.config.sink_size if self.config.acc_step <= 1 else self.config.acc_step
        if self.config.callback_step > self.config.step_per_epoch:
            self.logger.info("The callback step %s is smaller than "
                             "step_per_epoch %s,"
                             "so change it to be %s", self.config.callback_step,
                             self.config.step_per_epoch, self.config.step_per_epoch)
            self.config.callback_step = self.config.step_per_epoch
        self.config.actual_epoch_num = int(
            self.config.epoch_size * self.config.step_per_epoch / self.config.callback_step)

        # build callback
        callback = self.build_callback()

        # build optimizer
        optimizer = self.build_optimizer(net_with_loss)

        # build training net
        train_net = self.build_training_net(net_with_loss, optimizer)

        # run training
        self.model_train(train_net, ds, callback)

    def model_predict(self, inference_net):
        """model predict"""
        model = Model(inference_net)
        if self.config.generate:
            self.logger.info("Start to generate the words:")
            generate_words(sample=self.config.input_samples,
                           predict_model=model,
                           opt=self.config)
        else:
            self.logger.info("Start to eval on the datasets.")
            self.config.get_eval_dataset = True
            # download and build dataset
            self.logger.info("Start to build the dataset.")
            ds = self.download_and_build_dataset()
            self.logger.info("Build dataset finished.")

            acc = get_acc(model, ds.create_tuple_iterator(), self.config)

            self.logger.info("The accuracy is %f", acc)

    def predict(self):
        """Main predict process"""
        # Build model
        self.logger.info("Start to build model")
        model_config = self.check_and_build_model_config()
        model_config.is_training = False
        inference_net = self.build_model(model_config)
        self.logger.info("Build model finished")

        # load checkpoint
        self.load_checkpoint(inference_net)

        # run predict
        self.model_predict(inference_net)


def parse_config(config):
    """parse user input arguments"""
    parser = argparse.ArgumentParser()
    _, unknown = parser.parse_known_args()
    for item in unknown:
        source = item.split('=')
        if len(source) != 2:
            raise ValueError("You should add = to the passed arguments. "
                             "For example --seed=123, the store_true action is not supported yet.")
        k, v = item.split('=')
        parser.add_argument(k)
    cli = parser.parse_args(unknown)
    for k, v in cli.__dict__.items():
        if hasattr(config, k) and isinstance(v, str):
            setattr(config, k, type(getattr(config, k))(_mapper_string_to_bool(v)))
        else:
            setattr(config, k, v)
    print("Training Arguments are as follows:")
    print(json.dumps({k: str(v) for k, v in config.__dict__.items()}, indent=4))
    if config.seed:
        print("set seed:", config.seed)
        set_seed(config.seed)


if __name__ == "__main__":
    training_config = TrainingConfig()
    parse_config(training_config)
    trainer = Trainer(training_config)
    trainer.train()
