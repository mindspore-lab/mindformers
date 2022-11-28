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
"""MindFormers' Config API."""
import os
from typing import List, Optional, Union
from dataclasses import dataclass


@dataclass
class ContextConfig:
    """Context Config."""
    mode: Optional[Union[int, str]] = 0
    device_target: str = "Ascend"
    enable_graph_kernel: bool = False
    graph_kernel_flags: str = "--opt_level=0"
    max_call_depth: int = 10000
    save_graphs: bool = False
    device_id: int = int(os.getenv('DEVICE_ID', '0'))
    max_device_memory: str = "1024GB"
    save_graphs_path: str = "."
    enable_dump: bool = False
    save_dump_path: str = "."
    precompile_only: bool = False
    reserve_class_name_in_scope: bool = True
    pynative_synchronize: bool = False
    enable_reduce_precision: bool = False
    auto_tune_mode: str = 'NO_TUNE'
    check_bprop: bool = False
    enable_sparse: bool = False
    grad_for_scalar: bool = False
    enable_compile_cache: bool = False
    compile_cache_path: str = "."
    runtime_num_threads: int = 30


@dataclass
class ParallelContextConfig:
    """Parallel Context Config."""
    device_num: int = int(os.getenv('RANK_SIZE', '1'))
    gradient_fp32_sync: bool = False
    global_rank: int = int(os.getenv('RANK_ID', '0'))
    loss_repeated_mean: bool = True
    gradients_mean: bool = False
    search_mode: str = 'dynamic_programming'
    parallel_mode: str = 'STAND_ALONE'
    parameter_broadcast: bool = False
    strategy_ckpt_load_file: str = ""
    strategy_ckpt_save_file: str = ""
    enable_parallel_optimizer: bool = False
    dataset_strategy: Union[str, tuple] = "data_parallel"
    pipeline_stages: int = 1
    enable_alltoall: bool = False
    grad_accumulation_step: int = 1
    auto_parallel_search_mode: str = "search_mode"


@dataclass
class CloudConfig:
    obs_path: str = None
    root_path: str = '/cache'
    rank_id: int = None
    upload_frequence: int = 1
    keep_last: bool = True


@dataclass
class RunnerConfig:
    """MindFormers' config when running model."""
    epochs: int = 100
    batch_size: int = 64
    image_size: Optional[Union[int, List, tuple]] = 224
    sink_mode: bool = True
    per_epoch_size: int = 0
    initial_epoch: int = 0
    has_trained_epoches: int = 0
    has_trained_steps: int = 0
    checkpoint_name_or_path: str = None


@dataclass
class CheckpointConfig:
    """MindFormers' save checkpoint config."""
    type: str = 'CheckpointMointor'
    prefix: str = 'CKP'
    directory: str = None
    config: dict = None
    save_checkpoint_steps: int = 1
    save_checkpoint_seconds: int = 0
    keep_checkpoint_max: int = 5
    keep_checkpoint_per_n_minutes: int = 0
    integrated_save: bool = True
    async_save: bool = False
    saved_network: bool = None
    append_info: dict = None
    enc_key: dict = None
    enc_mode: str = 'AES-GCM'
    exception_save: bool = False


@dataclass
class OptimizerConfig:
    """MindFormers' optimizer config."""
    type: str = 'AdamWeightDecay'
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    learning_rate: float = 0.01
    weight_decay: float = 0.05


@dataclass
class LRConfig:
    """MindFormers' learning rate schedule config."""
    type: str = 'CosineDecayLR'
    max_lr: float = 0.00015
    min_lr: float = 0.
    decay_steps: int = 100


@dataclass
class DataLoaderConfig:
    """MindFormers' data loader config."""
    type: str = 'ImageFolderDataset'
    dataset_dir: str = None
    num_parallel_workers: int = 8
    shuffle: bool = True
    num_shards: int = int(os.getenv('RANK_SIZE', '1'))
    shard_id: int = int(os.getenv('DEVICE_ID', '0'))


@dataclass
class DatasetConfig:
    """MindFormers' dataset config."""
    input_columns: Optional[Union[str, list]] = None
    output_columns: Optional[Union[str, list]] = None
    column_order: Optional[Union[str, list]] = None
    num_parallel_workers: int = 8
    python_multiprocessing: bool = False
    drop_remainder: bool = True
    repeat: int = 1
    numa_enable: bool = False
    prefetch_size: int = 30
    batch_size: int = RunnerConfig.batch_size
    image_size: Optional[Union[int, list, tuple]] = RunnerConfig.image_size


@dataclass
class ConfigArguments:
    """MindFormers' config arguments."""

    def __init__(self, output_dir: str = './output', profile: bool = False,
                 seed: int = 0, auto_tune: bool = False, filepath_prefix: str = './autotune',
                 autotune_per_step: int = 10, data_loader: Optional[Union[dict, DataLoaderConfig]] = None,
                 train_dataset: Optional[Union[dict, DatasetConfig]] = None,
                 eval_dataset: Optional[Union[dict, DatasetConfig]] = None,
                 predict_dataset: Optional[Union[dict, DatasetConfig]] = None,
                 runner_config: Optional[Union[dict, RunnerConfig]] = None,
                 optimizer: Optional[Union[dict, OptimizerConfig]] = None,
                 lr_schedule: Optional[Union[dict, LRConfig]] = None,
                 save_checkpoint: Optional[Union[dict, CheckpointConfig]] = None,
                 cloud_config: Optional[Union[dict, CloudConfig]] = None):
        self.output_dir = output_dir
        self.profile = profile
        self.auto_tune = auto_tune
        self.filepath_prefix = filepath_prefix
        self.autotune_per_step = autotune_per_step
        self.seed = seed
        self.data_loader = data_loader.__dict__ \
            if isinstance(data_loader, DataLoaderConfig) else data_loader
        self.train_dataset = train_dataset.__dict__ \
            if isinstance(train_dataset, DatasetConfig) else train_dataset
        self.eval_dataset = eval_dataset.__dict__ \
            if isinstance(eval_dataset, DatasetConfig) else eval_dataset
        self.predict_dataset = predict_dataset.__dict__ \
            if isinstance(predict_dataset, DatasetConfig) else predict_dataset
        self.runner_config = runner_config.__dict__ \
            if isinstance(runner_config, RunnerConfig) else runner_config
        self.optimizer = optimizer.__dict__ \
            if isinstance(optimizer, OptimizerConfig) else optimizer
        self.lr_schedule = lr_schedule.__dict__ \
            if isinstance(lr_schedule, LRConfig) else lr_schedule
        self.callbacks = save_checkpoint.__dict__ \
            if isinstance(save_checkpoint, CheckpointConfig) else save_checkpoint
        self.aicc_config = cloud_config.__dict__ \
            if isinstance(cloud_config, CloudConfig) else cloud_config
        if data_loader is None:
            del self.data_loader
        if train_dataset is None:
            del self.train_dataset
        if eval_dataset is None:
            del self.eval_dataset
        if predict_dataset is None:
            del self.predict_dataset
        if runner_config is None:
            del self.runner_config
        if optimizer is None:
            del self.optimizer
        if lr_schedule is None:
            del self.lr_schedule
        if save_checkpoint is None:
            del self.callbacks
        if cloud_config is None:
            del self.aicc_config
