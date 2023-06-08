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
from typing import Optional, Union
from dataclasses import dataclass
import inspect

from mindformers.core.callback import CheckpointMointor
from mindformers.tools.register import MindFormerRegister, \
    MindFormerModuleType


__all__ = ['BaseArgsConfig', 'RunnerConfig', 'DatasetConfig', 'DataLoaderConfig',
           'ConfigArguments', 'ContextConfig', 'CloudConfig', 'CheckpointConfig',
           'ParallelContextConfig', 'OptimizerConfig', 'LRConfig', 'WrapperConfig']


@dataclass
class BaseArgsConfig:
    """Base Argument config."""
    _support_kwargs = []

    def __init__(self, **kwargs):
        if kwargs is not None:
            for key, value in kwargs.items():
                assert key in self._support_kwargs, \
                    f"The Config Class support input argument is {self._support_kwargs}, but get {key}"
                if value is None:
                    continue
                if isinstance(value, BaseArgsConfig):
                    value = value.__dict__
                self.__setattr__(key, value)


@dataclass
class ContextConfig(BaseArgsConfig):
    """Context Config."""
    _support_kwargs = [
        'mode', 'precompile_only', 'device_target', 'device_id', 'save_graphs',
        'save_graphs_path', 'enable_dump', 'auto_tune_mode',
        'save_dump_path', 'enable_reduce_precision', 'variable_memory_max_size',
        'enable_profiling', 'profiling_options', 'enable_auto_mixed_precision',
        'enable_graph_kernel', 'reserve_class_name_in_scope', 'check_bprop',
        'max_device_memory', 'print_file_path', 'enable_sparse', 'max_call_depth',
        'env_config_path', 'graph_kernel_flags', 'save_compile_cache', 'runtime_num_threads',
        'load_compile_cache', 'grad_for_scalar', 'pynative_synchronize', 'mempool_block_size'
    ]

    def __init__(self,
                 mode: Optional[Union[int, str]] = 0,
                 device_target: str = "Ascend",
                 device_id: int = int(os.getenv('DEVICE_ID', '0')),
                 save_graphs: bool = False, save_graphs_path: str = ".", **kwargs):
        super(ContextConfig, self).__init__(mode=mode,
                                            device_id=device_id,
                                            device_target=device_target,
                                            save_graphs=save_graphs,
                                            save_graphs_path=save_graphs_path, **kwargs)


@dataclass
class ParallelContextConfig(BaseArgsConfig):
    """Parallel Context Config."""

    _support_kwargs = [
        'device_num', 'global_rank', 'gradients_mean', 'gradient_fp32_sync', 'parallel_mode',
        'auto_parallel_search_mode', 'search_mode', 'parameter_broadcast', 'strategy_ckpt_load_file',
        'strategy_ckpt_save_file', 'full_batch', 'enable_parallel_optimizer', 'enable_alltoall',
        'all_reduce_fusion_config', 'pipeline_stages', 'grad_accumulation_step',
        'parallel_optimizer_config', 'comm_fusion'
    ]

    def __init__(self,
                 parallel_mode: str = 'STAND_ALONE',
                 device_num: int = int(os.getenv('RANK_SIZE', '1')),
                 gradients_mean: bool = False, **kwargs):
        super(ParallelContextConfig, self).__init__(parallel_mode=parallel_mode,
                                                    device_num=device_num,
                                                    gradients_mean=gradients_mean, **kwargs)


@dataclass
class CloudConfig(BaseArgsConfig):
    """Cloud Config For ModelArts."""
    _support_kwargs = [
        'obs_path', 'root_path', 'rank_id', 'upload_frequence',
        'keep_last', 'retry', 'retry_time'
    ]

    def __init__(self,
                 obs_path: str = None,
                 root_path: str = '/cache',
                 rank_id: int = None,
                 upload_frequence: int = 1,
                 keep_last: bool = False, **kwargs):
        super(CloudConfig, self).__init__(obs_path=obs_path,
                                          root_path=root_path,
                                          rank_id=rank_id,
                                          upload_frequence=upload_frequence,
                                          keep_last=keep_last, **kwargs)


@dataclass
class RunnerConfig(BaseArgsConfig):
    """MindFormers' config when running model."""

    _support_kwargs = [
        'epochs', 'batch_size', 'sink_mode', 'sink_size', 'initial_epoch',
        'has_trained_epoches', 'has_trained_steps', 'image_size', 'num_classes',
        'sink_size',
    ]

    def __init__(self,
                 epochs: int = None, batch_size: int = None,
                 sink_mode: bool = None, sink_size: int = None,
                 initial_epoch: int = None, has_trained_epoches: int = None,
                 has_trained_steps: int = None, **kwargs):
        super(RunnerConfig, self).__init__(epochs=epochs,
                                           batch_size=batch_size,
                                           sink_mode=sink_mode,
                                           sink_size=sink_size,
                                           initial_epoch=initial_epoch,
                                           has_trained_steps=has_trained_steps,
                                           has_trained_epoches=has_trained_epoches, **kwargs)


@dataclass
class CheckpointConfig(BaseArgsConfig):
    """MindFormers' save checkpoint config."""

    _support_kwargs = inspect.getfullargspec(CheckpointMointor).args

    def __init__(self,
                 prefix: str = 'mindformers',
                 directory: str = None,
                 save_checkpoint_steps: int = 1,
                 keep_checkpoint_max: int = 1,
                 integrated_save: bool = True,
                 async_save: bool = False,
                 saved_network: bool = None, **kwargs):
        super(CheckpointConfig, self).__init__(prefix=prefix,
                                               directory=directory,
                                               saved_network=saved_network,
                                               save_checkpoint_steps=save_checkpoint_steps,
                                               keep_checkpoint_max=keep_checkpoint_max,
                                               integrated_save=integrated_save,
                                               async_save=async_save, **kwargs)


@dataclass
class LRConfig(BaseArgsConfig):
    """MindFormers' learning rate schedule config."""
    _support_kwargs = [
        'type', 'max_lr', 'min_lr', 'decay_steps', 'decay_rate',
        'power', 'end_learning_rate', 'warmup_steps'
    ]

    def __init__(self, lr_type: str = None, **kwargs):
        if lr_type is not None:
            lr_schedule = MindFormerRegister.get_cls(
                module_type=MindFormerModuleType.LR, class_name=lr_type)
            self._support_kwargs.extend(inspect.getfullargspec(lr_schedule).args)
        super(LRConfig, self).__init__(type=lr_type, **kwargs)


@dataclass
class OptimizerConfig(BaseArgsConfig):
    """MindFormers' optimizer config."""
    _support_kwargs = [
        'type', 'learning_rate', 'beta1', 'beta2', 'eps', 'epsilon',
        'weight_decay', 'loss_scale', 'momentum'
    ]

    def __init__(self, optim_type: str = None,
                 learning_rate: Optional[Union[BaseArgsConfig, float]] = None,
                 **kwargs):
        if optim_type is not None:
            optimizer = MindFormerRegister.get_cls(
                module_type=MindFormerModuleType.OPTIMIZER, class_name=optim_type)
            self._support_kwargs.extend(inspect.getfullargspec(optimizer).args)

        super(OptimizerConfig, self).__init__(type=optim_type,
                                              learning_rate=learning_rate,
                                              **kwargs)


@dataclass
class WrapperConfig(BaseArgsConfig):
    """MindFormers' wrapper config."""
    _support_kwargs = [
        'type', 'sens', 'scale_sense'
    ]

    def __init__(self, wrapper_type: str = None, **kwargs):
        if wrapper_type is not None:
            wrapper = MindFormerRegister.get_cls(
                module_type=MindFormerModuleType.WRAPPER, class_name=wrapper_type)
            self._support_kwargs.extend(inspect.getfullargspec(wrapper).args)

        super(WrapperConfig, self).__init__(type=wrapper_type, **kwargs)


@dataclass
class DataLoaderConfig(BaseArgsConfig):
    """MindFormers' data loader config."""
    _support_kwargs = [
        'type', 'dataset_dir', 'num_samples', 'num_parallel_workers',
        'shuffle', 'sampler', 'extensions', 'class_indexing', 'language_pair',
        'decode', 'num_shards', 'shard_id', 'cache', 'decrypt', 'task', 'usage',
        'test_set', 'valid_set', 'padded_sample', 'num_padded'
    ]

    def __init__(self, dataloader_type: str = None, dataset_dir: str = None, **kwargs):
        if dataloader_type is not None:
            dataloader = MindFormerRegister.get_cls(
                MindFormerModuleType.DATASET_LOADER, class_name=dataloader_type)
            self._support_kwargs.extend(inspect.getfullargspec(dataloader).args)
        super(DataLoaderConfig, self).__init__(type=dataloader_type,
                                               dataset_dir=dataset_dir,
                                               **kwargs)


@dataclass
class DatasetConfig(BaseArgsConfig):
    """MindFormers' dataset config."""
    _support_kwargs = [
        'data_loader', 'input_columns', 'output_columns', 'column_order',
        'drop_remainder', 'repeat', 'batch_size', 'image_size', 'num_parallel_workers',
        'per_batch_map', 'python_multiprocessing', 'max_rowsize', 'cache', 'offload'
    ]

    def __init__(self,
                 data_loader: Optional[Union[dict, BaseArgsConfig]] = None,
                 input_columns: Optional[Union[str, list]] = None,
                 output_columns: Optional[Union[str, list]] = None,
                 column_order: Optional[Union[str, list]] = None,
                 drop_remainder: bool = True, repeat: int = 1, batch_size: int = None,
                 image_size: Optional[Union[int, list, tuple]] = None, **kwargs):
        super(DatasetConfig, self).__init__(data_loader=data_loader,
                                            batch_size=batch_size,
                                            image_size=image_size,
                                            repeat=repeat,
                                            input_columns=input_columns,
                                            output_columns=output_columns,
                                            column_order=column_order,
                                            drop_remainder=drop_remainder, **kwargs)


@dataclass
class ConfigArguments(BaseArgsConfig):
    """MindFormers' config arguments."""
    _support_kwargs = [
        'output_dir', 'profile', 'auto_tune', 'filepath_prefix', 'autotune_per_step',
        'train_dataset', 'eval_dataset', 'predict_dataset', 'runner_config', 'optimizer',
        'lr_schedule', 'save_checkpoint', 'cloud_config', 'seed', 'runner_wrapper'
    ]

    def __init__(self, output_dir: str = './output', profile: bool = False,
                 auto_tune: bool = False, filepath_prefix: str = './autotune',
                 autotune_per_step: int = 10, seed: int = None,
                 train_dataset: Optional[Union[dict, BaseArgsConfig]] = None,
                 eval_dataset: Optional[Union[dict, BaseArgsConfig]] = None,
                 runner_config: Optional[Union[dict, BaseArgsConfig]] = None,
                 optimizer: Optional[Union[dict, BaseArgsConfig]] = None,
                 runner_wrapper: Optional[Union[dict, BaseArgsConfig]] = None,
                 lr_schedule: Optional[Union[dict, BaseArgsConfig]] = None,
                 save_checkpoint: Optional[Union[dict, BaseArgsConfig]] = None,
                 cloud_config: Optional[Union[dict, BaseArgsConfig]] = None):
        super(ConfigArguments, self).__init__(output_dir=output_dir,
                                              profile=profile,
                                              auto_tune=auto_tune,
                                              seed=seed,
                                              filepath_prefix=filepath_prefix,
                                              autotune_per_step=autotune_per_step,
                                              train_dataset=train_dataset,
                                              eval_dataset=eval_dataset,
                                              runner_config=runner_config,
                                              optimizer=optimizer,
                                              runner_wrapper=runner_wrapper,
                                              lr_schedule=lr_schedule,
                                              save_checkpoint=save_checkpoint,
                                              cloud_config=cloud_config)
