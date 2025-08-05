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
import inspect
from dataclasses import dataclass
from typing import Optional, Union

from mindformers.core.config_args import BaseArgsConfig
from mindformers.core.callback import CheckpointMonitor
from mindformers.tools.register import MindFormerModuleType, MindFormerRegister


@dataclass
class CheckpointConfig(BaseArgsConfig):
    """MindFormers' save checkpoint config."""

    _support_kwargs = inspect.getfullargspec(CheckpointMonitor).args

    def __init__(
            self,
            prefix: str = 'mindformers',
            directory: str = None,
            save_checkpoint_steps: int = 1,
            keep_checkpoint_max: int = 1,
            integrated_save: bool = True,
            async_save: bool = False,
            saved_network: bool = None,
            **kwargs,
    ):
        super(CheckpointConfig, self).__init__(
            prefix=prefix,
            directory=directory,
            saved_network=saved_network,
            save_checkpoint_steps=save_checkpoint_steps,
            keep_checkpoint_max=keep_checkpoint_max,
            integrated_save=integrated_save,
            async_save=async_save,
            **kwargs
        )


@dataclass
class LRConfig(BaseArgsConfig):
    """MindFormers' learning rate schedule config."""
    _support_kwargs = [
        'type', 'max_lr', 'min_lr', 'decay_steps', 'decay_rate', 'power',
        'end_learning_rate', 'warmup_steps'
    ]

    def __init__(self, lr_type: str = None, **kwargs):
        if lr_type is not None:
            lr_schedule = MindFormerRegister.get_cls(
                module_type=MindFormerModuleType.LR, class_name=lr_type
            )
            self._support_kwargs.extend(
                inspect.getfullargspec(lr_schedule).args
            )
        super(LRConfig, self).__init__(type=lr_type, **kwargs)


@dataclass
class OptimizerConfig(BaseArgsConfig):
    """MindFormers' optimizer config."""
    _support_kwargs = [
        'type', 'learning_rate', 'beta1', 'beta2', 'eps', 'epsilon',
        'weight_decay', 'loss_scale', 'momentum'
    ]

    def __init__(
            self,
            optim_type: str = None,
            learning_rate: Optional[Union[BaseArgsConfig, float]] = None,
            **kwargs,
    ):
        if optim_type is not None:
            optimizer = MindFormerRegister.get_cls(
                module_type=MindFormerModuleType.OPTIMIZER,
                class_name=optim_type
            )
            self._support_kwargs.extend(inspect.getfullargspec(optimizer).args)

        super(OptimizerConfig, self).__init__(
            type=optim_type, learning_rate=learning_rate, **kwargs
        )


@dataclass
class WrapperConfig(BaseArgsConfig):
    """MindFormers' wrapper config."""
    _support_kwargs = ['type', 'sens', 'scale_sense']

    def __init__(self, wrapper_type: str = None, **kwargs):
        if wrapper_type is not None:
            wrapper = MindFormerRegister.get_cls(
                module_type=MindFormerModuleType.WRAPPER,
                class_name=wrapper_type
            )
            self._support_kwargs.extend(inspect.getfullargspec(wrapper).args)

        super(WrapperConfig, self).__init__(type=wrapper_type, **kwargs)


@dataclass
class DataLoaderConfig(BaseArgsConfig):
    """MindFormers' data loader config."""
    _support_kwargs = [
        'type',
        'dataset_dir',
        'num_samples',
        'num_parallel_workers',
        'shuffle',
        'sampler',
        'extensions',
        'class_indexing',
        'language_pair',
        'decode',
        'num_shards',
        'shard_id',
        'cache',
        'decrypt',
        'task',
        'usage',
        'test_set',
        'valid_set',
        'padded_sample',
        'num_padded',
    ]

    def __init__(
            self,
            dataloader_type: str = None,
            dataset_dir: str = None,
            **kwargs,
    ):
        if dataloader_type is not None:
            dataloader = MindFormerRegister.get_cls(
                MindFormerModuleType.DATASET_LOADER,
                class_name=dataloader_type
            )
            self._support_kwargs.extend(
                inspect.getfullargspec(dataloader).args
            )
        super(DataLoaderConfig, self).__init__(
            type=dataloader_type, dataset_dir=dataset_dir, **kwargs
        )
