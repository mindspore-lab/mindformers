# Copyright 2025 Huawei Technologies Co., Ltd
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
"""test get_optimizer_grouped_parameters api."""

import pytest
import numpy as np

import mindspore as ms
from mindspore import Tensor, Parameter, nn

from mindformers.trainer.optimizer_grouped_parameters import get_optimizer_grouped_parameters
from mindformers.core.lr.lr_schedule import LinearWithWarmUpLR
from mindformers.tools.register.config import MindFormerConfig


class Bias(nn.Cell):
    """ A simple bias module for test. """

    def __init__(self):
        super().__init__(auto_prefix=True)
        self.bias = Parameter(Tensor([0.1], ms.int32), name="bias", requires_grad=True)

    def construct(self, x):
        return x + self.bias


class Net(nn.Cell):
    """ A simple net for test. """

    def __init__(self):
        super().__init__(auto_prefix=True)
        self.weight = Parameter(Tensor(np.random.rand(128, 512), ms.float32), name="weight", requires_grad=True)
        self.value = Parameter(Tensor([2], ms.int32), name="value", requires_grad=True)
        self.model = Bias()

    def construct(self, x):
        x = x * self.weight * self.value
        output = self.model(x)
        return output


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_grouped_params():
    """
    Feature: get_optimizer_grouped_parameters api
    Description: Test get_optimizer_grouped_parameters function
    Expectation: No exception.
    """

    model = Net()
    weight_decay = 0.01
    dynamic_lr_schedule = LinearWithWarmUpLR(
        learning_rate=0.001,
        total_steps=100,
        warmup_steps=0,
        warmup_lr_init=0.0,
        warmup_ratio=None
    )

    grouped_params = get_optimizer_grouped_parameters(
        model=model,
        weight_decay=weight_decay,
        dynamic_lr_schedule=dynamic_lr_schedule,
        layer_scale=False,
        layer_decay=1.0,
        # use for ("PmaAdamW", "FusedPmaAdamW")
        optimizer_type='AdamW',
        model_params=None
    )

    target_dict = [
        {'weight_decay': 0.01, 'params': [model.weight]},
        {'weight_decay': 0.0, 'params': [model.value, model.model.bias]},
    ]
    assert grouped_params == target_dict, f"Get params {grouped_params}, but should be {target_dict}."


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_grouped_params_with_grouped_lr():
    """
    Feature: get_optimizer_grouped_parameters api
    Description: Test get_optimizer_grouped_parameters function with grouped lr scheduler
    Expectation: No exception.
    """

    model = Net()
    weight_decay = 0.01
    dynamic_lr_schedule = LinearWithWarmUpLR(
        learning_rate=0.001,
        total_steps=100,
        warmup_steps=0,
        warmup_lr_init=0.0,
        warmup_ratio=None
    )

    lr_config = MindFormerConfig(**{
        'type':'LinearWithWarmUpLR',
        'params': ['value*'],
        'learning_rate': 1.e-6,
        'warmup_steps': 0,
        'total_steps': -1
    })
    lr_scheduler = LinearWithWarmUpLR(
        learning_rate=1.e-6,
        total_steps=100,
        warmup_steps=0,
        warmup_lr_init=0.0,
        warmup_ratio=None
    )
    grouped_lr_schedule = [{
        'params': lr_config.params,
        'lr_scheduler': lr_scheduler,
        'lr_config': lr_config
    }]

    grouped_params = get_optimizer_grouped_parameters(
        model=model,
        weight_decay=weight_decay,
        dynamic_lr_schedule=dynamic_lr_schedule,
        layer_scale=False,
        layer_decay=1.0,
        # use for ("PmaAdamW", "FusedPmaAdamW")
        optimizer_type='AdamW',
        model_params=None,
        grouped_lr_schedule=grouped_lr_schedule,
    )

    target_dict = [
        {'weight_decay': 0.01, 'params': [model.weight]},
        {'weight_decay': 0.0, 'params': [model.value], 'lr': lr_scheduler},
        {'weight_decay': 0.0, 'params': [model.model.bias]}
    ]
    assert grouped_params == target_dict, f"Get params {grouped_params}, but should be {target_dict}."


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_grouped_params_with_invalid_group():
    """
    Feature: get_optimizer_grouped_parameters api
    Description: Test get_optimizer_grouped_parameters function with invalid group
    Expectation: ValueError.
    """

    model = Net()
    weight_decay = 0.01
    dynamic_lr_schedule = LinearWithWarmUpLR(
        learning_rate=0.001,
        total_steps=100,
        warmup_steps=0,
        warmup_lr_init=0.0,
        warmup_ratio=None
    )

    lr_config = MindFormerConfig(**{
        'type':'LinearWithWarmUpLR',
        'params': ['alpha*'],
        'learning_rate': 1.e-6,
        'warmup_steps': 0,
        'total_steps': -1
    })
    lr_scheduler = LinearWithWarmUpLR(
        learning_rate=1.e-6,
        total_steps=100,
        warmup_steps=0,
        warmup_lr_init=0.0,
        warmup_ratio=None
    )
    grouped_lr_schedule = [{
        'params': lr_config.params,
        'lr_scheduler': lr_scheduler,
        'lr_config': lr_config
    }]

    with pytest.raises(ValueError):
        get_optimizer_grouped_parameters(
            model=model,
            weight_decay=weight_decay,
            dynamic_lr_schedule=dynamic_lr_schedule,
            layer_scale=False,
            layer_decay=1.0,
            # use for ("PmaAdamW", "FusedPmaAdamW")
            optimizer_type='AdamW',
            model_params=None,
            grouped_lr_schedule=grouped_lr_schedule,
        )
