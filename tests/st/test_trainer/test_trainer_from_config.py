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
Test module for testing the interface used for mindformers.
How to run this:
pytest tests/st/test_trainer/test_trainer_from_config.py
"""
import pytest
import numpy as np

from mindspore.dataset import GeneratorDataset
from mindspore.nn import DynamicLossScaleUpdateCell

from mindformers.trainer import Trainer
from mindformers.models import MaeModel
from mindformers.trainer.config_args import ConfigArguments, \
    OptimizerConfig, RunnerConfig, LRConfig, WrapperConfig


class MyDataLoader:
    """Self-Define DataLoader."""
    def __init__(self):
        self._data = [np.zeros((3, 224, 224), np.float32) for _ in range(64)]

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_trainer_train_from_config():
    """
    Feature: Create Trainer From Config
    Description: Test Trainer API to train from config
    Expectation: TypeError
    """
    runner_config = RunnerConfig(epochs=10, batch_size=2, image_size=224)  # 运行超参
    lr_schedule_config = LRConfig(lr_type='WarmUpLR', learning_rate=0.001, warmup_steps=10)
    optim_config = OptimizerConfig(optim_type='Adam', beta1=0.009, learning_rate=lr_schedule_config)
    loss_scale = DynamicLossScaleUpdateCell(loss_scale_value=2**12, scale_factor=2, scale_window=1000)
    wrapper_config = WrapperConfig(wrapper_type='TrainOneStepWithLossScaleCell', scale_sense=loss_scale)

    dataset = GeneratorDataset(source=MyDataLoader(), column_names='image')
    dataset = dataset.batch(batch_size=2)

    config = ConfigArguments(seed=2022, runner_config=runner_config,
                             optimizer=optim_config, runner_wrapper=wrapper_config)
    mae_model = MaeModel()
    mim_trainer = Trainer(task_name='masked_image_modeling',
                          model=mae_model,
                          config=config,
                          train_dataset=dataset)
    mim_trainer.train(resume_from_checkpoint=False)
