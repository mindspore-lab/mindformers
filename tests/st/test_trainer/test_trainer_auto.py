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
pytest tests/st/test_trainer/test_trainer_auto.py
"""
import pytest
import numpy as np

from mindspore.dataset import GeneratorDataset

from mindformers.trainer import Trainer
from mindformers.trainer.config_args import ConfigArguments, \
    RunnerConfig
from mindformers.tools.logger import logger


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
def test_trainer_train_auto():
    """
    Feature: Auto Create Trainer.
    Description: Test Trainer API to train.
    Expectation: TypeError
    """
    runner_config = RunnerConfig(epochs=10, batch_size=2, image_size=224)  # 运行超参
    config = ConfigArguments(runner_config=runner_config)

    dataset = GeneratorDataset(source=MyDataLoader(), column_names='image')
    dataset = dataset.batch(batch_size=2)

    # example 1: 输入标准的数据集, 自动创建已有任务和模型的训练
    mim_trainer = Trainer(
        task='masked_image_modeling',
        model='mae_vit_base_p16',
        train_dataset=dataset,
        config=config)
    mim_trainer.train()


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_trainer_predict_auto():
    """
    Feature: Auto Create Trainer.
    Description: Test Trainer API to train.
    Expectation: TypeError
    """
    zero_shot_image_cls_trainer = Trainer(
        task='zero_shot_image_classification',
        model='clip_vit_b_32')
    image = np.random.random((224, 224, 3))
    predict_result = zero_shot_image_cls_trainer.predict(
        input_data=image,
        candidate_labels=["sunflower", "tree", "dog", "cat", "toy"]
    )
    logger.info(predict_result)
