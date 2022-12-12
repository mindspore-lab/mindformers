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
python tests/ut/test_trainer/test_trainer_auto.py
"""
import pytest
from mindformers.trainer import Trainer
from mindformers.common.context import init_context
from mindformers.trainer.config_args import ConfigArguments, \
    DatasetConfig, RunnerConfig, ContextConfig

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
    context_config = ContextConfig(device_id=4, device_target='Ascend', mode=0)
    init_context(use_parallel=False, context_config=context_config)

    # 额外的设定，为了CI门禁测试小数据集，需要更改以下参数，正常模型使用时无需指定
    runner_config = RunnerConfig(epochs=10, batch_size=2, image_size=224)  # 运行超参
    train_dataset_config = DatasetConfig(batch_size=2)
    config = ConfigArguments(runner_config=runner_config, train_dataset=train_dataset_config)

    # example 1: 输入标准的数据集, 自动创建已有任务和模型的训练
    mim_trainer = Trainer(
        task_name='masked_image_modeling',
        model='mae_vit_base_p16',
        train_dataset="/home/workspace/mindformers/vit/train",
        config=config)  # 为了CI门禁测试小数据集而新增的config， 正常使用无需指定该部分内容
    #  "/home/jenkins/qianjiahong/mindformers/transformer/test/vit/train"
    mim_trainer.train()
    # resume, default from last checkpoint, 断点续训功能
    # mim_trainer_a.train(resume_from_checkpoint=True, initial_epoch=100)
