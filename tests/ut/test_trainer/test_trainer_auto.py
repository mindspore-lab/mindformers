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
from mindformers.trainer import Trainer
from mindformers.common.context import init_context
from mindformers.trainer.config_args import ContextConfig


def test_trainer_train_auto():
    """
    Feature: Auto Create Trainer.
    Description: Test Trainer API to train.
    Expectation: TypeError
    """
    context_config = ContextConfig(device_id=1, device_target='Ascend', mode=0)
    init_context(seed=2022, use_parallel=False, context_config=context_config)
    # example 1: 输入标准的数据集, 自动创建已有任务和模型的训练
    mim_trainer_a = Trainer(task_name='masked_image_modeling',
                            model='mae_vit_base_p16',
                            train_dataset="/data/imageNet-1k/train")
    mim_trainer_a.train()
    # resume, default from last checkpoint, 断点续训功能
    # mim_trainer_a.train(resume_from_checkpoint=True, initial_epoch=100)
