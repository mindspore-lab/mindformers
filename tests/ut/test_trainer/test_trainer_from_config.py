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
python tests/ut/test_trainer/test_trainer_from_config.py
"""
from mindformers.trainer import Trainer
from mindformers.models import MaeModel
from mindformers.common.context import init_context
from mindformers.trainer.config_args import ConfigArguments, \
    OptimizerConfig, DatasetConfig, DataLoaderConfig, RunnerConfig, \
    ContextConfig, LRConfig


def test_trainer_train_from_config():
    """
    Feature: Create Trainer From Config
    Description: Test Trainer API to train from config
    Expectation: TypeError
    """
    # example 2: 自定义视觉自监督任务的 模型和超参
    # 初始化环境
    context_config = ContextConfig(device_id=4, device_target='Ascend', mode=0)
    init_context(use_parallel=False, context_config=context_config)

    runner_config = RunnerConfig(epochs=10, batch_size=2, image_size=224)  # 运行超参
    lr_schedule_config = LRConfig(lr_type='WarmUpLR', learning_rate=0.001, warmup_steps=10)
    # 默认支持AdamWeightDecay相应参数
    optim_config = OptimizerConfig(optim_type='Adam', beta1=0.009, learning_rate=lr_schedule_config)
    data_loader_config = DataLoaderConfig(
        dataset_dir="/home/jenkins/qianjiahong/mindformers/transformer/test/vit/train")   # 数据加载参数设定
    # 数据集超参
    train_dataset_config = DatasetConfig(data_loader=data_loader_config,
                                         input_columns=["image"],
                                         output_columns=["image"],
                                         column_order=["image"],
                                         batch_size=2,
                                         image_size=224)
    # mae_model = MaeModel.from_config("mae_vit_base_p16") # 获取模型实例
    # mae_model = MaeModel.from_pretrain("mae_vit_base_p16") # 获取载入预训练权重的模型实例, 可用于finetune
    # 统一超参配置接口
    config = ConfigArguments(seed=2022, runner_config=runner_config,
                             train_dataset=train_dataset_config, optimizer=optim_config)
    mae_model = MaeModel()
    mim_trainer = Trainer(task_name='masked_image_modeling', model=mae_model, config=config)
    mim_trainer.train(resume_from_checkpoint=False)
