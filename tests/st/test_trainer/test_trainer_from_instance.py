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
pytest tests/st/test_trainer/test_trainer_from_instance.py
"""
import os
import pytest
import numpy as np
from PIL import Image
from mindspore.nn import AdamWeightDecay, WarmUpLR, \
    DynamicLossScaleUpdateCell, TrainOneStepWithLossScaleCell
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindformers.mindformer_book import MindFormerBook
from mindformers.tools.register.config import MindFormerConfig
from mindformers.dataset.build_dataset import build_dataset
from mindformers.trainer import Trainer
from mindformers.models import ViTMAEForPreTraining
from mindformers.trainer.config_args import ConfigArguments, \
    RunnerConfig


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_trainer_train_from_instance():
    """
    Feature: Create Trainer From Instance
    Description: Test Trainer API to train from self-define instance API.
    Expectation: TypeError
    """
    runner_config = RunnerConfig(epochs=2, batch_size=8, image_size=224, sink_mode=True, per_epoch_size=1)
    config = ConfigArguments(seed=2022, runner_config=runner_config)

    mae_model_with_loss = ViTMAEForPreTraining()

    project_path = MindFormerBook.get_project_path()

    config_path = os.path.join(
        project_path, "configs", "mae", "run_mae_vit_base_p16_224_800ep.yaml"
    )
    dataset_config = MindFormerConfig(config_path)

    new_dataset_dir = make_local_directory(dataset_config)
    make_dataset(new_dataset_dir, num=16)

    dataset_config.train_dataset.data_loader.dataset_dir = new_dataset_dir
    dataset_config.train_dataset_task.dataset_config.data_loader.dataset_dir = new_dataset_dir

    dataset = build_dataset(dataset_config.train_dataset_task)

    lr_schedule = WarmUpLR(learning_rate=0.001, warmup_steps=100)
    optimizer = AdamWeightDecay(beta1=0.009, beta2=0.999,
                                learning_rate=lr_schedule,
                                params=mae_model_with_loss.trainable_params())

    loss_cb = LossMonitor(per_print_times=2)
    time_cb = TimeMonitor()
    callbacks = [loss_cb, time_cb]

    mim_trainer = Trainer(task='masked_image_modeling',
                          model=mae_model_with_loss,  # include loss compute
                          args=config,
                          optimizers=optimizer,
                          train_dataset=dataset,
                          callbacks=callbacks)

    mim_trainer.train(resume_or_finetune_from_checkpoint=False)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_trainer_wrapper_from_instance():
    """
    Feature: Create Trainer From Instance
    Description: Test Trainer API to train from self-define instance API.
    Expectation: TypeError
    """
    runner_config = RunnerConfig(epochs=2, batch_size=8, image_size=224, sink_mode=True, per_epoch_size=1)
    config = ConfigArguments(seed=2022, runner_config=runner_config)

    mae_model_with_loss = ViTMAEForPreTraining()

    project_path = MindFormerBook.get_project_path()

    config_path = os.path.join(
        project_path, "configs", "mae", "run_mae_vit_base_p16_224_800ep.yaml"
    )
    dataset_config = MindFormerConfig(config_path)

    new_dataset_dir = make_local_directory(dataset_config)
    make_dataset(new_dataset_dir, num=16)

    dataset_config.train_dataset.data_loader.dataset_dir = new_dataset_dir
    dataset_config.train_dataset_task.dataset_config.data_loader.dataset_dir = new_dataset_dir

    dataset = build_dataset(dataset_config.train_dataset_task)

    lr_schedule = WarmUpLR(learning_rate=0.001, warmup_steps=100)
    optimizer = AdamWeightDecay(beta1=0.009, beta2=0.999,
                                learning_rate=lr_schedule,
                                params=mae_model_with_loss.trainable_params())
    loss_scale = DynamicLossScaleUpdateCell(loss_scale_value=2 ** 12, scale_factor=2, scale_window=1000)
    wrapper = TrainOneStepWithLossScaleCell(mae_model_with_loss, optimizer, scale_sense=loss_scale)

    loss_cb = LossMonitor(per_print_times=2)
    time_cb = TimeMonitor()
    callbacks = [loss_cb, time_cb]

    mim_trainer_wrapper = Trainer(task='masked_image_modeling',
                                  args=config,
                                  wrapper=wrapper,
                                  train_dataset=dataset,
                                  callbacks=callbacks)

    mim_trainer_wrapper.train(resume_or_finetune_from_checkpoint=False)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_trainer_general_from_instance():
    """
    Feature: Create Trainer From Instance
    Description: Test Trainer API to train from self-define instance API.
    Expectation: TypeError
    """
    runner_config = RunnerConfig(epochs=2, batch_size=8, image_size=224, sink_mode=True, per_epoch_size=1)
    config = ConfigArguments(seed=2022, runner_config=runner_config)

    mae_model_with_loss = ViTMAEForPreTraining()

    project_path = MindFormerBook.get_project_path()

    config_path = os.path.join(
        project_path, "configs", "mae", "run_mae_vit_base_p16_224_800ep.yaml"
    )
    dataset_config = MindFormerConfig(config_path)

    new_dataset_dir = make_local_directory(dataset_config)
    make_dataset(new_dataset_dir, num=16)

    dataset_config.train_dataset.data_loader.dataset_dir = new_dataset_dir
    dataset_config.train_dataset_task.dataset_config.data_loader.dataset_dir = new_dataset_dir

    dataset = build_dataset(dataset_config.train_dataset_task)

    lr_schedule = WarmUpLR(learning_rate=0.001, warmup_steps=100)
    optimizer = AdamWeightDecay(beta1=0.009, beta2=0.999,
                                learning_rate=lr_schedule,
                                params=mae_model_with_loss.trainable_params())
    loss_scale = DynamicLossScaleUpdateCell(loss_scale_value=2 ** 12, scale_factor=2, scale_window=1000)
    wrapper = TrainOneStepWithLossScaleCell(mae_model_with_loss, optimizer, scale_sense=loss_scale)

    loss_cb = LossMonitor(per_print_times=2)
    time_cb = TimeMonitor()
    callbacks = [loss_cb, time_cb]

    no_task_name_trainer = Trainer(args=config,
                                   wrapper=wrapper,
                                   train_dataset=dataset,
                                   callbacks=callbacks)

    no_task_name_trainer.train(resume_or_finetune_from_checkpoint=False)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_trainer_auto_to_save_config():
    """
    Feature: Auto Create Trainer.
    Description: Test Trainer API to train.
    Expectation: TypeError
    """
    runner_config = RunnerConfig(epochs=2, batch_size=8, image_size=224)
    config = ConfigArguments(runner_config=runner_config)

    project_path = MindFormerBook.get_project_path()

    config_path = os.path.join(
        project_path, "configs", "mae", "run_mae_vit_base_p16_224_800ep.yaml"
    )
    dataset_config = MindFormerConfig(config_path)

    new_dataset_dir = make_local_directory(dataset_config)
    make_dataset(new_dataset_dir, num=16)

    dataset_config.train_dataset.data_loader.dataset_dir = new_dataset_dir
    dataset_config.train_dataset_task.dataset_config.data_loader.dataset_dir = new_dataset_dir

    dataset = build_dataset(dataset_config.train_dataset_task)

    mim_trainer = Trainer(
        task='masked_image_modeling',
        model='mae_vit_base_p16',
        train_dataset=dataset,
        args=config,
        save_config=True)
    mim_trainer.train()


def make_local_directory(config):
    """make local directory"""
    dataset_dir = config.train_dataset.data_loader.dataset_dir
    new_dataset_dir = ""
    for item in dataset_dir.split("/"):
        new_dataset_dir = os.path.join(new_dataset_dir, item)
    os.makedirs(new_dataset_dir, exist_ok=True)
    return new_dataset_dir


def make_dataset(new_dataset_dir, num):
    """make a fake ImageNet dataset"""
    for label in range(4):
        os.makedirs(os.path.join(new_dataset_dir, str(label)), exist_ok=True)
        for index in range(num):
            image = Image.fromarray(np.ones((255, 255, 3)).astype(np.uint8))
            image.save(os.path.join(new_dataset_dir, str(label), f"test_image_{index}.jpg"))
