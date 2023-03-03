# Copyright 2023 Huawei Technologies Co., Ltd
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
pytest tests/st/test_trainer/test_trainer_from_training_args.py
"""
import os
import pytest
import numpy as np
from PIL import Image

from mindformers.mindformer_book import MindFormerBook
from mindformers.tools.register.config import MindFormerConfig
from mindformers.dataset.build_dataset import build_dataset
from mindformers.trainer import Trainer
from mindformers.models import ViTMAEForPreTraining
from mindformers.trainer.training_args import TrainingArguments


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_trainer_train_from_training_args():
    """
    Feature: Create Trainer From Config
    Description: Test Trainer API to train from training arguments class
    Expectation: TypeError
    """
    project_path = MindFormerBook.get_project_path()

    config_path = os.path.join(
        project_path, "configs", "mae", "task_config", "mae_dataset.yaml"
    )
    dataset_config = MindFormerConfig(config_path)

    new_dataset_dir = make_local_directory(dataset_config)
    make_dataset(new_dataset_dir, num=16)

    dataset_config.train_dataset.data_loader.dataset_dir = new_dataset_dir
    dataset_config.train_dataset_task.dataset_config.data_loader.dataset_dir = new_dataset_dir

    dataset = build_dataset(dataset_config.train_dataset_task)

    mae_config = TrainingArguments(
        lr_scheduler_type="cosine", optim="AdamWeightDecay", num_train_epochs=2,
        batch_size=8, per_device_train_batch_size=8, adam_epsilon=1e-9, adam_beta1=0.999,
        adam_beta2=0.95, weight_decay=0.5, save_strategy='steps', save_steps=1000, sink_mode=False)

    mae_model = ViTMAEForPreTraining()
    mim_trainer = Trainer(task='masked_image_modeling',
                          model=mae_model,
                          args=mae_config,
                          train_dataset=dataset)
    mim_trainer.train(resume_or_finetune_from_checkpoint=False)


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
