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
import os
import pytest
import numpy as np
from PIL import Image
from mindformers.mindformer_book import MindFormerBook
from mindformers.tools.register.config import MindFormerConfig
from mindformers.dataset.build_dataset import build_dataset
from mindformers.trainer import Trainer
from mindformers.tools.logger import logger


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_trainer_train_auto():
    """prepare for test"""
    project_path = MindFormerBook.get_project_path()

    config_path = os.path.join(
        project_path, "configs", "mae", "run_mae_vit_base_p16_224_800ep.yaml"
    )
    config = MindFormerConfig(config_path)

    new_dataset_dir = make_local_directory(config)
    make_dataset(new_dataset_dir, num=16)

    config.train_dataset.data_loader.dataset_dir = new_dataset_dir
    config.train_dataset_task.dataset_config.data_loader.dataset_dir = new_dataset_dir

    config.runner_config.epochs = 2
    config.runner_config.batch_size = 8

    dataset = build_dataset(config.train_dataset_task)
    mim_trainer = Trainer(
        task_name='masked_image_modeling',
        model='mae_vit_base_p16',
        train_dataset=dataset,
        config=config)
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
