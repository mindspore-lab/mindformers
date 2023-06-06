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
Test Module for testing clip_pretrain dataset for clip trainer.

How to run this:
windows:
pytest .\\tests\\st\\test_trainer\\test_image_classification_trainer\\test_dataset.py
linux:
pytest ./tests/st/test_trainer/test_image_classification_trainer/test_dataset.py
"""
import os
import pytest
import numpy as np
from PIL import Image
from mindformers.mindformer_book import MindFormerBook
from mindformers.tools.register.config import MindFormerConfig
from mindformers.dataset.build_dataset import build_dataset


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
class TestVitTrainDataset:
    """A test class for testing ImageClassificationTrainDataset classes"""
    def setup_method(self):
        """prepare for test"""
        project_path = MindFormerBook.get_project_path()

        config_path = os.path.join(
            project_path, "configs", "vit",
            "run_vit_base_p16_224_100ep.yaml"
        )
        config = MindFormerConfig(config_path)

        new_dataset_dir = self.make_local_directory(config)
        self.make_dataset(new_dataset_dir, num=16)

        config.train_dataset.data_loader.dataset_dir = new_dataset_dir
        config.train_dataset_task.dataset_config.data_loader.dataset_dir = new_dataset_dir

        self.config = config

    def test_dataset(self):
        """
        Feature: ImageClassificationTrainDataset
        Description: A data set for contrastive language image pretrain
        Expectation: TypeError, ValueError
        """
        data_loader = build_dataset(self.config.train_dataset_task)
        for item in data_loader:
            assert item[0].shape == (32, 3, 224, 224)
            assert item[1].shape == (32, 1000)

    def make_local_directory(self, config):
        """make local directory"""
        dataset_dir = config.train_dataset.data_loader.dataset_dir
        new_dataset_dir = ""
        for item in dataset_dir.split("/"):
            new_dataset_dir = os.path.join(new_dataset_dir, item)
        os.makedirs(new_dataset_dir, exist_ok=True)
        return new_dataset_dir

    def make_dataset(self, new_dataset_dir, num):
        """make a fake ImageNet dataset"""
        for label in range(4):
            os.makedirs(os.path.join(new_dataset_dir, str(label)), exist_ok=True)
            for index in range(num):
                image = Image.fromarray(np.ones((255, 255, 3)).astype(np.uint8))
                image.save(os.path.join(new_dataset_dir, str(label), f"test_image_{index}.jpg"))
