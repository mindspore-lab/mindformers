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

windows:
pytest .\\tests\\st\\test_trainer
test_contrastive_language_image_pretrain_trainer
\\test_trainer_auto.py
linux:
pytest ./tests/st/test_trainer/
test_contrastive_language_image_pretrain_trainer
/test_trainer_auto.py
"""
import os
import numpy as np
from PIL import Image
# import pytest

from mindformers.mindformer_book import MindFormerBook
from mindformers.tools.register.config import MindFormerConfig
from mindformers.trainer import Trainer

# @pytest.mark.level0
# @pytest.mark.platform_x86_ascend_training
# @pytest.mark.platform_arm_ascend_training
# @pytest.mark.env_onecard
class TestTrainer:
    """A test class for testing Trainer"""
    def setup_method(self):
        """prepare for test"""
        project_path = MindFormerBook.get_project_path()

        config_path = os.path.join(
            project_path, "configs", "clip",
            "run_clip_vit_b_32_pretrain_flickr8k.yaml"
        )
        config = MindFormerConfig(config_path)

        new_root_dir, new_dataset_dir, new_annotation_dir,\
        local_root, output_dir = self.make_local_directory(config)
        self.make_dataset(new_dataset_dir, new_annotation_dir, num=50)
        self.local_root = local_root
        self.output_dir = output_dir

        config.train_dataset.data_loader.dataset_dir = new_root_dir
        config.train_dataset_task.dataset_config.data_loader.dataset_dir = new_root_dir
        config.output_dir = output_dir

        self.config = config

    def test_trainer_auto(self):
        """
        Feature: Auto Create Trainer.
        Description: Test Trainer API to train.
        Expectation: TypeError
        """
        trainer = Trainer(task='contrastive_language_image_pretrain', args=self.config)
        trainer.train()

    def make_local_directory(self, config):
        """make local directory"""

        dataset_dir = config.train_dataset.data_loader.dataset_dir
        new_root_dir = MindFormerBook.get_project_path()
        for item in dataset_dir.split("/")[1:]:
            new_root_dir = os.path.join(new_root_dir, item)

        annotation_dir = os.path.join(dataset_dir, "Flickr8k_text")
        dataset_dir = os.path.join(dataset_dir, "Flickr8k_Dataset", "Flickr8k_Dataset")
        local_root = os.path.join(MindFormerBook.get_project_path(), dataset_dir.split("/")[1])

        new_dataset_dir = MindFormerBook.get_project_path()
        for item in dataset_dir.split("/")[1:]:
            new_dataset_dir = os.path.join(new_dataset_dir, item)

        new_annotation_dir = MindFormerBook.get_project_path()
        for item in annotation_dir.split("/")[1:]:
            new_annotation_dir = os.path.join(new_annotation_dir, item)

        output_dir = new_annotation_dir

        os.makedirs(new_dataset_dir, exist_ok=True)
        os.makedirs(new_annotation_dir, exist_ok=True)
        return new_root_dir, new_dataset_dir, new_annotation_dir, local_root, output_dir

    def make_dataset(self, new_dataset_dir, new_annotation_dir, num):
        """make a fake Flickr8k dataset"""
        for index in range(num):
            image = Image.fromarray(np.ones((478, 269, 3)).astype(np.uint8))
            image.save(os.path.join(new_dataset_dir, f"test_image_{index}.jpg"))

        token_file = os.path.join(new_annotation_dir, "Flickr8k.token.txt")
        with open(token_file, 'w', encoding='utf-8') as filer:
            for index in range(num):
                filer.write(f"test_image_{index}.jpg#0"
                            f"   A child in a pink dress is climbing"
                            f" up a set of stairs in an entry way .\n")
                filer.write(f"test_image_{index}.jpg#1"
                            f"   A girl going into a wooden building .\n")
                filer.write(f"test_image_{index}.jpg#2"
                            f"   A little girl climbing into a wooden playhouse .\n")
                filer.write(f"test_image_{index}.jpg#3"
                            f"   A little girl climbing the stairs to her playhouse .\n")
                filer.write(f"test_image_{index}.jpg#4"
                            f"   A little girl in a pink dress going into a wooden cabin .\n")

        train_file = os.path.join(new_annotation_dir, "Flickr_8k.trainImages.txt")
        with open(train_file, 'w', encoding='utf-8') as filer:
            for index in range(num):
                filer.write(f"test_image_{index}.jpg\n")

        test_file = os.path.join(new_annotation_dir, "Flickr_8k.testImages.txt")
        with open(test_file, 'w', encoding='utf-8') as filer:
            for index in range(num):
                filer.write(f"test_image_{index}.jpg\n")

        dev_file = os.path.join(new_annotation_dir, "Flickr_8k.devImages.txt")
        with open(dev_file, 'w', encoding='utf-8') as filer:
            for index in range(num):
                filer.write(f"test_image_{index}.jpg\n")
