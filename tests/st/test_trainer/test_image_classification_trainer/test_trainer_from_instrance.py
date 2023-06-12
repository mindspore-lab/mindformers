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
pytest .\\tests\\st\\test_trainertest_image_classification_trainer\\test_trainer_from_instance.py
linux:
pytest ./tests/st/test_trainer/test_image_classification_trainer/test_trainer_from_instance.py
"""
import os
import numpy as np
# import pytest
from PIL import Image
import mindspore as ms
from mindformers.mindformer_book import MindFormerBook
from mindformers.tools.register.config import MindFormerConfig
from mindformers.models import ViTForImageClassification, ViTConfig
from mindformers.trainer import Trainer
from mindformers.trainer.config_args import ConfigArguments, RunnerConfig
from mindformers.dataset.build_dataset import build_dataset
from mindformers.core.lr import WarmUpCosineDecayV1


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
            project_path, "configs", "vit",
            "run_vit_base_p16_224_100ep.yaml"
        )
        config = MindFormerConfig(config_path)

        new_dataset_dir = self.make_local_directory(config)
        self.make_dataset(new_dataset_dir, num=16)

        config.train_dataset.data_loader.dataset_dir = new_dataset_dir
        config.train_dataset_task.dataset_config.data_loader.dataset_dir = new_dataset_dir

        self.config = config

    def test_trainer_train_from_instance(self):
        """
        Feature: Create Trainer From Instance
        Description: Test Trainer API to train from self-define instance API.
        Expectation: TypeError
        """
        runner_config = RunnerConfig(
            epochs=2, batch_size=8,
            image_size=224, sink_mode=False,
            sink_size=-1, initial_epoch=0,
            has_trained_epoches=0, has_trained_steps=0
        )
        config = ConfigArguments(seed=2022, runner_config=runner_config)

        vit_config = ViTConfig.from_pretrained('vit_base_p16')
        vit_config.checkpoint_name_or_path = None
        vit_model = ViTForImageClassification(vit_config)
        vit_model.set_train()

        dataset = build_dataset(self.config.train_dataset_task)

        lr_scheduler = WarmUpCosineDecayV1(
            min_lr=0.0, base_lr=0.0000625, warmup_steps=10, decay_steps=190
        )
        optimizer = ms.nn.AdamWeightDecay(
            params=vit_model.trainable_params(),
            learning_rate=lr_scheduler, weight_decay=0.05
        )

        loss_cb = ms.LossMonitor(per_print_times=1)
        callbacks = [loss_cb]

        trainer = Trainer(task='image_classification',
                          model=vit_model,
                          args=config,
                          optimizers=optimizer,
                          train_dataset=dataset,
                          callbacks=callbacks)
        trainer.train(resume_or_finetune_from_checkpoint=False)

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
