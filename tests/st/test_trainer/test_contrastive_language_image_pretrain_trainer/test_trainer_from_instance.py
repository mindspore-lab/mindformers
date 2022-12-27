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
\\test_trainer_from_config.py
linux:
pytest ./tests/st/test_trainer/
test_contrastive_language_image_pretrain_trainer
/test_trainer_from_config.py
"""
import os
import numpy as np
from PIL import Image

from mindformers.mindformer_book import MindFormerBook
from mindformers.tools.register.config import MindFormerConfig
from mindformers.models import ClipModel, ClipConfig
from mindformers.trainer import Trainer
from mindformers.trainer.config_args import ConfigArguments, RunnerConfig
from mindformers.dataset.build_dataset import build_dataset


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

        new_dataset_dir, new_annotation_dir,\
        local_root, output_dir = self.make_local_directory(config)
        self.make_dataset(new_dataset_dir, new_annotation_dir, num=50)
        self.local_root = local_root
        self.output_dir = output_dir

        config.train_dataset.data_loader.dataset_dir = new_dataset_dir
        config.train_dataset.data_loader.annotation_dir = new_annotation_dir
        config.train_dataset_task.dataset_config.data_loader.dataset_dir = new_dataset_dir
        config.train_dataset_task.dataset_config.data_loader.annotation_dir = new_annotation_dir
        config.output_dir = output_dir

        self.config = config

    def test_trainer_train_from_instance(self):
        """
        Feature: Create Trainer From Instance
        Description: Test Trainer API to train from self-define instance API.
        Expectation: TypeError
        """
        runner_config = RunnerConfig(
            epochs=5, batch_size=32,
            image_size=224, sink_mode=False,
            per_epoch_size=-1, initial_epoch=0,
            has_trained_epoches=0, has_trained_steps=0
        )
        config = ConfigArguments(seed=2022, runner_config=runner_config)

        clip_config = ClipConfig.from_pretrained('clip_vit_b_32')
        clip_config.checkpoint_name_or_path = None
        clip_model = ClipModel(clip_config)
        clip_model.set_train()

        dataset = build_dataset(self.config.train_dataset_task)

        optimizer = ms.nn.AdamWeightDecay(
            params=clip_model.trainable_params(),
            learning_rate=1e-5, weight_decay=1e-3
        )

        loss_cb = ms.LossMonitor(per_print_times=1)
        lr_scheduler = ms.ReduceLROnPlateau(
            monitor="loss", mode="min",
            patience=2, factor=0.5, verbose=True
        )
        callbacks = [loss_cb, lr_scheduler]

        mim_trainer = Trainer(task='contrastive_language_image_pretrain',
                              model=clip_model,
                              config=config,
                              optimizers=optimizer,
                              train_dataset=dataset,
                              callbacks=callbacks)
        mim_trainer.train(resume_or_finetune_from_checkpoint=False)

    def make_local_directory(self, config):
        """make local directory"""

        dataset_dir = config.train_dataset.data_loader.dataset_dir
        local_root = os.path.join(MindFormerBook.get_project_path(), dataset_dir.split("/")[1])

        new_dataset_dir = MindFormerBook.get_project_path()
        for item in dataset_dir.split("/")[1:]:
            new_dataset_dir = os.path.join(new_dataset_dir, item)

        annotation_dir = config.train_dataset.data_loader.annotation_dir
        new_annotation_dir = MindFormerBook.get_project_path()
        for item in annotation_dir.split("/")[1:]:
            new_annotation_dir = os.path.join(new_annotation_dir, item)

        output_dir = new_annotation_dir

        os.makedirs(new_dataset_dir, exist_ok=True)
        os.makedirs(new_annotation_dir, exist_ok=True)
        return new_dataset_dir, new_annotation_dir, local_root, output_dir

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
