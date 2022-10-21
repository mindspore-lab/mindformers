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
"""Vit Trainer"""
from transformer.models.vit import VitConfig, VitWithLoss, ViT
from transformer.trainer import Trainer, TrainingConfig, parse_config
from transformer.data import create_image_dataset


class VitTrainingConfig(TrainingConfig):
    """
    VitTrainingConfig
    """

    def __init__(self, *args, **kwargs):
        super(VitTrainingConfig, self).__init__(*args, **kwargs)
        self.epoch_size = 1
        self.train_data_path = ""
        self.optimizer = "adam"
        self.parallel_mode = "stand_alone"
        self.full_batch = False
        self.global_batch_size = 128
        self.checkpoint_prefix = "vit"
        self.device_target = "GPU"
        self.interpolation = 'BILINEAR'
        self.image_size = 224
        self.autoaugment = 1
        self.mixup = 0.2
        self.crop_min = 0.05
        self.num_workers = 12
        self.num_classes = 1000
        self.sink_size = 625


class VitTrainer(Trainer):
    """
    VitTrainer
    """

    def build_model_config(self):
        model_config = VitConfig()
        return model_config

    def build_model(self, model_config):
        if self.config.is_training:
            net = VitWithLoss(model_config)
        else:
            net = ViT(model_config)
        return net

    def build_dataset(self):
        return create_image_dataset(self.config)


if __name__ == "__main__":
    config = VitTrainingConfig()
    parse_config(config)
    trainer = VitTrainer(config)
    if config.is_training:
        trainer.train()
    else:
        trainer.predict()
