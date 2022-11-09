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
"""ViT Trainer"""
import math
import numpy as np

from mindspore.common.tensor import Tensor
from mindtransformer.models.vit import ViTConfig, ViTWithLoss, ViT
from mindtransformer.trainer import Trainer, TrainingConfig, parse_config
from mindtransformer.data import create_imagenet_dataset
from mindtransformer.optim.optimizer import build_optimizer


def get_lr(global_step, lr_init, lr_end, lr_max, warmup_epochs, \
           total_epochs, steps_per_epoch, lr_decay_mode, poly_power):
    """
    generate learning rate array

    Args:
       global_step(int): total steps of the training
       lr_init(float): init learning rate
       lr_end(float): end learning rate
       lr_max(float): max learning rate
       warmup_epochs(int): number of warmup epochs
       total_epochs(int): total epoch of training
       steps_per_epoch(int): steps of one epoch
       lr_decay_mode(string): learning rate decay mode, including steps, poly, cosine or default

    Returns:
       np.array, learning rate array
    """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = int(steps_per_epoch) * int(warmup_epochs)
    if lr_decay_mode == 'steps':
        decay_epoch_index = [0.3 * total_steps, 0.6 * total_steps, 0.8 * total_steps]
        for i in range(total_steps):
            if i < decay_epoch_index[0]:
                lr = lr_max
            elif i < decay_epoch_index[1]:
                lr = lr_max * 0.1
            elif i < decay_epoch_index[2]:
                lr = lr_max * 0.01
            else:
                lr = lr_max * 0.001
            lr_each_step.append(lr)
    elif lr_decay_mode == 'poly':
        if warmup_steps != 0:
            inc_each_step = (float(lr_max) - float(lr_init)) / float(warmup_steps)
        else:
            inc_each_step = 0
        for i in range(total_steps):
            if i < warmup_steps:
                lr = float(lr_init) + inc_each_step * float(i)
            else:
                base = (1.0 - (float(i) - float(warmup_steps)) / (float(total_steps) - float(warmup_steps)))
                lr = float(lr_max - lr_end) * base ** poly_power + lr_end
                lr = max(lr, 0.0)
            lr_each_step.append(lr)
    elif lr_decay_mode == 'cosine':
        decay_steps = total_steps - warmup_steps
        for i in range(total_steps):
            if i < warmup_steps:
                lr_inc = (float(lr_max) - float(lr_init)) / float(warmup_steps)
                lr = float(lr_init) + lr_inc * (i + 1)
            else:
                cur_step = i + 1 - warmup_steps
                lr = lr_max * (1 + math.cos(math.pi * cur_step / decay_steps)) / 2
            lr_each_step.append(lr)
    else:
        for i in range(total_steps):
            if i < warmup_steps:
                lr = lr_init + (lr_max - lr_init) * i / warmup_steps
            else:
                lr = lr_max - (lr_max - lr_end) * (i - warmup_steps) / (total_steps - warmup_steps)
            lr_each_step.append(lr)

    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]

    return learning_rate



class ViTTrainingConfig(TrainingConfig):
    """
    ViTTrainingConfig
    """

    def __init__(self, *args, **kwargs):
        super(ViTTrainingConfig, self).__init__(*args, **kwargs)
        self.epoch_size: int = 1
        self.train_data_path: str = ""
        self.optimizer: str = "adamw"
        self.parallel_mode: str = "stand_alone"
        self.full_batch: bool = False
        self.checkpoint_prefix: str = "vit"
        self.device_target: str = "GPU"
        self.interpolation: str = 'BILINEAR'
        self.image_size: int = 224
        self.autoaugment: int = 1
        self.mixup: float = 0.2
        self.crop_min: float = 0.05
        self.num_workers: int = 12
        self.num_classes: int = 1000
        self.sink_size: int = 625
        self.lr_max: float = 0.00355
        self.warmup_epochs: int = 40
        self.lr_decay_mode: str = 'cosine'
        self.poly_power: float = 2.0
        self.weight_decay: float = 0.05
        self.loss_scale: int = 1024
        self.beta1: float = 0.9
        self.beta2: float = 0.999
        self.no_weight_decay_filter: str = "beta,bias"
        self.gc_flag: int = 0
        self.generate = False

class ViTTrainer(Trainer):
    """
    ViTTrainer
    """

    def build_model_config(self):
        model_config = ViTConfig()
        return model_config

    def build_model(self, model_config):
        if self.config.is_training:
            net = ViTWithLoss(model_config)
        else:
            net = ViT(model_config)
        return net

    def build_dataset(self):
        return create_imagenet_dataset(self.config)

    def build_lr(self):
        lr_array = get_lr(global_step=0, lr_init=self.config.start_lr, lr_end=self.config.end_lr,
                          lr_max=self.config.lr_max, warmup_epochs=self.config.warmup_epochs,
                          total_epochs=self.config.epoch_size, steps_per_epoch=self.config.step_per_epoch,
                          lr_decay_mode=self.config.lr_decay_mode, poly_power=self.config.poly_power)
        lr = Tensor(lr_array)
        return lr

    def build_optimizer(self, net_with_loss):
        return build_optimizer(net=net_with_loss,
                               lr=self.build_lr(),
                               optimizer_name=self.config.optimizer,
                               args=self.config,
                               stage_num=1,
                               fused=True,
                               opt_offload=self.config.opt_offload,
                               flatten_weights=self.config.flatten_weights)


if __name__ == "__main__":
    config = ViTTrainingConfig()
    parse_config(config)
    trainer = ViTTrainer(config)
    if config.is_training:
        trainer.train()
    else:
        trainer.predict()
