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

"""T5 Trainer"""
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor

from mindtransformer.models.t5 import TransformerConfig, TransformerNetworkWithLoss
from mindtransformer.trainer import Trainer, TrainingConfig, parse_config
from mindtransformer.data import create_t5_dataset
from mindtransformer.learning_rate import create_dynamic_lr


class T5TrainingConfig(TrainingConfig):
    """
    T5TrainingConfig
    """

    def __init__(self, *args, **kwargs):
        super(T5TrainingConfig, self).__init__(*args, **kwargs)
        self.epoch_size = 1
        self.train_data_path = ""
        self.optimizer = "adam"
        self.parallel_mode = "stand_alone"
        self.full_batch = False
        self.global_batch_size = 4
        self.checkpoint_prefix = "T5"

        self.d_kv = 64
        self.attention_dropout_prob = 0.1
        self.hidden_dropout_prob = 0.1
        self.bucket_boundaries = 16
        self.warmup_steps = 100
        self.start_decay_step = 100
        self.min_lr = 1e-5


class T5Trainer(Trainer):
    """
    T5Trainer
    """

    def build_model_config(self):
        model_config = TransformerConfig()
        return model_config

    def build_model(self, model_config):
        """
        build t5 model
        """
        return TransformerNetworkWithLoss(model_config)

    def build_dataset(self):
        return create_t5_dataset(self.config)

    def build_lr(self):
        learning_rate = self.config.learning_rate if self.config.device_target == "Ascend" else 1.0
        lr = Tensor(create_dynamic_lr(schedule="constant*rsqrt_hidden*linear_warmup*rsqrt_decay",
                                      training_steps=self.config.actual_epoch_num * self.config.step_per_epoch,
                                      learning_rate=learning_rate,
                                      warmup_steps=self.config.warmup_steps,
                                      hidden_size=self.config.hidden_size,
                                      start_decay_step=self.config.start_decay_step,
                                      min_lr=self.config.min_lr), mstype.float32)
        return lr


if __name__ == "__main__":
    config = T5TrainingConfig()
    parse_config(config)
    trainer = T5Trainer(config)
    trainer.train()
