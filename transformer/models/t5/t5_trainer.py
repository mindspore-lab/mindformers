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
from transformer.models.t5 import TransformerConfig, TransformerNetworkWithLoss
from transformer.trainer import Trainer, TrainingConfig, parse_config
from transformer.data import create_t5_dataset


class T5TrainingConfig(TrainingConfig):
    """
    T5TrainingConfig
    """

    def __init__(self, *args, **kwargs):
        super(T5TrainingConfig, self).__init__(*args, **kwargs)
        self.epoch_size = 1
        self.data_url = ""
        self.optimizer = "adam"
        self.parallel_mode = "stand_alone"
        self.full_batch = False
        self.global_batch_size = 4
        self.ckpt_prefix = "T5"

        self.d_kv = 64
        self.attention_dropout_prob = 0.1
        self.hidden_dropout_prob = 0.1
        self.bucket_boundaries = 16


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


if __name__ == "__main__":
    config = T5TrainingConfig()
    parse_config(config)
    trainer = T5Trainer(config)
    trainer.train()
