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

"""GPT Trainer"""
from mindtransformer.models.gpt import GPTConfig, GPTWithLoss
from mindtransformer.trainer import Trainer, TrainingConfig, parse_config
from mindtransformer.data import create_gpt_dataset


class GPTTrainingConfig(TrainingConfig):
    """
    GPTTrainingConfig
    """

    def __init__(self, *args, **kwargs):
        super(GPTTrainingConfig, self).__init__(*args, **kwargs)
        self.epoch_size = 1
        self.train_data_path = ""
        self.optimizer = "adam"
        self.parallel_mode = "stand_alone"
        self.full_batch = False
        self.global_batch_size = 4
        self.checkpoint_prefix = "gpt"


class GPTTrainer(Trainer):
    """
    GPTTrainer
    """

    def build_model_config(self):
        model_config = GPTConfig()
        return model_config

    def build_model(self, model_config):
        net_with_loss = GPTWithLoss(model_config)
        return net_with_loss

    def build_dataset(self):
        return create_gpt_dataset(self.config)


if __name__ == "__main__":
    config = GPTTrainingConfig()
    parse_config(config)
    trainer = GPTTrainer(config)
    trainer.train()
