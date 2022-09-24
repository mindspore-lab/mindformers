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

"""OPT Trainer"""
from mindspore.nn.transformer.loss import CrossEntropyLoss
from transformer.models.opt import OPTConfig, OPT, OPTWithLoss
from transformer.trainer import Trainer, TrainingConfig, parse_config
from transformer.data import create_wiki_dataset


class OPTTrainingConfig(TrainingConfig):
    """
    OPTTrainingConfig
    """

    def __init__(self, *args, **kwargs):
        super(OPTTrainingConfig, self).__init__(*args, **kwargs)
        self.epoch_size = 1
        self.data_url = ""
        self.optimizer = "adam"
        self.parallel_mode = "stand_alone"
        self.full_batch = False
        self.global_batch_size = 4
        self.ckpt_prefix = "opt"


class OPTTrainer(Trainer):
    """
    OPTTrainer
    """

    def build_model_config(self):
        model_config = OPTConfig()
        return model_config

    def build_model(self, model_config):
        net = OPT(model_config)
        loss = CrossEntropyLoss(model_config.parallel_config.dp_mp_config)
        net_with_loss = OPTWithLoss(net, loss, model_config.parallel_config)
        return net_with_loss

    def build_dataset(self, training_config, device_num, rank):
        return create_wiki_dataset(training_config.global_batch_size, training_config.data_path, device_num, rank)


if __name__ == "__main__":
    config = OPTTrainingConfig()
    parse_config(config)
    trainer = OPTTrainer(config)
    trainer.train()
