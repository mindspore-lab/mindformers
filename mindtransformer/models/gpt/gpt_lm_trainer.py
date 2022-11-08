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
from mindspore import load_checkpoint, load_param_into_net

from mindtransformer.models.gpt import GPTConfig
from mindtransformer.models.gpt.gpt_lm import GPT2LM
from mindtransformer.trainer import Trainer, TrainingConfig, parse_config
from mindtransformer.data import create_language_model_dataset
from mindtransformer.utils import  get_newest_ckpt


class GPTLMTrainingConfig(TrainingConfig):
    """
    GPTTrainingConfig
    """

    def __init__(self, *args, **kwargs):
        super(GPTLMTrainingConfig, self).__init__(*args, **kwargs)
        self.epoch_size = 1
        self.train_data_path = ""
        self.optimizer = "adam"
        self.parallel_mode = "stand_alone"
        self.full_batch = False
        self.global_batch_size = 4
        self.checkpoint_prefix = "gpt2_language_model"
        self.repeat_count = 1

class GPTTrainer(Trainer):
    """
    GPTTrainer
    """

    def build_model_config(self):
        model_config = GPTConfig()
        return model_config

    def build_model(self, model_config):
        net_with_loss = GPT2LM(model_config)
        return net_with_loss

    def build_dataset(self):
        return create_language_model_dataset(self.config)

    def load_checkpoint(self, net_with_loss):
        """load checkpoint"""
        if self.config.load_checkpoint_path == "" and self.config.save_checkpoint_path != "" \
                and self.config.checkpoint_prefix != "":
            self.config.load_checkpoint_path = get_newest_ckpt(self.config.save_checkpoint_path,
                                                               self.config.checkpoint_prefix)

        if self.config.load_checkpoint_path != "":
            if self.config.load_checkpoint_path.endswith('.ckpt'):
                self.logger.info("Start to load the ckpt from %s", self.config.load_checkpoint_path)
            else:
                self.config.load_checkpoint_path = get_newest_ckpt(self.config.load_checkpoint_path,
                                                                   self.config.checkpoint_prefix)

        if self.config.is_train:
            final_param_dict = {}
            param_dict = load_checkpoint(self.config.load_checkpoint_path)
            for name, _ in param_dict.items():
                final_param_dict['gpt2.' + name] = param_dict[name]
            final_param_dict['gpt2.dense1.weight'] = param_dict['backbone.word_embedding.embedding_table']
            load_param_into_net(net_with_loss, final_param_dict)
        else:

            ckpt = load_checkpoint(self.config.load_checkpoint_path)
            load_param_into_net(net_with_loss, ckpt)

if __name__ == "__main__":
    config = GPTLMTrainingConfig()
    parse_config(config)
    trainer = GPTTrainer(config)
    trainer.train()
