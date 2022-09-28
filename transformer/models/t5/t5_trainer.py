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
from mindspore.nn.transformer.loss import CrossEntropyLoss
from transformer.models.t5 import TransformerConfig, TransformerModel, TransformerNetworkWithLoss
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
        parallel_config = model_config.parallel_config
        network = TransformerModel(config=model_config)
        loss = CrossEntropyLoss(parallel_config=parallel_config.dp_mp_config)
        net_with_loss = TransformerNetworkWithLoss(network=network, loss=loss)
        net_with_loss.set_train(True)

        # disable the bias
        for param in net_with_loss.trainable_params():
            if ('bias' in param.name or 'beta' in param.name) and 'relative' not in param.name:
                param.requires_grad = False
            self.logger.info(f"Param name {param.name} is disabled gradients.")
        return net_with_loss

    def build_dataset(self, training_config, device_num, rank):
        return create_t5_dataset(training_config.global_batch_size, training_config.data_path, device_num, rank)


if __name__ == "__main__":
    config = T5TrainingConfig()
    parse_config(config)
    trainer = T5Trainer(config)
    trainer.train()
