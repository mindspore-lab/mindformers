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

"""T5 Predict"""
from transformer.models.t5 import TransformerConfig, TransformerModel, EvalNet
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
        self.global_batch_size = 1
        self.ckpt_prefix = "T5"
        self.ckpt_path = "./converted_mindspore_t5.ckpt"
        self.vocab_path = "./vocab.json"
        self.input_samples = "Hello world"
        self.generate = True
        self.device_target = "Ascend"


class T5Predict(Trainer):
    """
    T5Predict
    """

    def build_model_config(self):
        model_config = TransformerConfig()
        return model_config

    def build_model(self, model_config):
        network = TransformerModel(config=model_config)
        net = EvalNet(network, generate=self.config.generate)
        return net

    def build_dataset(self):
        return create_t5_dataset(self.config)


if __name__ == "__main__":
    config = T5TrainingConfig()
    parse_config(config)
    trainer = T5Predict(config)
    trainer.predict()
