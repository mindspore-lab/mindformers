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

"""OPT Predict"""
from transformer.models.opt import OPTConfig, OPT, EvalNet
from transformer.trainer import Trainer, TrainingConfig, parse_config
from transformer.data import create_wiki_dataset


class OPTTrainingConfig(TrainingConfig):
    """
    OPTTrainingConfig
    """

    def __init__(self, *args, **kwargs):
        super(OPTTrainingConfig, self).__init__(*args, **kwargs)
        self.epoch_size = 1
        self.train_data_path = ""
        self.optimizer = "adam"
        self.parallel_mode = "stand_alone"
        self.full_batch = False
        self.global_batch_size = 1
        self.checkpoint_prefix = "opt"
        self.load_checkpoint_path = "./converted_mindspore_opt.ckpt"
        self.vocab_path = "./vocab.json"
        self.input_samples = "Hello world"
        self.generate = True
        self.device_target = "Ascend"


class OPTPredict(Trainer):
    """
    OPTPredict
    """

    def build_model_config(self):
        model_config = OPTConfig()
        return model_config

    def build_model(self, model_config):
        net = OPT(model_config)
        net = EvalNet(net, generate=self.config.generate)
        return net

    def build_dataset(self):
        return create_wiki_dataset(self.config)


if __name__ == "__main__":
    config = OPTTrainingConfig()
    parse_config(config)
    trainer = OPTPredict(config)
    trainer.predict()
