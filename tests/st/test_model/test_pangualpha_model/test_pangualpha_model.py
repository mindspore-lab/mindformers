# Copyright 2023 Huawei Technologies Co., Ltd
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
"""
Test module for testing the pangualpha interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_pangualpha_model/test_pangualpha_model.py
"""
from dataclasses import dataclass
import os
import numpy as np
import pytest
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.dataset import GeneratorDataset

from mindformers.models import BaseConfig
from mindformers.trainer import Trainer
from mindformers.trainer.config_args import ConfigArguments, RunnerConfig
from mindformers.models.pangualpha import PanguAlphaHeadModel, PanguAlphaConfig
from mindformers.core.lr import WarmUpDecayLR
from mindformers import MindFormerBook
from mindformers.models import BaseModel
from mindformers.core.optim import FP32StateAdamWeightDecay


def generator():
    """dataset generator"""
    seq_len = 129
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    position_ids = np.ones((seq_len)).astype(np.int32)
    attention_mask = np.ones((seq_len, seq_len)).astype(np.int32)

    train_data = (input_ids, position_ids, attention_mask)
    for _ in range(8):
        yield train_data

@dataclass
class Tempconfig:
    seed: int = 0
    runner_config: RunnerConfig = None
    data_size: int = 0
    resume_or_finetune_checkpoint: str = ""

@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_pangualpha_trainer_train_from_instance():
    """
    Feature: Create Trainer From Instance
    Description: Test Trainer API to train from self-define instance API.
    Expectation: TypeError
    """
    # Config definition
    runner_config = RunnerConfig(epochs=1, batch_size=2, sink_mode=False)

    config = ConfigArguments(seed=2022, runner_config=runner_config)

    # Model Config
    pangualpha_config = PanguAlphaConfig(batch_size=2,
                                         seq_length=128,
                                         hidden_size=128,
                                         ffn_hidden_size=128*4,
                                         num_layers=2,
                                         num_heads=4)

 # Model
    pangualpha_model = PanguAlphaHeadModel(config=pangualpha_config)

    # Dataset and operations
    dataset = GeneratorDataset(generator, column_names=["input_ids", "position_id", "attention_mask"])
    dataset = dataset.batch(batch_size=2)

    # optimizer
    lr_schedule = WarmUpDecayLR(learning_rate=0.0001, end_learning_rate=0.00001,
                                warmup_steps=0, decay_steps=512)
    optimizer = FP32StateAdamWeightDecay(beta1=0.009, beta2=0.999,
                                         learning_rate=lr_schedule,
                                         params=pangualpha_model.trainable_params())

    # callback
    loss_cb = LossMonitor(per_print_times=2)
    time_cb = TimeMonitor()
    callbacks = [loss_cb, time_cb]

    llm_trainer = Trainer(model=pangualpha_model,
                          config=config,
                          optimizers=optimizer,
                          train_dataset=dataset,
                          callbacks=callbacks,
                          task="text_generation")
    llm_trainer.train(resume_or_finetune_from_checkpoint=False)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
class TestModelForPanguAlphaMethod:
    '''A test class for testing Model classes'''
    def setup_method(self):
        """get_input"""
        self.save_directory = os.path.join(MindFormerBook.get_project_path(), 'checkpoint_save', 'pangualpha')

    def test_pangualpha_model(self):
        """
        Feature: PanguAlphaHeadModel, input config
        Description: Test to get model instance by PanguAlphaHeadModel and input config
        Expectation: TypeError, ValueError, RuntimeError
        """

        # input model name, load model and weights
        config = PanguAlphaConfig(batch_size=2,
                                  seq_length=128,
                                  hidden_size=128,
                                  ffn_hidden_size=128*4,
                                  num_layers=2,
                                  num_heads=4)

        pangualpha_model = PanguAlphaHeadModel(config)
        assert isinstance(pangualpha_model, BaseModel)

    def test_pangualpha_config(self):
        """
        Feature: PanguAlphaHeadModel, input config
        Description: Test to get config instance by PanguAlphaConfig.from_pretrained
                    and input config
        Expectation: TypeError, ValueError, RuntimeError
        """

        # input model name, load model and weights
        pangualpha_config = PanguAlphaConfig.from_pretrained("pangualpha_2_6b")
        assert isinstance(pangualpha_config, BaseConfig)

    def test_save_model(self):
        """
        Feature: save_pretrained method of PanguAlphaHeadModel
        Description: Test to save checkpoint for PanguAlphaHeadModel
        Expectation: ValueError, AttributeError
        """
        config = PanguAlphaConfig(batch_size=2,
                                  seq_length=16,
                                  hidden_size=128,
                                  ffn_hidden_size=128*4,
                                  num_layers=2,
                                  num_heads=4,
                                  hidden_dropout_rate=0.1,
                                  attention_dropout_rate=0.1)
        pangualpha = PanguAlphaHeadModel(config)
        pangualpha.save_pretrained(self.save_directory, save_name='pangualpha_test')
        new_pangualpha = PanguAlphaHeadModel.from_pretrained(self.save_directory)
        new_pangualpha.save_pretrained(self.save_directory, save_name='pangualpha_test')
