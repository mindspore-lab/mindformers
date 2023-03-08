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
Test module for testing the gpt interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_gpt_model/test_gpt_from_instance.py
"""
from dataclasses import dataclass
import os
import numpy as np
import pytest
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.dataset import GeneratorDataset

from mindformers.trainer import Trainer
from mindformers.trainer.config_args import ConfigArguments, \
    RunnerConfig
from mindformers.models.gpt2.gpt2 import GPT2LMHeadModel, Gpt2Config
from mindformers.core.lr import WarmUpDecayLR
from mindformers import MindFormerBook, AutoModel, AutoConfig
from mindformers.tools import logger
from mindformers.models import BaseModel
from mindformers.core.optim import FusedAdamWeightDecay


def generator():
    """dataset generator"""
    seq_len = 1024
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    input_mask = np.ones_like(input_ids)
    label_ids = input_ids
    train_data = (input_ids, input_mask, label_ids)
    for _ in range(512):
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
def test_gpt_trainer_train_from_instance():
    """
    Feature: Create Trainer From Instance
    Description: Test Trainer API to train from self-define instance API.
    Expectation: TypeError
    """
    # Config definition
    runner_config = RunnerConfig(epochs=1, batch_size=8, sink_mode=True, per_epoch_size=2)
    config = ConfigArguments(seed=2022, runner_config=runner_config)

    # Model
    gpt_model = GPT2LMHeadModel()

    # Dataset and operations
    dataset = GeneratorDataset(generator, column_names=["input_ids", "input_mask", "label_ids"])
    dataset = dataset.batch(batch_size=8)

    # optimizer
    lr_schedule = WarmUpDecayLR(learning_rate=0.0001, end_learning_rate=0.00001, warmup_steps=0, decay_steps=512)
    optimizer = FusedAdamWeightDecay(beta1=0.009, beta2=0.999,
                                     learning_rate=lr_schedule,
                                     params=gpt_model.trainable_params())

    # callback
    loss_cb = LossMonitor(per_print_times=2)
    time_cb = TimeMonitor()
    callbacks = [loss_cb, time_cb]

    mlm_trainer = Trainer(model=gpt_model,  # model and loss
                          config=config,
                          optimizers=optimizer,
                          train_dataset=dataset,
                          callbacks=callbacks)
    mlm_trainer.train(resume_or_finetune_from_checkpoint=False)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
class TestModelForGptMethod:
    '''A test class for testing Model classes'''
    def setup_method(self):
        """get_input"""
        self.save_directory = os.path.join(MindFormerBook.get_project_path(), 'checkpoint_save', 'gpt2')

    def test_gpt_model(self):
        """
        Feature: GptForPretraining, from_pretrained, input config
        Description: Test to get model instance by ClipModel.from_pretrained
                    and input config
        Expectation: TypeError, ValueError, RuntimeError
        """

        # input model name, load model and weights
        config = Gpt2Config(num_hidden_layers=1)
        gpt2_model = GPT2LMHeadModel(config)
        assert isinstance(gpt2_model, BaseModel)

    def test_gpt_config_model(self):
        """
        Feature: GPT2LMHeadModel, input config
        Description: Test to get config instance by Gpt2Config.from_pretrained
                    and input config
        Expectation: TypeError, ValueError, RuntimeError
        """

        # input model name, load model and weights
        config = Gpt2Config.from_pretrained("gpt2")
        gpt2_model = GPT2LMHeadModel(config)
        assert isinstance(gpt2_model, BaseModel)

    def test_save_model(self):
        """
        Feature: save_pretrained method of GptModel
        Description: Test to save checkpoint for GptModel
        Expectation: ValueError, AttributeError
        """
        gpt = GPT2LMHeadModel(Gpt2Config(num_layers=1, hidden_dropout_prob=0.0,
                                         attention_probs_dropout_prob=0.0, batch_size=2, seq_length=16))
        gpt.save_pretrained(self.save_directory, save_name='gpt_test')
        new_gpt = GPT2LMHeadModel.from_pretrained(self.save_directory)
        new_gpt.save_pretrained(self.save_directory, save_name='gpt_test')

    def test_auto_model(self):
        """
        Feature: AutoModel, from_pretrained, from_config
        Description: Test to get model instance by AutoModel.from_pretrained
                    and AutoModel.from_config
        Expectation: TypeError, ValueError, RuntimeError
        """
        AutoModel.show_support_list()
        self.config_path = os.path.join(MindFormerBook.get_project_path(),
                                        'configs', 'gpt2', 'model_config', "gpt2.yaml")
        support_list = AutoModel.get_support_list()
        logger.info(support_list)
        # input yaml path, load model without weights
        model = AutoModel.from_config(self.config_path)
        assert isinstance(model, BaseModel)

    def test_auto(self):
        """
        Feature: AutoModel, AutoConfig, from_pretrained
        Description: Test to get model and config instance by AutoModel.from_pretrained
                    and AutoConfig.from_pretrained
        """
        config = AutoConfig.from_pretrained("gpt2")
        model = AutoModel.from_pretrained("gpt2")
        assert isinstance(model, BaseModel) and isinstance(config, Gpt2Config)
