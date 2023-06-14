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
pytest tests/st/test_model/test_llama_model/test_llama_model.py
"""
from dataclasses import dataclass
import os
import numpy as np
# import pytest

from mindspore.dataset import GeneratorDataset

from mindformers.trainer import Trainer
from mindformers.trainer.config_args import ConfigArguments, \
    OptimizerConfig, RunnerConfig
from mindformers.models.llama import LlamaConfig, LlamaForCausalLM
from mindformers import MindFormerBook
from mindformers.models import BaseModel


def generator():
    """dataset generator"""
    seq_len = 513
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    input_mask = np.ones_like(input_ids)
    label_ids = input_ids
    train_data = (input_ids, input_mask, label_ids)
    for _ in range(8):
        yield train_data[0]

@dataclass
class Tempconfig:
    seed: int = 0
    runner_config: RunnerConfig = None
    data_size: int = 0
    load_checkpoint: str = ""

# @pytest.mark.level0
# @pytest.mark.platform_x86_ascend_training
# @pytest.mark.platform_arm_ascend_training
# @pytest.mark.env_onecard
def test_llama_trainer_train_from_instance():
    """
    Feature: Create Trainer From Instance
    Description: Test Trainer API to train from self-define instance API.
    Expectation: TypeError
    """
    # Config definition
    runner_config = RunnerConfig(epochs=1, batch_size=4, sink_mode=False)
    optim_config = OptimizerConfig(optim_type='AdamWeightDecay', beta1=0.9, learning_rate=0.001)
    config = ConfigArguments(seed=2022, runner_config=runner_config, optimizer=optim_config)

    # Model Config
    llama_config = LlamaConfig(
        batch_size=1,
        seq_length=512,
        hidden_size=32,
        num_layers=1,
        num_heads=8,
    )

    # Model
    llama_model = LlamaForCausalLM(config=llama_config)

    # Dataset and operations
    dataset = GeneratorDataset(generator, column_names=["input_ids"])
    dataset = dataset.batch(batch_size=1)

    mim_trainer = Trainer(task='text_generation',
                          model=llama_model,
                          args=config,
                          train_dataset=dataset)
    mim_trainer.train(resume_or_finetune_from_checkpoint=False)


# @pytest.mark.level0
# @pytest.mark.platform_x86_ascend_training
# @pytest.mark.platform_arm_ascend_training
# @pytest.mark.env_onecard
class TestModelForLlamaMethod:
    '''A test class for testing Model classes'''
    def setup_method(self):
        """get_input"""
        self.save_directory = os.path.join(MindFormerBook.get_project_path(), 'checkpoint_save', 'llama')

    def test_llama_model(self):
        """
        Feature: LlamaForCausalLM, input config
        Description: Test to get model instance by LlamaForCausalLM and input config
        Expectation: TypeError, ValueError, RuntimeError
        """

        # input model name, load model and weights
        config = LlamaConfig(num_hidden_layers=1)
        llama_model = LlamaForCausalLM(config)
        assert isinstance(llama_model, BaseModel)

    def test_save_model(self):
        """
        Feature: save_pretrained method of LlamaForCausalLM
        Description: Test to save checkpoint for LlamaForCausalLM
        Expectation: ValueError, AttributeError
        """
        llama = LlamaForCausalLM(LlamaConfig(num_layers=1, batch_size=2, seq_length=16))
        llama.save_pretrained(self.save_directory, save_name='llama_test')
        new_llama = LlamaForCausalLM.from_pretrained(self.save_directory)
        new_llama.save_pretrained(self.save_directory, save_name='llama_test')
