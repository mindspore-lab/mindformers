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
"""
Test module for testing the t5 trainer used for mindformers.
How to run this:
pytest tests/st/test_model/test_t5_model/test_t5_trainer.py
"""
import numpy as np
import pytest
from mindspore.nn import AdamWeightDecay, WarmUpLR
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.dataset import GeneratorDataset

from mindformers.trainer import Trainer
from mindformers.trainer.config_args import ConfigArguments, \
    RunnerConfig
from mindformers import T5Config, T5ModelForGeneration


def generator():
    """dataset generator"""
    input_ids = np.random.randint(low=0, high=15, size=(16,)).astype(np.int32)
    attention_mask = np.random.randint(low=0, high=15, size=(16,)).astype(np.int32)
    labels = np.random.randint(low=0, high=15, size=(8,)).astype(np.int32)

    for _ in range(2):
        yield input_ids, attention_mask, labels


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_t5_trainer_train_using_trainer():
    """
    Feature: Create Trainer From Instance
    Description: Test Trainer API to train from self-define instance API.
    Expectation: TypeError
    """

    batch_size = 2
    dataset = GeneratorDataset(generator, column_names=["input_ids", "attention_mask", "labels"])
    # Dataset and operations
    dataset = dataset.batch(batch_size=batch_size)
    # Config definition
    runner_config = RunnerConfig(epochs=1, batch_size=batch_size, sink_mode=True,
                                 per_epoch_size=dataset.get_dataset_size())
    config = ConfigArguments(seed=2022, runner_config=runner_config)
    model_config = T5Config(batch_size=batch_size, num_heads=8, num_hidden_layers=1, hidden_size=512,
                            seq_length=16, max_decode_length=8)
    # Model
    model = T5ModelForGeneration(model_config)
    # optimizer
    lr_schedule = WarmUpLR(learning_rate=0.001, warmup_steps=100)
    optimizer = AdamWeightDecay(beta1=0.009, beta2=0.999,
                                learning_rate=lr_schedule,
                                params=model.trainable_params())

    # callback
    loss_cb = LossMonitor(per_print_times=2)
    time_cb = TimeMonitor()
    callbacks = [loss_cb, time_cb]

    trainer = Trainer(task_name='masked_language_modeling',
                      model=model,
                      config=config,
                      optimizers=optimizer,
                      train_dataset=dataset,
                      callbacks=callbacks)
    trainer.train(resume_from_checkpoint=False)
