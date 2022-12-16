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
Test module for testing the bert interface used for mindformers.
How to run this:
pytest tests/st/test_bert/test_bert_from_instance.py
"""
import numpy as np
import pytest
from mindspore.nn import AdamWeightDecay, WarmUpLR
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.dataset import GeneratorDataset

from mindformers.trainer import Trainer
from mindformers.common.context import init_context
from mindformers.trainer.config_args import ConfigArguments, \
    RunnerConfig, ContextConfig
from mindformers.models.bert.bert import BertForPretraining


def generator():
    """dataset generator"""
    data = np.random.randint(low=0, high=15, size=(128,)).astype(np.int32)
    input_mask = np.ones_like(data)
    token_type_id = np.zeros_like(data)
    next_sentence_lables = np.array([1]).astype(np.int32)
    masked_lm_positions = np.array([1, 2]).astype(np.int32)
    masked_lm_ids = np.array([1, 2]).astype(np.int32)
    masked_lm_weights = np.ones_like(masked_lm_ids)
    train_data = (data, input_mask, token_type_id, next_sentence_lables,
                  masked_lm_positions, masked_lm_ids, masked_lm_weights)
    for _ in range(32):
        yield train_data

@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_bert_trainer_train_from_instance():
    """
    Feature: Create Trainer From Instance
    Description: Test Trainer API to train from self-define instance API.
    Expectation: TypeError
    """
    # example bert:
    context_config = ContextConfig(device_id=0, device_target='Ascend', mode=0)
    init_context(use_parallel=False, context_config=context_config)

    # Config definition
    runner_config = RunnerConfig(epochs=1, batch_size=16, sink_mode=True, per_epoch_size=2)
    config = ConfigArguments(seed=2022, runner_config=runner_config)

    # Model
    bert_model = BertForPretraining()

    # Dataset and operations
    dataset = GeneratorDataset(generator, column_names=["input_ids", "input_mask", "segment_ids",
                                                        "next_sentence_labels", "masked_lm_positions",
                                                        "masked_lm_ids", "masked_lm_weights"])
    dataset = dataset.batch(batch_size=16)

    # optimizer
    lr_schedule = WarmUpLR(learning_rate=0.001, warmup_steps=100)
    optimizer = AdamWeightDecay(beta1=0.009, beta2=0.999,
                                learning_rate=lr_schedule,
                                params=bert_model.trainable_params())

    # callback
    loss_cb = LossMonitor(per_print_times=2)
    time_cb = TimeMonitor()
    callbacks = [loss_cb, time_cb]

    mlm_trainer = Trainer(task_name='masked_language_modeling',
                          model=bert_model, # model and loss
                          config=config,
                          optimizers=optimizer,
                          train_dataset=dataset,
                          callbacks=callbacks)
    mlm_trainer.train(resume_from_checkpoint=False)
