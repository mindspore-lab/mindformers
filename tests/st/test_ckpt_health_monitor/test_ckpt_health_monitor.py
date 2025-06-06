# Copyright 2025 Huawei Technologies Co., Ltd
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
Test module for testing the trainer interface used for mindformers.
How to run this:
    pytest tests/st/test_trainer/test_trainer.py
"""
import json
import os
import copy
import numpy as np
import pytest

import mindspore as ms
from mindspore.dataset import GeneratorDataset

from mindformers import LlamaConfig, LlamaForCausalLM
from mindformers import Trainer, TrainingArguments
from mindformers.core.callback import MFLossMonitor, TrainingStateMonitor, CheckpointMonitor

ms.set_context(mode=0)

EPOCHS = 1
NUM_LAYERS = 1
HIDDEN_SIZE = 16
NUM_HEADS = 2
SEQ_LENGTH = 32
TRAIN_BATCH_SIZE = 2
EVAL_STEPS = 4
DATA_SIZE = 8


def generator_train():
    """train dataset generator"""
    seq_len = SEQ_LENGTH + 1
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    for _ in range(DATA_SIZE):
        yield input_ids


def generator_eval():
    """eval dataset generator"""
    seq_len = SEQ_LENGTH
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    for _ in range(DATA_SIZE):
        yield input_ids

MODEL_CONFIG = LlamaConfig(num_layers=NUM_LAYERS, hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS,
                           seq_length=SEQ_LENGTH, pa_block_size=1, pa_num_blocks=1)
MODEL = LlamaForCausalLM(MODEL_CONFIG)
MODEL_CONFIG.checkpoint_name_or_path = ""


TRAIN_DATASET = GeneratorDataset(generator_train, column_names=["input_ids"])
TRAIN_DATASET_FOR_TRAINER_WITH_ARGS = TRAIN_DATASET.batch(batch_size=TRAIN_BATCH_SIZE)
TRAIN_DATASET_FOR_TRAINER_WITH_ARGS_ = copy.deepcopy(TRAIN_DATASET_FOR_TRAINER_WITH_ARGS)

ARGS = TrainingArguments(batch_size=4, num_train_epochs=1)

def run_trainer(args, task, model, train_dataset, embedding_local_norm_threshold):
    """static method of running trainer."""
    callbacks = []
    callbacks.append(MFLossMonitor(learning_rate=1.0, origin_epochs=1, dataset_size=DATA_SIZE))
    callbacks.append(TrainingStateMonitor(origin_epochs=1, dataset_size=DATA_SIZE,
                                          embedding_size=HIDDEN_SIZE * 32000,
                                          use_local_norm=True))
    callbacks.append(CheckpointMonitor(health_ckpts_record_dir="./output",
                                       embedding_local_norm_threshold=embedding_local_norm_threshold,
                                       embedding_size=HIDDEN_SIZE*32000,
                                       use_checkpoint_health_monitor=True))
    trainer = Trainer(args=args, task=task, model=model,
                      train_dataset=train_dataset, callbacks=callbacks)
    trainer.config.runner_wrapper.local_norm = True
    trainer.train()

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_use_ckpt_health_monitor_unhealthy_ckpt():
    """
    Feature: Trainer
    Description: Test trainer with use_ckpt_health_monitor.
    Expectation: no exception
    """
    run_trainer(ARGS, "text_generation", MODEL, TRAIN_DATASET_FOR_TRAINER_WITH_ARGS, 0.0)

    with open("./output/log/rank_0/info.log", 'r') as file:
        content = file.read()
        assert "embedding_local_norm: " in content
        file.close()

    assert os.path.exists("./output/health_ckpts.json")
    with open("./output/health_ckpts.json", 'r') as file:
        json_data = json.load(file)
        assert json_data is not None
        not_health_ckpts = [item.get("ckpt_name") for item in json_data if
                            item.get("is_health", -1) != 0]
        assert not not_health_ckpts
        file.close()
