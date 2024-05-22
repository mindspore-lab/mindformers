# Copyright 2024 Huawei Technologies Co., Ltd
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
Test module for testing resume training.
How to run this:
pytest tests/st/test_resume/test_resume.py
"""
import os
import numpy as np
import pytest
import mindspore as ms

from mindspore.dataset import GeneratorDataset

from mindformers.tools.utils import LOCAL_DEFAULT_PATH, get_real_rank
from mindformers.trainer import Trainer, TrainingArguments
from mindformers.models.gpt2 import GPT2LMHeadModel, GPT2Config
from utils import extract_loss_values

ms.set_context(mode=0)

def generator():
    """dataset generator"""
    np.random.seed(42)
    seq_len = 1025
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    input_mask = np.ones_like(input_ids)
    train_data = (input_ids, input_mask)
    for _ in range(32):
        yield train_data

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_gpt_trainer_train_from_instance():
    """
    Feature: Create Trainer From Instance
    Description: Test Trainer API to train from self-define instance API.
    Expectation: TypeError
    """
    # Config definition
    config = TrainingArguments(
        num_train_epochs=2,
        batch_size=8,
        save_steps=1,
        save_directory=os.path.join(LOCAL_DEFAULT_PATH, "test_resume"),
    )

    # Model
    model_config = GPT2Config(num_layers=2)
    model = GPT2LMHeadModel(model_config)

    # Dataset and operations
    dataset = GeneratorDataset(generator, column_names=["input_ids", "input_mask"])
    dataset = dataset.batch(batch_size=8)

    trainer = Trainer(model=model,
                      args=config,
                      train_dataset=dataset)
    trainer.train(train_checkpoint=False)

    checkpoint_dir = os.path.join(LOCAL_DEFAULT_PATH, "test_resume", "checkpoint",
                                  "rank_{}".format(get_real_rank()))
    output_checkpoint_path = [
        checkpoint for checkpoint in os.listdir(checkpoint_dir)
        if checkpoint.endswith('.ckpt')
    ]
    output_checkpoint_path = sorted(output_checkpoint_path,
                                    key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
    for _ in range(2):
        os.remove(os.path.join(checkpoint_dir, output_checkpoint_path.pop()))

    # Dataset and operations
    dataset = GeneratorDataset(generator, column_names=["input_ids", "input_mask"])
    dataset = dataset.batch(batch_size=8)

    trainer = Trainer(model=model,
                      args=config,
                      train_dataset=dataset)
    trainer.train(resume_from_checkpoint=os.path.join(LOCAL_DEFAULT_PATH, "test_resume", "checkpoint"),
                  resume_training=True)
    loss = extract_loss_values(f"{LOCAL_DEFAULT_PATH}/log/rank_0/info.log")
    assert abs(loss[-4] - loss[-2]) < 0.005
    assert abs(loss[-3] - loss[-1]) < 0.005
