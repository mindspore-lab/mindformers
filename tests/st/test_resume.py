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
pytest tests/st/test_resume.py
"""
import os
import numpy as np
import pytest
import mindspore as ms

from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.dataset import GeneratorDataset

from mindformers.tools.utils import LOCAL_DEFAULT_PATH, get_real_rank
from mindformers.trainer import Trainer
from mindformers.models.gpt2 import GPT2LMHeadModel, GPT2Config
from mindformers import CheckpointMointor, TrainingArguments, PolynomialWithWarmUpLR
from mindformers.core.optim import FusedAdamWeightDecay

ms.set_context(mode=0)


def generator():
    """dataset generator"""
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
    config = TrainingArguments(num_train_epochs=2, batch_size=8, sink_mode=True, sink_size=2, seed=2022)

    # Model
    model_config = GPT2Config(num_layers=2)
    gpt_model = GPT2LMHeadModel(model_config)

    # Dataset and operations
    dataset = GeneratorDataset(generator, column_names=["input_ids", "input_mask"])
    dataset = dataset.batch(batch_size=8)

    # optimizer
    lr_schedule = PolynomialWithWarmUpLR(learning_rate=0.0001, lr_end=0.00001, warmup_steps=0, total_steps=512)
    optimizer = FusedAdamWeightDecay(beta1=0.009, beta2=0.999,
                                     learning_rate=lr_schedule,
                                     params=gpt_model.trainable_params())

    # callback
    loss_cb = LossMonitor(per_print_times=2)
    time_cb = TimeMonitor()
    ckpt_cb = CheckpointMointor(directory=os.path.join(LOCAL_DEFAULT_PATH, "test_resume"))
    callbacks = [loss_cb, time_cb, ckpt_cb]

    lm_trainer = Trainer(model=gpt_model,
                         args=config,
                         optimizers=optimizer,
                         train_dataset=dataset,
                         callbacks=callbacks,
                         task="text_generation")
    lm_trainer.train(train_checkpoint=False)

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

    # callback
    loss_cb = LossMonitor(per_print_times=2)
    time_cb = TimeMonitor()
    ckpt_cb = CheckpointMointor(directory=os.path.join(LOCAL_DEFAULT_PATH, "test_resume"))
    callbacks = [loss_cb, time_cb, ckpt_cb]

    lm_trainer = Trainer(model=gpt_model,
                         args=config,
                         optimizers=optimizer,
                         train_dataset=dataset,
                         callbacks=callbacks,
                         task="text_generation")
    lm_trainer.train(train_checkpoint=os.path.join(LOCAL_DEFAULT_PATH, "test_resume", "checkpoint"),
                     resume_training=True)
