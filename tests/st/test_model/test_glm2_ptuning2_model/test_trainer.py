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
Test module for testing the glm2 p-tuning-v2 interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_glm2_ptuning2_model/test_trainer.py
"""

import numpy as np
import pytest

from mindformers import ChatGLM2Config
from mindformers import Trainer, TrainingArguments
from mindformers.models.glm2.glm2 import ChatGLM2WithPtuning2
from mindformers.pet.pet_config import Ptuning2Config
from mindspore import context
from mindspore.dataset import GeneratorDataset


def generator_train():
    """train dataset generator"""
    seq_len = 65
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    labels = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    train_data = (input_ids, labels)
    for _ in range(32):
        yield train_data


def generator_eval():
    """eval dataset generator"""
    seq_len = 64
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    labels = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    eval_data = (input_ids, labels)
    for _ in range(8):
        yield eval_data


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
class TestGlm2Ptuning2TrainerMethod:
    """A test class for testing pipeline."""

    def setup_method(self):
        """init task trainer."""
        context.set_context(mode=0, device_target="Ascend")
        args = TrainingArguments(batch_size=2)
        train_dataset = GeneratorDataset(generator_train, column_names=["input_ids", "labels"])
        eval_dataset = GeneratorDataset(generator_eval, column_names=["input_ids", "labels"])
        train_dataset = train_dataset.batch(batch_size=2)
        eval_dataset = eval_dataset.batch(batch_size=2)

        # padded_vocab_size=64793 to avoid tokenizer index violation
        config = ChatGLM2Config(num_layers=2, seq_length=65, hidden_size=32, inner_hidden_size=None,
                                num_heads=2, position_encoding_2d=True, padded_vocab_size=64793)
        config.pet_config = Ptuning2Config(pre_seq_len=8)

        model = ChatGLM2WithPtuning2(config=config)
        self.task_trainer = Trainer(task='text_generation',
                                    model=model,
                                    model_name='glm2_6b_ptuning2',
                                    args=args,
                                    train_dataset=train_dataset,
                                    eval_dataset=eval_dataset)

    @pytest.mark.run(order=1)
    def test_finetune(self):
        """
        Feature: Trainer.finetune()
        Description: Test trainer for finetune.
        Expectation: TypeError, ValueError, RuntimeError
        """
        self.task_trainer.config.runner_config.epochs = 1
        self.task_trainer.finetune()

    @pytest.mark.run(order=2)
    def test_predict(self):
        """
        Feature: Trainer.predict()
        Description: Test trainer for predict.
        Expectation: TypeError, ValueError, RuntimeError
        """
        self.task_trainer.predict(input_data="hello world!", max_length=20, repetition_penalty=1,
                                  top_k=3, top_p=1)
