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
Test module for testing the bloom interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_bloom_model/test_trainer.py
"""
import numpy as np
import pytest

import mindspore as ms
from mindspore.dataset import GeneratorDataset

from mindformers import AutoTokenizer
from mindformers import BloomLMHeadModel, BloomConfig
from mindformers import Trainer, TrainingArguments

ms.set_context(mode=0)


def generator_train():
    """train dataset generator"""
    seq_len = 513
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    for _ in range(16):
        yield input_ids


def generator_eval():
    """eval dataset generator"""
    seq_len = 512
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    for _ in range(16):
        yield input_ids


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
class TestBloomTrainerMethod:
    """A test class for testing pipeline."""

    def setup_method(self):
        """init task trainer."""
        args = TrainingArguments(batch_size=4, num_train_epochs=1)
        train_dataset = GeneratorDataset(generator_train, column_names=["input_ids"])
        eval_dataset = GeneratorDataset(generator_eval, column_names=["input_ids"])
        train_dataset = train_dataset.batch(batch_size=2)
        eval_dataset = eval_dataset.batch(batch_size=2)

        model_config = BloomConfig(num_layers=2, hidden_size=32, num_heads=2, seq_length=512, max_decode_length=10)
        model = BloomLMHeadModel(model_config)
        tokenizer = AutoTokenizer.from_pretrained("bloom_560m")
        self.task_trainer = Trainer(task='text_generation',
                                    model=model,
                                    args=args,
                                    tokenizer=tokenizer,
                                    train_dataset=train_dataset,
                                    eval_dataset=eval_dataset)

    @pytest.mark.run(order=1)
    def test_train(self):
        """
        Feature: Trainer.train()
        Description: Test trainer for train.
        Expectation: TypeError, ValueError, RuntimeError
        """
        self.task_trainer.train()

    # @pytest.mark.run(order=2)
    # def test_eval(self):
    #     """
    #     Feature: Trainer.evaluate()
    #     Description: Test trainer for evaluate.
    #     Expectation: TypeError, ValueError, RuntimeError
    #     """
    #     self.task_trainer.evaluate()

    @pytest.mark.run(order=3)
    def test_predict(self):
        """
        Feature: Trainer.predict()
        Description: Test trainer for predict.
        Expectation: TypeError, ValueError, RuntimeError
        """
        self.task_trainer.predict(input_data="hello world!", add_special_tokens=False, max_length=20)

    # @pytest.mark.run(order=4)
    # def test_finetune(self):
    #     """
    #     Feature: Trainer.finetune()
    #     Description: Test trainer for finetune.
    #     Expectation: TypeError, ValueError, RuntimeError
    #     """
    #     self.task_trainer.finetune()
