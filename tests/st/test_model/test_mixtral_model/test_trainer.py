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
Test module for testing the mixtral interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_mixtral_model/test_trainer.py
"""
import numpy as np
import pytest

import mindspore as ms

from mindspore.dataset import GeneratorDataset
from mindformers.models.llama.llama import LlamaForCausalLM
from mindformers.models.llama.llama_config import LlamaConfig
from mindformers.models.llama.llama_tokenizer import LlamaTokenizer
from mindformers import Trainer, TrainingArguments
from mindformers.modules.transformer.moe import MoEConfig

from tests.utils.model_tester import create_tokenizer

ms.set_context(mode=0)

_, tokenizer_model_path = create_tokenizer()


def generator_train():
    """train dataset generator"""
    seq_len = 513
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    # input_mask = np.ones_like(input_ids)
    train_data = (input_ids)
    for _ in range(16):
        yield train_data


def generator_eval():
    """eval dataset generator"""
    seq_len = 512
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    # input_mask = np.ones_like(input_ids)
    train_data = (input_ids)
    for _ in range(16):
        yield train_data


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
class TestMixtralTrainerMethod:
    """A test class for testing pipeline."""

    def setup_method(self):
        """init task trainer."""
        args = TrainingArguments(batch_size=4, num_train_epochs=1)
        train_dataset = GeneratorDataset(generator_train, column_names=["input_ids"])
        eval_dataset = GeneratorDataset(generator_eval, column_names=["input_ids"])
        train_dataset = train_dataset.batch(batch_size=4)
        eval_dataset = eval_dataset.batch(batch_size=4)

        moe_config = MoEConfig(expert_num=8,
                               capacity_factor=1.1,
                               aux_loss_factor=0.05,
                               routing_policy="TopkRouterV2",
                               enable_sdrop=True)
        model_config = LlamaConfig(num_layers=2, hidden_size=32, num_heads=2, seq_length=512, moe_config=moe_config,
                                   vocab_size=200)
        model = LlamaForCausalLM(model_config)

        self.task_trainer = Trainer(task='text_generation',
                                    model=model,
                                    args=args,
                                    train_dataset=train_dataset,
                                    eval_dataset=eval_dataset,
                                    tokenizer=LlamaTokenizer(
                                        vocab_file=tokenizer_model_path)
                                    )

    @pytest.mark.run(order=1)
    def test_train(self):
        """
        Feature: Trainer.train()
        Description: Test trainer for train.
        Expectation: TypeError, ValueError, RuntimeError
        """
        self.task_trainer.config.runner_config.epochs = 1
        self.task_trainer.train()

    @pytest.mark.run(order=2)
    def test_eval(self):
        """
        Feature: Trainer.evaluate()
        Description: Test trainer for evaluate.
        Expectation: TypeError, ValueError, RuntimeError
        """
        self.task_trainer.model.set_train(False)
        self.task_trainer.evaluate()

    @pytest.mark.run(order=3)
    def test_predict(self):
        """
        Feature: Trainer.predict()
        Description: Test trainer for predict.
        Expectation: TypeError, ValueError, RuntimeError
        """
        self.task_trainer.predict(input_data="hello world!", max_length=20, repetition_penalty=1, top_k=3, top_p=1)

    @pytest.mark.run(order=4)
    def test_finetune(self):
        """
        Feature: Trainer.finetune()
        Description: Test trainer for finetune.
        Expectation: TypeError, ValueError, RuntimeError
        """
        self.task_trainer.config.runner_config.epochs = 1
        self.task_trainer.finetune()
