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
Test module for testing the glm3 interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_glm3_model/test_training_precision.py
"""
import numpy as np
import pytest

import mindspore as ms
from mindspore import set_seed
from mindspore.dataset import GeneratorDataset

from mindformers import Trainer, TrainingArguments, AutoTokenizer, ChatGLM2Config, ChatGLM2ForConditionalGeneration, \
    FP32StateAdamWeightDecay, CosineWithWarmUpLR
from mindformers.trainer.optimizer_grouped_parameters import get_optimizer_grouped_parameters

from tests.st.training_checker import TrainingChecker


ms.set_context(mode=0)


def generator_train():
    """train dataset generator"""
    seq_len = 512
    step_num = 20
    batch_size = 4
    vocab_size = 65024
    input_ids = np.random.randint(low=0, high=vocab_size, size=(step_num * batch_size, seq_len,)).astype(np.int32)
    labels = np.random.randint(low=0, high=vocab_size, size=(step_num * batch_size, seq_len,)).astype(np.int32)
    for idx in range(len(input_ids)):
        yield input_ids[idx], labels[idx]


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
class TestGLM3TrainingPrecision:
    """A test class for testing training precision."""

    def setup_method(self):
        """init task trainer."""
        set_seed(0)
        np.random.seed(0)

        args = TrainingArguments(batch_size=4, num_train_epochs=1)
        train_dataset = GeneratorDataset(generator_train, column_names=["input_ids", "labels"])
        train_dataset = train_dataset.batch(batch_size=4)

        model_config = ChatGLM2Config(num_layers=2,
                                      seq_length=512,
                                      inner_hidden_size=None,
                                      position_encoding_2d=True,
                                      padded_vocab_size=65024,
                                      use_flash_attention=True)
        model = ChatGLM2ForConditionalGeneration(model_config)

        lr_schedule = CosineWithWarmUpLR(learning_rate=2.e-5, lr_end=1.e-6, warmup_steps=0, total_steps=20)
        group_params = get_optimizer_grouped_parameters(model=model)
        optimizer = FP32StateAdamWeightDecay(params=group_params,
                                             beta1=0.9,
                                             beta2=0.95,
                                             eps=1.e-8,
                                             learning_rate=lr_schedule)

        loss_list_std = [11.284688, 11.295941, 11.279095, 11.300832, 11.294192,
                         11.280991, 11.271296, 11.281275, 11.267831, 11.289062,
                         11.284439, 11.292294, 11.279754, 11.302, 11.288793,
                         11.295439, 11.2769375, 11.281092, 11.303209, 11.262702]
        avg_step_time_std = 140
        callback = TrainingChecker(loss_list_std=loss_list_std, avg_step_time_std=avg_step_time_std)

        self.tokenizer = AutoTokenizer.from_pretrained("glm3_6b")
        self.task_trainer = Trainer(task='text_generation',
                                    model=model,
                                    model_name='glm3_6b',
                                    tokenizer=self.tokenizer,
                                    args=args,
                                    train_dataset=train_dataset,
                                    callbacks=callback,
                                    optimizers=optimizer)

    @pytest.mark.run(order=1)
    def test_train(self):
        """
        Feature: Trainer.train()
        Description: Test trainer for train.
        Expectation: AssertionError
        """
        self.task_trainer.config.runner_config.epochs = 1
        self.task_trainer.config.runner_config.sink_mode = False
        self.task_trainer.config.runner_wrapper.scale_sense.loss_scale_value = 1024
        self.task_trainer.config.callbacks = self.task_trainer.config.callbacks[:1]
        self.task_trainer.train()
