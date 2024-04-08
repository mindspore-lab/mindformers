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
Test module for testing the internlm interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_internlm_model/test_training_precision.py
"""
import sys
import numpy as np
import pytest

import mindspore as ms
from mindspore import set_seed
from mindspore.dataset import GeneratorDataset

from mindformers import Trainer, TrainingArguments, CosineWithWarmUpLR, FP32StateAdamWeightDecay
from mindformers.trainer.optimizer_grouped_parameters import get_optimizer_grouped_parameters

from tests.st.training_checker import TrainingChecker

for path in sys.path:
    if path.endswith('/testcases'):
        sys.path.append(path + '/research')
    if path.endswith('/research'):
        sys.path.append(path + '/internlm')
# pylint: disable=C0413
from research.internlm.internlm import InternLMForCausalLM
from research.internlm.internlm_config import InternLMConfig


ms.set_context(mode=0)


def generator_train():
    """train dataset generator"""
    seq_len = 513
    step_num = 20
    batch_size = 4
    vocab_size = 103168
    input_ids = np.random.randint(low=0, high=vocab_size, size=(step_num * batch_size, seq_len,)).astype(np.int32)
    for idx in range(len(input_ids)):
        yield input_ids[idx]


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
class TestInternLMTrainingPrecision:
    """A test class for testing training precision."""

    def setup_method(self):
        """init task trainer."""
        set_seed(0)
        np.random.seed(0)

        args = TrainingArguments(batch_size=4, num_train_epochs=1)
        train_dataset = GeneratorDataset(generator_train, column_names=["input_ids"])
        train_dataset = train_dataset.batch(batch_size=4)

        model_config = InternLMConfig(num_layers=2, seq_length=512, use_flash_attention=True)
        model = InternLMForCausalLM(model_config)

        lr_schedule = CosineWithWarmUpLR(learning_rate=1.e-5, warmup_steps=0, total_steps=20)
        group_params = get_optimizer_grouped_parameters(model=model)
        optimizer = FP32StateAdamWeightDecay(params=group_params,
                                             beta1=0.9,
                                             beta2=0.999,
                                             eps=1.e-8,
                                             learning_rate=lr_schedule)

        loss_list_std = [11.7715, 11.751391, 11.737675, 11.73484, 11.7693815,
                         11.780886, 11.729794, 11.737111, 11.727144, 11.733832,
                         11.718602, 11.77964, 11.751205, 11.736936, 11.757384,
                         11.743711, 11.752291, 11.760548, 11.750416, 11.744327]
        avg_step_time_std = 179
        callback = TrainingChecker(loss_list_std=loss_list_std, avg_step_time_std=avg_step_time_std)

        self.task_trainer = Trainer(task='text_generation',
                                    model=model,
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
