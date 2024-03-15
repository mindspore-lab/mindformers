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
Test module for testing the codellama interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_codellama_model/test_training_precision.py
"""
import numpy as np
import pytest

import mindspore as ms
from mindspore import set_seed

from mindspore.dataset import GeneratorDataset
from mindformers.core import build_lr, build_optim
from mindformers.models.llama.llama import LlamaForCausalLM
from mindformers.models.llama.llama_config import LlamaConfig
from mindformers import Trainer, TrainingArguments
from mindformers.trainer.optimizer_grouped_parameters import get_optimizer_grouped_parameters
from tests.st.training_checker import TrainingChecker

ms.set_context(mode=0)


def generator_train():
    """train dataset generator"""
    seq_len = 513
    step_num = 20
    batch_size = 4
    vocab_size = 32000
    input_ids = np.random.randint(low=0, high=vocab_size, size=(step_num * batch_size, seq_len,)).astype(np.int32)
    for idx in range(len(input_ids)):
        yield input_ids[idx]


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
class TestCodeLlamaTrainingPrecision:
    """A test class for testing training precision."""

    def setup_method(self):
        """init task trainer."""
        set_seed(0)
        np.random.seed(0)

        args = TrainingArguments(batch_size=4, num_train_epochs=1)
        train_dataset = GeneratorDataset(generator_train, column_names=["input_ids"])
        train_dataset = train_dataset.batch(batch_size=4)

        model_config = LlamaConfig(num_layers=2,
                                   hidden_size=8192,
                                   num_heads=64,
                                   seq_length=512,
                                   vocab_size=32000,
                                   multiple_of=256,
                                   rms_norm_eps=1.0e-5,
                                   use_flash_attention=True)
        model = LlamaForCausalLM(model_config)

        lr_config = {'type': 'CosineWithWarmUpLR',
                     'learning_rate': 2.e-5,
                     'lr_end': 1.e-6,
                     'warmup_steps': 0,
                     'total_steps': 20}
        lr_schedule = build_lr(lr_config)
        group_params = get_optimizer_grouped_parameters(model=model)
        optim_config = {'type': 'FP32StateAdamWeightDecay', 'beta1': 0.9, 'beta2': 0.95, 'eps': 1.e-8}
        optimizer = build_optim(config=optim_config, default_args={"params": group_params,
                                                                   "learning_rate": lr_schedule})

        loss_list_std = [10.781137, 10.792782, 10.777257, 10.81901, 10.753006,
                         10.784267, 10.805832, 10.778866, 10.800653, 10.787513,
                         10.789893, 10.826312, 10.765125, 10.783563, 10.802842,
                         10.790674, 10.775396, 10.796202, 10.731129, 10.76932]
        avg_step_time_std = 339
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
