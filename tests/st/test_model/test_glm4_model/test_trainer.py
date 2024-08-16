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
Test module for testing the glm4 interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_glm4_model/test_trainer.py
"""
import pytest

from mindspore import context

from mindformers import ChatGLM2Config, ChatGLM2ForConditionalGeneration
from mindformers import Trainer, TrainingArguments


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
class TestGLM4TrainerMethod:
    """A test class for testing pipeline."""

    def setup_method(self):
        """init task trainer."""
        context.set_context(mode=0, device_target="Ascend")

        args = TrainingArguments(num_train_epochs=1, batch_size=2)

        model_config = ChatGLM2Config(num_layers=2, seq_length=128, hidden_size=32, inner_hidden_size=None,
                                      num_heads=2, rope_ratio=500)
        model_config.vocab_size = 15152
        model = ChatGLM2ForConditionalGeneration(model_config)
        self.task_trainer = Trainer(task='text_generation',
                                    model=model,
                                    args=args)

    @pytest.mark.run(order=1)
    def test_predict(self):
        """
        Feature: Trainer.predict()
        Description: Test trainer for predict.
        Expectation: TypeError, ValueError, RuntimeError
        """
        self.task_trainer.predict(input_data="使用python编写快速排序代码", max_length=30, repetition_penalty=1, top_k=3,
                                  top_p=1)
