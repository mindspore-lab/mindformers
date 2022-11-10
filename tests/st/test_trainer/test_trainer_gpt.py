# Copyright 2022 Huawei Technologies Co., Ltd
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
Test module for testing the Trainer
How to run this:
pytest tests/test_trainer.py
"""
import os
import numpy as np
import pytest
from mindspore.dataset import GeneratorDataset

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_trainer_gpt_train():
    """
    Feature: The GPT training test using CPU from python class
    Description: Using cpu to train GPT without basic error
    Expectation: The returned ret is not 0.
    """
    from mindtransformer.trainer import Trainer, TrainingConfig

    class GPTTrainer(Trainer):
        """GPT trainer"""
        def build_model(self, model_config):
            from mindtransformer.models.gpt import GPTWithLoss
            my_net = GPTWithLoss(model_config)
            return my_net

        def build_model_config(self):
            from mindtransformer.models.gpt import GPTConfig
            bs = self.config.global_batch_size
            return GPTConfig(num_layers=1, hidden_size=8, num_heads=1, seq_length=14, batch_size=bs)

        def build_dataset(self):
            def generator():
                data = np.random.randint(low=0, high=15, size=(15,)).astype(np.int32)
                for _ in range(10):
                    yield data

            ds = GeneratorDataset(generator, column_names=["text"])
            ds = ds.batch(self.config.global_batch_size)
            return ds

        def build_lr(self):
            return 0.01

    trainer = GPTTrainer(TrainingConfig(device_target='CPU', epoch_size=2, sink_size=2, global_batch_size=2))
    trainer.train()

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_trainer_gpt_by_cmd():
    """
    Feature: The GPT training test using CPU
    Description: Using cpu to train GPT without basic error
    Expectation: The returned ret is not 0.
    """
    res = os.system("""
            python -m mindtransformer.trainer.trainer \
                --auto_model="gpt" \
                --epoch_size=1 \
                --train_data_path=/home/workspace/mindtransformer/ \
                --optimizer="adam"  \
                --seq_length=14 \
                --parallel_mode="stand_alone" \
                --global_batch_size=2 \
                --vocab_size=50257 \
                --hidden_size=8 \
                --init_loss_scale_value=1 \
                --num_layers=1 \
                --num_heads=2 \
                --full_batch=False \
                --device_target=CPU  """)

    res1 = os.system("""
            python -m mindtransformer.trainer.trainer \
                --auto_model="gpt" \
                --epoch_size=1 \
                --train_data_path=/home/workspace/mindtransformer/ \
                --optimizer="adam"  \
                --seq_length=14 \
                --parallel_mode="stand_alone" \
                --global_batch_size=2 \
                --vocab_size=50257 \
                --hidden_size=8 \
                --init_loss_scale_value=1 \
                --num_layers=1 \
                --num_heads=2 \
                --full_batch=False \
                --device_target=GPU  """)

    assert res == 0
    assert res1 == 0
