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
pytest tests/test_trainer_t5.py
"""
import numpy as np
import pytest
from mindspore.dataset import GeneratorDataset


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_trainer_t5_train():
    """
    Feature: The T5 training test using CPU from python class
    Description: Using cpu to train T5 without basic error
    Expectation: The returned ret is not 0.
    """
    from mindtransformer.trainer import Trainer, TrainingConfig

    class T5Trainer(Trainer):
        """GPT trainer"""
        def build_model(self, model_config):
            from mindtransformer.models.t5 import TransformerNetworkWithLoss
            my_net = TransformerNetworkWithLoss(model_config)
            return my_net

        def build_model_config(self):
            from mindtransformer.models.t5 import TransformerConfig
            bs = self.config.global_batch_size
            return TransformerConfig(num_hidden_layers=1, num_heads=1, seq_length=16, max_decode_length=8,
                                     batch_size=bs)

        def build_dataset(self):
            """Build the fake dataset."""
            columns_list = ["input_ids", "attention_mask", "labels"]

            def generator():
                input_ids = np.random.randint(low=0, high=15, size=(16,)).astype(np.int32)
                attention_mask = np.random.randint(low=0, high=15, size=(16,)).astype(np.int32)
                labels = np.random.randint(low=0, high=15, size=(8,)).astype(np.int32)

                for _ in range(2):
                    yield input_ids, attention_mask, labels

            ds = GeneratorDataset(generator, column_names=columns_list)
            ds = ds.batch(self.config.global_batch_size)
            return ds

        def build_lr(self):
            return 0.01

    trainer = T5Trainer(TrainingConfig(device_target='CPU', epoch_size=1, sink_size=2, global_batch_size=2,
                                       save_checkpoint=False))
    trainer.train()
