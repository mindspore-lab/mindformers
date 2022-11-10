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
import numpy as np
import pytest
from mindspore.dataset import GeneratorDataset

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_trainer_bert_train():
    """
    Feature: The BERT training test using CPU from python class
    Description: Using cpu to train BERT without basic error
    Expectation: The returned ret is not 0.
    """
    from mindtransformer.trainer import Trainer, TrainingConfig

    class BERTTrainer(Trainer):
        """BERT trainer"""
        def build_model(self, model_config):
            from mindtransformer.models.bert import BertNetworkWithLoss
            my_net = BertNetworkWithLoss(model_config)
            return my_net

        def build_model_config(self):
            from mindtransformer.models.bert import BertConfig
            return BertConfig(num_layers=1, embedding_size=8, num_heads=1, seq_length=15)

        def build_dataset(self):
            """build dataset"""
            def generator():
                data = np.random.randint(low=0, high=15, size=(15,)).astype(np.int32)
                input_mask = np.ones_like(data)
                token_type_id = np.zeros_like(data)
                next_sentence_lables = np.array([1]).astype(np.int32)
                masked_lm_positions = np.array([1, 2]).astype(np.int32)
                masked_lm_ids = np.array([1, 2]).astype(np.int32)
                masked_lm_weights = np.ones_like(masked_lm_ids)
                train_data = (data, input_mask, token_type_id, next_sentence_lables,
                              masked_lm_positions, masked_lm_ids, masked_lm_weights)
                for _ in range(16):
                    yield train_data

            ds = GeneratorDataset(generator, column_names=["input_ids", "input_mask", "segment_ids",
                                                           "next_sentence_labels", "masked_lm_positions",
                                                           "masked_lm_ids", "masked_lm_weights"])
            ds = ds.batch(4)
            return ds

        def build_lr(self):
            return 0.01

    trainer = BERTTrainer(TrainingConfig(device_target='CPU', epoch_size=1, sink_size=4))
    trainer.train()
