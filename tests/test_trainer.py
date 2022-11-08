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

from mindspore.mindrecord import FileWriter
from mindspore.dataset import GeneratorDataset


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
            return GPTConfig(num_layers=1, hidden_size=8, num_heads=1, seq_length=14)

        def build_dataset(self):
            def generator():
                data = np.random.randint(low=0, high=15, size=(15,)).astype(np.int32)
                for _ in range(10):
                    yield data

            ds = GeneratorDataset(generator, column_names=["text"])
            ds = ds.batch(2)
            return ds

        def build_lr(self):
            return 0.01

    trainer = GPTTrainer(TrainingConfig(device_target='CPU', epoch_size=2, sink_size=2))
    trainer.train()


def test_trainer_gpt_by_cmd():
    """
    Feature: The GPT training test using CPU
    Description: Using cpu to train GPT without basic error
    Expectation: The returned ret is not 0.
    """
    def generator():
        data = np.random.randint(low=0, high=15, size=(15)).astype(np.int32)
        for _ in range(10):
            yield data

    ds = GeneratorDataset(generator, column_names=["text"])

    data_record_path = 'tests/test_gpt_mindrecord'
    writer = FileWriter(file_name=data_record_path, shard_num=1, overwrite=True)
    data_schema = {"text": {"type": "int32", "shape": [-1]}}
    writer.add_schema(data_schema, "test_schema")
    for item in ds.create_dict_iterator():
        for k in item.keys():
            item[k] = item[k].asnumpy()
        writer.write_raw_data([item])
    writer.commit()

    res = os.system("""
            python -m mindtransformer.trainer.trainer \
                --auto_model="gpt" \
                --epoch_size=1 \
                --train_data_path=tests/ \
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
    os.remove(data_record_path)
    os.remove(data_record_path + '.db')
    assert res == 0
