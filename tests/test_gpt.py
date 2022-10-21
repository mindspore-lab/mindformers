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
Test module for testing the network
How to run this:
pytest tests/test_gpt.py
"""

import os

import numpy as np

from mindspore.dataset import GeneratorDataset
from mindspore.mindrecord import FileWriter


def test_gpt_network():
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
            python -m transformer.trainer.trainer \
                --auto_model="gpt" \
                --epoch_size=1 \
                --train_data_path=tests/ \
                --optimizer="adam"  \
                --seq_length=14 \
                --parallel_mode="stand_alone" \
                --global_batch_size=1 \
                --vocab_size=60 \
                --hidden_size=32 \
                --init_loss_scale_value=1 \
                --num_layers=1 \
                --num_heads=16 \
                --full_batch=False \
                --device_target=CPU  """)
    assert res == 0
