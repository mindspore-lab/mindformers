# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Data generation utilities for deepseekv3 test"""
from functools import partial
import numpy as np

from mindspore.dataset import GeneratorDataset

from mindformers.dataset.causal_language_model_dataset import asl_batch_wrapper

def generate_data(seq_len, vocab_size, batch_size=4, step_num=20, use_actual_seq_len=False):
    """generate data for testing model."""
    np.random.seed(0)
    input_ids = np.random.randint(low=0, high=vocab_size, size=(step_num * batch_size, seq_len,)).astype(np.int32)
    actual_seq_len = [i for i in range(0, seq_len, seq_len // 16)]
    actual_seq_len = np.array([*actual_seq_len[1:], seq_len], np.int32)

    for input_data in input_ids:
        data = [input_data, input_data]
        if use_actual_seq_len:
            data.append(actual_seq_len)
        yield data


def get_tnd_dataset(seq_length, vocab_size, micro_batch_num, batch_size, step_num):
    """build dataset with actual_seq_len for model training."""
    prepare_data = partial(generate_data,
                           seq_len=seq_length,
                           vocab_size=vocab_size,
                           batch_size=batch_size,
                           step_num=step_num,
                           use_actual_seq_len=True)
    per_batch_map_func = partial(
        asl_batch_wrapper,
        micro_batch_num=micro_batch_num
    )
    column_names = ['input_ids', 'labels', 'actual_seq_len']
    dataset = GeneratorDataset(prepare_data, column_names=column_names)
    dataset = dataset.batch(
        batch_size=batch_size,
        output_columns=column_names,
        per_batch_map=per_batch_map_func
    )
    return dataset


def get_dataset(seq_length, vocab_size, batch_size, step_num):
    """build dataset for model training."""

    prepare_data = partial(generate_data,
                           seq_len=seq_length,
                           vocab_size=vocab_size,
                           batch_size=batch_size,
                           step_num=step_num)

    dataset = GeneratorDataset(prepare_data, column_names=['input_ids', 'labels'])
    dataset = dataset.batch(batch_size=batch_size)
    return dataset
