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

"""Dataset module."""

import numpy as np

import mindspore as ms
import mindspore.dataset as ds
from mindspore import Tensor

from mindformers.tools.logger import logger
from mindformers.experimental.parallel_core.pynative.parallel_state import get_dp_rank, get_dp_world_size


class RandomData:
    """
    generate a test dataset
    """
    def __init__(self, seed, samples=100, vocab_size=200, seq_length=512, hidden_size=1024):
        super().__init__()
        np.random.seed(seed)
        self.input_data = np.random.rand(samples, seq_length, hidden_size)
        self.labels = np.random.randint(0, vocab_size, (samples, seq_length))
        self.dataset_size = self.input_data.shape[0]

    def __getitem__(self, index):
        return (Tensor(self.input_data[index], ms.float32), Tensor(self.labels[index], ms.int32))

    def __len__(self):
        return self.dataset_size


def batch_and_generate_attention_mask(hidden_states, labels, rank, stride, use_flash_attn=False):
    """
    Generate position_id and attention_mask according to hidden_states considering eod reset
    """
    seq_length = hidden_states.shape[1]
    batch_hidden_states = hidden_states[rank * stride : (rank + 1) * stride]
    batch_labels = labels[rank * stride : (rank + 1) * stride]
    batch_attention_mask = np.ones((stride, seq_length, seq_length), dtype=np.uint8)
    if use_flash_attn:
        tril_dev = 1 - np.tril(np.ones(shape=(seq_length, seq_length), dtype=np.uint8))
    else:
        tril_dev = np.tril(np.ones(shape=(seq_length, seq_length), dtype=np.uint8))

    # Loop through batches
    for bs_i, _ in enumerate(range(len(batch_hidden_states))):
        # Get normal position_ids and attention_mask
        batch_attention_mask[bs_i] = tril_dev

    # [B,S,S] -> [B,N,S,S]
    batch_attention_mask = np.expand_dims(batch_attention_mask, axis=1)
    return batch_hidden_states, batch_labels, batch_attention_mask


def get_random_dataset(dataset_config, model_config, seed=42, training_iters=20, use_flash_atten=True):
    """ Get the random dataset iterator."""
    batch_size = dataset_config.batch_size
    micro_batch_num = dataset_config.micro_batch_num
    shuffle = dataset_config.shuffle
    drop_remainder = dataset_config.drop_remainder

    samples = batch_size * micro_batch_num * get_dp_world_size() * training_iters

    dataset = RandomData(seed, samples, model_config.vocab_size, model_config.seq_length, model_config.hidden_size)
    dataset = ds.GeneratorDataset(dataset, column_names=["hidden_states", "labels"], shuffle=shuffle)
    per_rank_batch_size = batch_size * micro_batch_num
    global_batch_size = per_rank_batch_size * get_dp_world_size()
    logger.warning(f"dataset global batch size: {global_batch_size}")
    dataset = dataset.batch(global_batch_size, drop_remainder=drop_remainder)

    def map_func(hidden_states, labels):
        return batch_and_generate_attention_mask(
            hidden_states,
            labels,
            get_dp_rank(),
            per_rank_batch_size,
            use_flash_atten
        )

    dataset = dataset.map(
        operations=map_func,
        input_columns=["hidden_states", "labels"],
        output_columns=["hidden_states", "labels", "attention_mask"],
    )

    return dataset, dataset
