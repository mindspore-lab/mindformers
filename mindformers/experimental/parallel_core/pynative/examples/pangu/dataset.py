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
"""Dataset for training pangu"""

from pathlib import Path

import numpy as np
import mindspore.dataset as ds

from mindformers.experimental.parallel_core.pynative.parallel_state import get_data_parallel_rank, get_data_parallel_world_size
from mindformers.tools import logger


def generate_position_attention_map(input_ids, eod_id, rank, stride):
    """
    Generate position_id and attention_mask according to input_ids considering eod reset

    Inputs:
        input_ids: the input token ids
        eod_id: the id for <EOD>
        stride: the slice value for each rank
    returns:
        input_ids: the input token ids
        position_id: the position ids cosidering eod reset
        attention_mask: the attention mask considering eod reset
    """
    seq_length = input_ids.shape[1] - 1
    # Initialize position_ids and attention_mask
    batch_input_ids = input_ids[rank * stride : (rank + 1) * stride]
    batch_position_ids = np.ones((stride, seq_length), dtype=np.int32)
    batch_attention_mask = np.ones((stride, seq_length, seq_length), dtype=np.int8)
    tril_dev = np.tril(np.ones(shape=(seq_length, seq_length), dtype=np.int8))

    # Loop through batches
    for bs_i, _ in enumerate(range(len(batch_input_ids))):
        # Get normal position_ids and attention_mask
        local_ids = batch_input_ids[bs_i]
        batch_attention_mask[bs_i] = tril_dev
        batch_position_ids[bs_i] = np.arange(seq_length, dtype=np.int32)
        # Find eod_of_document
        eod_index = batch_position_ids[bs_i, local_ids[:-1] == eod_id].astype(np.int32)
        prev_index = 0
        for i in range(eod_index.size):
            # Reset position_ids and attention_mask considering EOD
            index = eod_index[i]
            batch_attention_mask[bs_i, (index + 1) :, : (index + 1)] = 0
            batch_position_ids[bs_i, (index + 1) :] -= index + 1 - prev_index
            prev_index = index + 1

    # [B,S,S] -> [B,N,S,S]
    batch_attention_mask = np.expand_dims(batch_attention_mask, axis=1)
    return batch_input_ids, batch_position_ids, batch_attention_mask


# training config
def get_individual_dataset(
        dataset_dir,
        batch_size,
        micro_batch_num,
        eod_id,
        shuffle=False,
        drop_remainder=True,
        mode="train",
):
    """
    Creates and returns a dataset for a specific mode (train, valid, or test).

    Args:
        dataset_dir (str): The directory path where the dataset is located.
        batch_size (int): The batch size for the dataset.
        micro_batch_num (int): The number of micro-batches.
        eod_id (int): The end-of-document token ID.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        drop_remainder (bool, optional): Whether to drop the remainder of the dataset
        if it doesn't divide evenly by the batch size. Defaults to True.
        mode (str, optional): The mode of the dataset (train, valid, or test). Defaults to "train".

    Returns:
        dataset: The created dataset.

    Raises:
        ValueError: If an invalid mode is provided.

    """
    if mode not in ["train", "valid", "test"]:
        raise ValueError(f"Invalid mode: {mode}")
    data_dir = dataset_dir + "/" + mode
    files = [str(data_file_path.absolute()) for data_file_path in Path(data_dir).glob("*.mindrecord")]
    files.sort()

    dataset = ds.MindDataset(files, columns_list=["input_ids"], shuffle=shuffle)
    per_rank_batch_size = batch_size * micro_batch_num
    global_batch_size = per_rank_batch_size * get_data_parallel_world_size()
    logger.info(f"{mode} dataset global batch size: {global_batch_size}")

    def map_func(input_ids):
        return generate_position_attention_map(
            input_ids,
            eod_id,
            get_data_parallel_rank(),
            per_rank_batch_size,
        )

    dataset = dataset.batch(global_batch_size, drop_remainder=drop_remainder)
    dataset = dataset.map(
        operations=map_func,
        input_columns=["input_ids"],
        output_columns=["input_ids", "position_ids", "attention_mask"],
    )

    return dataset


def get_dataset(dataset_config):
    """
    Get the train and valid dataset iterators.

    Args:
        dataset_config (DatasetConfig): The configuration object for the dataset.

    Returns:
        tuple: A tuple containing the train dataset iterator and the valid dataset iterator.
    """
    train_dataset_iterator = get_individual_dataset(
        dataset_dir=dataset_config.dataset_dir,
        batch_size=dataset_config.batch_size,
        micro_batch_num=dataset_config.micro_batch_num,
        eod_id=dataset_config.eod_id,
        shuffle=dataset_config.shuffle,
        drop_remainder=dataset_config.drop_remainder,
        mode="train",
    )
    valid_dataset_iterator = get_individual_dataset(
        dataset_dir=dataset_config.dataset_dir,
        batch_size=dataset_config.batch_size,
        micro_batch_num=dataset_config.micro_batch_num,
        eod_id=dataset_config.eod_id,
        shuffle=dataset_config.shuffle,
        drop_remainder=dataset_config.drop_remainder,
        mode="valid",
    )
    return train_dataset_iterator, valid_dataset_iterator
