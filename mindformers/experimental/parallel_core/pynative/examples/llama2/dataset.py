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
from pathlib2 import Path

import mindspore.dataset as ds

from mindformers.tools import logger
from mindformers.experimental.parallel_core.pynative.parallel_state import get_data_parallel_rank, get_data_parallel_world_size


def batch_and_generate_attention_mask(input_ids, eos_token_id, pad_token_id, rank, stride):
    """
    Generate position_id and attention_mask according to input_ids considering eod reset

    Inputs:
        input_ids: the input token ids
        eos_token_id: the id for <eos>
        pad_token_id: the id for <pad>
        stride: the slice value for each rank
    returns:
        input_ids: the input token ids
        attention_mask: the attention mask considering eod reset
        labels: the labels for the input token ids
        loss_mask: the loss mask for the input token ids
    """
    seq_length = input_ids.shape[1] - 1
    # Initialize position_ids and attention_mask
    batch_input_ids = input_ids[rank * stride : (rank + 1) * stride]
    batch_attention_mask = np.ones((stride, seq_length, seq_length), dtype=np.uint8)
    tril_dev = 1 - np.tril(np.ones(shape=(seq_length, seq_length), dtype=np.uint8))
    non_attn_mask_value = 1

    # Loop through batches
    for bs_i, _ in enumerate(range(len(batch_input_ids))):
        # Get normal position_ids and attention_mask
        local_ids = batch_input_ids[bs_i]
        batch_attention_mask[bs_i] = tril_dev
        position_ids = np.arange(seq_length, dtype=np.int32)

        # packing
        # Find eos
        eod_index = position_ids[local_ids[:-1] == eos_token_id].astype(np.int32)
        for i in range(eod_index.size):
            # Reset attention_mask considering eos
            index = eod_index[i]
            batch_attention_mask[bs_i, (index + 1) :, : (index + 1)] = non_attn_mask_value

    # [B,S,S] -> [B,N,S,S]
    batch_attention_mask = np.expand_dims(batch_attention_mask, axis=1)

    batch_labels = batch_input_ids[:, 1:]
    batch_input_ids = batch_input_ids[:, :-1]
    loss_mask = np.not_equal(batch_input_ids, pad_token_id)
    return batch_input_ids, batch_attention_mask, batch_labels, loss_mask


# training config
def get_individual_dataset(
        dataset_dir,
        batch_size,
        micro_batch_num,
        eos_token_id,
        pad_token_id,
        shuffle=False,
        drop_remainder=True,
        mode="train",
        heterogeneous_pipeline=False,
):
    """
    Creates and returns a dataset for a specific mode (train, valid, or test).

    Args:
        dataset_dir (str): The directory path where the dataset is located.
        batch_size (int): The batch size for the dataset.
        micro_batch_num (int): The number of micro-batches.
        eos_token_id (int): The id for <eos>.
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
    if heterogeneous_pipeline:
        global_batch_size = per_rank_batch_size
    else:
        global_batch_size = per_rank_batch_size * get_data_parallel_world_size()
    logger.info(f"{mode} dataset global batch size: {global_batch_size}")

    def map_func(input_ids):
        if heterogeneous_pipeline:
            batch_input_ids, batch_attention_mask, batch_labels, loss_mask = batch_and_generate_attention_mask(
                input_ids, eos_token_id, pad_token_id, 0, per_rank_batch_size
            )
        else:
            batch_input_ids, batch_attention_mask, batch_labels, loss_mask = batch_and_generate_attention_mask(
                input_ids, eos_token_id, pad_token_id, get_data_parallel_rank(), per_rank_batch_size
            )
        return batch_input_ids, batch_attention_mask, batch_labels, loss_mask

    dataset = dataset.batch(global_batch_size, drop_remainder=drop_remainder)
    dataset = dataset.map(
        operations=map_func,
        input_columns=["input_ids"],
        output_columns=["input_ids", "attention_mask", "labels", "loss_mask"],
    )
    return dataset


def get_dataset(dataset_config, parallel_config):
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
        eos_token_id=dataset_config.eos_token_id,
        shuffle=dataset_config.shuffle,
        drop_remainder=dataset_config.drop_remainder,
        pad_token_id=dataset_config.pad_token_id,
        mode="train",
        heterogeneous_pipeline=parallel_config.heterogeneous_pipeline
    )
    valid_dataset_iterator = get_individual_dataset(
        dataset_dir=dataset_config.dataset_dir,
        batch_size=dataset_config.batch_size,
        micro_batch_num=dataset_config.micro_batch_num,
        eos_token_id=dataset_config.eos_token_id,
        pad_token_id=dataset_config.pad_token_id,
        shuffle=dataset_config.shuffle,
        drop_remainder=dataset_config.drop_remainder,
        mode="valid",
        heterogeneous_pipeline=parallel_config.heterogeneous_pipeline
    )
    return train_dataset_iterator, valid_dataset_iterator
