# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""Data operations, will be used in pretrain_t5.py."""
import os
import numpy as np

import mindspore as ms
import mindspore.dataset as de

de.config.set_seed(1)


def generate_memory_mask(source_mask, target_mask):
    """ generate the memory mask from the source mask and the target_mask
    :param source_mask: src_seq_length x src_seq_length
    :param target_mask: tgt_seq_length x tgt_seq_length
    :return: a float16 numpy array with shape tgt_seq_length x src_seq_length
    """
    s = source_mask[0, :]
    t = target_mask[:, 0]
    # print(s.shape, t.shape)
    memory_mask = np.ones((np.shape(target_mask)[0], np.shape(source_mask)[0]))
    memory_mask = memory_mask * np.expand_dims(s, 0)
    memory_mask = memory_mask * np.expand_dims(t, 1)
    return source_mask, target_mask, memory_mask.astype(np.float16)


def create_buckets_dataset(batch_size, data_path,
                           device_num=1, rank=0, drop=True,
                           bucket_boundaries=None):
    """Create the dataset"""

    def batch_per_bucket(bucket_len, dataset_path):
        dataset_path = dataset_path + "_" + str(bucket_len) + "_00"
        ds = de.MindDataset(dataset_path,
                            columns_list=["source_eos_ids", "source_eos_mask",
                                          "target_sos_ids", "target_sos_mask",
                                          "target_eos_ids", "target_eos_mask"],
                            shuffle=True, num_shards=device_num, shard_id=rank)
        type_cast_op = de.transforms.c_transforms.TypeCast(ms.int32)
        ds = ds.map(operations=type_cast_op, input_columns="source_eos_ids")
        ds = ds.map(operations=type_cast_op, input_columns="source_eos_mask")
        ds = ds.map(operations=type_cast_op, input_columns="target_sos_ids")
        ds = ds.map(operations=type_cast_op, input_columns="target_sos_mask")
        ds = ds.map(operations=type_cast_op, input_columns="target_eos_ids")
        ds = ds.map(operations=type_cast_op, input_columns="target_eos_mask")

        # apply batch operations
        ds = ds.batch(batch_size, drop_remainder=drop)
        return ds

    for i, _ in enumerate(bucket_boundaries):
        bucket_len = bucket_boundaries[i]
        ds_per = batch_per_bucket(bucket_len, data_path)
        if i == 0:
            ds = ds_per
        else:
            ds = ds + ds_per
    ds = ds.shuffle(ds.get_dataset_size())
    ds.channel_name = 'transformer'
    return ds


def create_t5_dataset(config):
    """Create the dataset"""
    device_num = config.dataset_device_num
    rank = config.dataset_rank
    batch_size = config.dataset_batch_size
    data_path = config.dataset_path
    drop = config.dataset_drop_remainder

    data = [os.path.join(data_path, name) for name in os.listdir(data_path) if name.endswith("mindrecord")]
    ds = de.MindDataset(data,
                        columns_list=["input_ids", "attention_mask", "labels"],
                        shuffle=True, num_shards=device_num, shard_id=rank)

    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=drop)
    ds = ds.shuffle(ds.get_dataset_size())

    print(f"Dataset column name is {ds.get_col_names()}\n"
          f"The shape for each column is {ds.output_shapes()}")
    return ds
