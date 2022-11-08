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
Data operations, will be used in run_pretrain.py
"""
import os
import math
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms as C


def create_language_model_dataset(config):
    """create dataset like language model task"""
    device_num = config.dataset_device_num
    rank = config.dataset_rank
    batch_size = config.dataset_batch_size
    data_dir = config.dataset_path
    do_shuffle = config.dataset_do_shuffle
    repeat_count = config.repeat_count
    type_cast_op = C.TypeCast(mstype.int32)
    data_set = ds.MindDataset([data_dir],
                              columns_list=["input_ids", "input_mask", "label_ids"],
                              shuffle=do_shuffle, num_shards=device_num, shard_id=rank)

    data_set = data_set.map(operations=type_cast_op, input_columns="input_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_mask")
    data_set = data_set.map(operations=type_cast_op, input_columns="label_ids")
    data_set = data_set.batch(batch_size, drop_remainder=True)
    data_set = data_set.repeat(repeat_count)

    print("dataset size: {}".format(data_set.get_dataset_size()))
    print("repeat count: {}".format(data_set.get_repeat_count()))
    print("output shape: {}".format(data_set.output_shapes()))
    print("output type: {}".format(data_set.output_types()))
    print("============== create dataset successful ===============")

    return data_set

def create_classification_dataset(config):
    """create finetune or evaluation dataset"""
    dataset_format = config.dataset_format
    device_num = config.dataset_device_num
    rank = config.dataset_rank
    batch_size = config.dataset_batch_size
    data_dir = config.dataset_path
    do_shuffle = config.dataset_do_shuffle
    schema_dir = config.dataset_schema_dir
    assessment_method = config.assessment_method.lower()

    type_cast_op = C.TypeCast(mstype.int32)
    if dataset_format == "mindrecord":
        data_set = ds.MindDataset([data_dir],
                                  columns_list=["input_ids", "input_mask", "segment_ids", "label_ids"],
                                  shuffle=do_shuffle, num_shards=device_num, shard_id=rank)
    elif dataset_format == "tfrecord":

        data_set = ds.TFRecordDataset(data_dir, schema_dir if schema_dir != "" else None,
                                      columns_list=["input_ids", "input_mask", "segment_ids", "label_ids"],
                                      shuffle=do_shuffle, num_shards=device_num, shard_id=rank, shard_equal_rows=True)
    else:
        raise NotImplementedError("Only supported dataset_format for tfrecord or mindrecord.")
    if assessment_method == "Spearman_correlation":
        type_cast_op_float = C.TypeCast(mstype.float32)
        data_set = data_set.map(operations=type_cast_op_float, input_columns="label_ids")
    else:
        data_set = data_set.map(operations=type_cast_op, input_columns="label_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="segment_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_mask")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_ids")
    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)
    return data_set


def generator_squad(data_features):
    for feature in data_features:
        yield (feature.input_ids, feature.input_mask, feature.segment_ids, feature.unique_id)


def create_squad_dataset(config):
    """create finetune or evaluation dataset"""
    dataset_format = config.dataset_format
    device_num = config.dataset_device_num
    rank = config.dataset_rank
    batch_size = config.dataset_batch_size
    data_dir = config.dataset_path
    do_shuffle = config.dataset_do_shuffle
    schema_dir = config.dataset_schema_dir
    is_training = config.is_training

    type_cast_op = C.TypeCast(mstype.int32)
    if is_training:
        if dataset_format == "mindrecord":
            data_set = ds.MindDataset([data_dir],
                                      columns_list=["input_ids", "input_mask", "segment_ids", "start_positions",
                                                    "end_positions", "unique_ids", "is_impossible"],
                                      shuffle=do_shuffle, num_shards=device_num, shard_id=rank)
            data_set = data_set.map(operations=type_cast_op, input_columns="start_positions")
            data_set = data_set.map(operations=type_cast_op, input_columns="end_positions")
        elif dataset_format == "tfrecord":
            data_set = ds.TFRecordDataset([data_dir], schema_dir if schema_dir != "" else None,
                                          columns_list=["input_ids", "input_mask", "segment_ids", "start_positions",
                                                        "end_positions", "unique_ids", "is_impossible"],
                                          shuffle=do_shuffle, num_shards=device_num, shard_id=rank)
    else:
        data_set = ds.GeneratorDataset(generator_squad(data_dir), shuffle=do_shuffle,
                                       column_names=["input_ids", "input_mask", "segment_ids", "unique_ids"])
    data_set = data_set.map(operations=type_cast_op, input_columns="segment_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_mask")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="unique_ids")
    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)
    return data_set





def create_eval_dataset(batchsize=32, device_num=1, rank=0, data_dir=None, schema_dir=None):
    """create evaluation dataset"""
    data_files = []
    if os.path.isdir(data_dir):
        files = os.listdir(data_dir)
        for file_name in files:
            if "tfrecord" in file_name:
                data_files.append(os.path.join(data_dir, file_name))
    else:
        data_files.append(data_dir)
    data_set = ds.TFRecordDataset(data_files, schema_dir if schema_dir != "" else None,
                                  columns_list=["input_ids", "input_mask", "segment_ids", "next_sentence_labels",
                                                "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"],
                                  shard_equal_rows=True)
    ori_dataset_size = data_set.get_dataset_size()
    print("origin eval size: ", ori_dataset_size)
    dtypes = data_set.output_types()
    shapes = data_set.output_shapes()
    output_batches = math.ceil(ori_dataset_size / device_num / batchsize)
    padded_num = output_batches * device_num * batchsize - ori_dataset_size
    print("padded num: ", padded_num)
    if padded_num > 0:
        item = {"input_ids": np.zeros(shapes[0], dtypes[0]),
                "input_mask": np.zeros(shapes[1], dtypes[1]),
                "segment_ids": np.zeros(shapes[2], dtypes[2]),
                "next_sentence_labels": np.zeros(shapes[3], dtypes[3]),
                "masked_lm_positions": np.zeros(shapes[4], dtypes[4]),
                "masked_lm_ids": np.zeros(shapes[5], dtypes[5]),
                "masked_lm_weights": np.zeros(shapes[6], dtypes[6])}
        padded_samples = [item for x in range(padded_num)]
        padded_ds = ds.PaddedDataset(padded_samples)
        eval_ds = data_set + padded_ds
        sampler = ds.DistributedSampler(num_shards=device_num, shard_id=rank, shuffle=False)
        eval_ds.use_sampler(sampler)
    else:
        eval_ds = ds.TFRecordDataset(data_files, schema_dir if schema_dir != "" else None,
                                     columns_list=["input_ids", "input_mask", "segment_ids", "next_sentence_labels",
                                                   "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"],
                                     num_shards=device_num, shard_id=rank, shard_equal_rows=True)

    type_cast_op = C.TypeCast(mstype.int32)
    eval_ds = eval_ds.map(input_columns="masked_lm_ids", operations=type_cast_op)
    eval_ds = eval_ds.map(input_columns="masked_lm_positions", operations=type_cast_op)
    eval_ds = eval_ds.map(input_columns="next_sentence_labels", operations=type_cast_op)
    eval_ds = eval_ds.map(input_columns="segment_ids", operations=type_cast_op)
    eval_ds = eval_ds.map(input_columns="input_mask", operations=type_cast_op)
    eval_ds = eval_ds.map(input_columns="input_ids", operations=type_cast_op)

    eval_ds = eval_ds.batch(batchsize, drop_remainder=True)
    print("eval data size: {}".format(eval_ds.get_dataset_size()))
    print("eval repeat count: {}".format(eval_ds.get_repeat_count()))
    return eval_ds
