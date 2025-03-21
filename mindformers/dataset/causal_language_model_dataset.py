# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Causal Image Modeling Dataset."""
import os
import copy
import re
from functools import partial
from typing import Union, Optional, Callable, List
from importlib import import_module
import numpy as np

import mindspore.common.dtype as mstype
from mindspore.dataset.transforms.transforms import TypeCast

from mindformers.tools.register.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.logger import logger
from mindformers.version_control import get_dataset_map

from .dataloader.build_dataloader import build_dataset_loader
from .base_dataset import BaseDataset

CAST_TO_INT_COLUMNS = ["input_ids", "labels"]


def _use_compressed_eod_mask(data_loader):
    if (hasattr(data_loader, 'config') and data_loader.config and
            data_loader.config.create_compressed_eod_mask):  # megatron dataset
        return True
    if (hasattr(data_loader, 'adaptor_config') and data_loader.adaptor_config and
            data_loader.adaptor_config.compress_mask):  # common dataloader
        return True
    return False


def dyn_batch_wrapper(*cols, divisor, remainder, pad_token_id=None):
    """Dynamic batch process function for padding each batch data."""
    if pad_token_id is None:
        pad_token_id = [0, -100]
    elif not isinstance(pad_token_id, list):
        raise ValueError("pad_token_id should be list.")

    columns = cols[:-1]
    outputs = []
    for col_idx, col in enumerate(columns):
        # set dynamic batch max length
        max_length = max([len(sample) for sample in col])
        if divisor and remainder:
            max_length = ((max_length - remainder - 1) // divisor + 1) * divisor + remainder
        else:
            logger.info("dynamic batch 'divisor' or 'remainder' not set.")

        # pad samples
        pad = pad_token_id[col_idx]
        cur_col = []
        for sample in col:
            sample = np.pad(
                sample, (0, max_length - len(sample)),
                mode='constant',
                constant_values=pad
            )
            cur_col.append(sample)
        outputs.append(cur_col)
    return tuple(outputs)


def asl_batch_wrapper(*cols, micro_batch_num):
    """Add offset to actual_seq_len for each batch data."""
    columns = cols[:-1]

    # actual_seq_len have to set in last column from dataset.__getitem__
    actual_seq_len = columns[-1]
    if len(actual_seq_len) == 1:
        # not process if actual_seq_len's batch size = 1
        return columns

    # add offset to each sample
    batch_size = len(columns[-1]) // micro_batch_num
    cur_seq_len = []
    for micro_idx in range(micro_batch_num):
        offset = 0
        start_seq_idx = micro_idx * batch_size
        end_seq_idx = (micro_idx + 1) * batch_size
        for seq_idx in range(start_seq_idx, end_seq_idx):
            per_seq = actual_seq_len[seq_idx] + offset
            offset = per_seq[-1]
            cur_seq_len.append(per_seq)

    # only replace last column
    columns = columns[:-1] + (cur_seq_len,)
    return columns


def dataset_batch_func(config, dataset):
    """Dataset batch process function"""
    # set per_batch_map
    per_batch_map_func = None
    use_compressed_eod_mask = _use_compressed_eod_mask(config.data_loader)

    if config.dynamic_batch and use_compressed_eod_mask:
        raise ValueError("dynamic_batch and use_compressed_eod_mask not supported simultaneously.")

    if config.dynamic_batch:
        # set dynamic batch wrapper
        per_batch_map_func = partial(
            dyn_batch_wrapper,
            divisor=config.divisor,
            remainder=config.remainder,
            pad_token_id=config.pad_token_id
        )
    elif use_compressed_eod_mask:
        context_module = import_module("mindformers.core.context.build_context")
        context_instance = context_module.Context()
        if context_instance is not None and context_instance.is_exists():
            context_instance = context_module.Context()
            # set batch actual_seq_len wrapper
            per_batch_map_func = partial(
                asl_batch_wrapper,
                micro_batch_num=context_instance.parallel_opr.parallel.micro_batch_num
            )

    # set num_parallel_workers
    if config.eod_reset:
        # this branch might be abandoned
        num_parallel_workers = None
    else:
        num_parallel_workers = config.num_parallel_workers

    dataset = dataset.batch(
        batch_size=config.batch_size,
        drop_remainder=config.drop_remainder,
        output_columns=config.input_columns,
        num_parallel_workers=num_parallel_workers,
        per_batch_map=per_batch_map_func)
    return dataset


def get_input_data_batch_slice_map(input_ids, eod_token_id, dis, rank_id: int = 0):
    """
    Generate position_id and attention_mask according to input_ids considering eod reset

    Args:
        input_ids: the input token ids
        eod_token_id: the id for <EOD>
        dis: the slice value for each rank
        rank_id: the current rank id
    Returns:
        batch_input_ids: the input token ids
        batch_position_ids: the position ids considering eod reset
        batch_attention_mask: the attention mask considering eod reset
    """
    rank = int(rank_id)
    input_ids = input_ids[rank * dis: (rank + 1) * dis]
    seq_length = input_ids.shape[1] - 1
    # Initialize position_ids and attention_mask
    batch_input_ids = input_ids
    batch_position_ids = np.ones((dis, seq_length))
    batch_attention_mask = np.ones((dis, seq_length, seq_length))

    # Loop through batches
    for bs_i, local_ids in enumerate(input_ids):
        # Get normal position_ids and attention_mask
        batch_attention_mask[bs_i] = np.tril(np.ones(shape=(seq_length, seq_length)))
        batch_position_ids[bs_i] = np.arange(seq_length)
        # Find the index of <EOS>
        eod_index = batch_position_ids[bs_i, local_ids[:-1] == eod_token_id].astype(np.int32)
        prev_index = 0
        for i in range(eod_index.size):
            # Reset position_ids and attention_mask considering <EOS>
            index = eod_index[i]
            batch_attention_mask[bs_i, (index + 1):, :(index + 1)] = 0
            batch_position_ids[bs_i, (index + 1):] -= (index + 1 - prev_index)
            prev_index = index + 1
    return batch_input_ids, batch_position_ids, batch_attention_mask


@MindFormerRegister.register(MindFormerModuleType.DATASET)
class CausalLanguageModelDataset(BaseDataset):
    """
    Causal Language Model pretrain dataset.

    The columns of generated dataset depend on the config provided by user.
    The tensor of each column will be cast to int32 type.

    Args:
        dataset_config (dict, optional):
            Config for dataset. When `dataset_config` is an empty dict or is None, all arguments below
            will build a non-empty `dataset_config`. Otherwise, they will be ignored. Default: ``None``.
        data_loader (Union[dict, Callable], optional):
            Config for data loader or a data loader object. When `data_loader` is a `dict`,
            the string "type", "dataset_dir", "dataset_files" and "shuffle" are the keys can be parsed.
            Default: ``None``.

            - type: Required. Indicates the type of dataset. The value must be string or class type.
              When the value is "MindDataset" or "TFRecordDataset",
              one of `dataset_dir` and `dataset_files` is required, where `dataset_dir` takes effect first;
              otherwise `dataset_dir` is required.

            - dataset_dir: The path or directory of dataset. When `type` is "MindDataset" or "TFRecordDataset"
              and `dataset_dir` is a directory, search for files in `mindrecord` or `tfrecord` format recursively
              in the directory.

            - dataset_files: The path of files in `mindrecord` or `tfrecord` format.
              Take effect when `type` is "MindDataset" or "TFRecordDataset", otherwise this key is ignored.
              Must be `list` or `tuple`.

            - shuffle: Optional. Whether to perform shuffle on the dataset. Must be `bool`.

        input_columns (list[str], optional):
            Column names before the map function. Default: ``None``.
        output_columns (list[str], optional):
            Column names after the map function.
            Reuired when `eod_reset` is True; otherwise ignored. Default: ``None``.
        batch_size (int, optional):
            Size of each batch. Default: ``8``.
        drop_remainder (bool, optional):
            Whether to discard the last batch when the number of data items contained in the last batch is smaller
            than batch_size. Default: ``True``.
        num_parallel_workers (int, optional):
            Specifies the number of concurrent processes or threads for map operations
            to accelerate processing. Default: ``8``.
        python_multiprocessing (bool, optional):
            Enabling the Python Multi-Process Mode to Accelerate Map Operations. Default: ``False``.
        repeat (int, optional):
            Number of times this dataset is repeated. Default: ``1``.
        seed (int, optional):
            Random seed number. Default: ``0``.
        prefetch_size (int, optional):
            Buffer queue size of each data processing operation in the pipeline. Default: ``1``.
        numa_enable (bool, optional):
            Indicates whether to use the NUMA binding function. Default: ``False``.
        eod_reset (bool, optional):
            Specifies whether to reset the EOD. Default: ``False``.
        eod_token_id (int, optional):
            Indicates the token id of the EOD. Default: ``None``, don't set the token id of the EOD manually.
        auto_tune (bool, optional):
            Indicates whether to enable automatic optimization of data processing parameters. Default: ``False``.
        autotune_per_step (int, optional):
            Specifies the interval for adjusting the configuration step of automatic data acceleration. Default: ``10``.
        filepath_prefix (str, optional):
            Path for saving optimized parameter configurations. Default: ``'./autotune'``.
        profile (bool, optional):
            Whether to enable data collection. Default: ``False``.

    Returns:
        Instance of CausalLanguageModelDataset.

    Raises:
        ValueError: If `dataset_config.batch_size` is not a multiple of device number
                    when `dataset_config.eod_reset` is True and dataset isn't imported in full.
        ValueError: If `dataset_config` doesn't contain "dataset_dir" or "dataset_files" as its key.

    Examples:
        >>> from mindspore.dataset import MindDataset
        >>> from mindformers.dataset import CausalLanguageModelDataset
        >>> # Note:
        >>> #     `"/path/to/dataset"` should be replaced with the real path of the dataset file.
        >>> #     The detailed data setting could refer to
        >>> #     https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/llama2.md
        >>> data_loader = MindDataset(dataset_files="/path/to/dataset", shuffle=True)
        >>> dataset_from_param = CausalLanguageModelDataset(data_loader=data_loader,
        ...                                                 input_columns=["input_ids", "attention_mask"])
    """

    # pylint: disable=W0613
    def __new__(cls,
                dataset_config: Optional[dict] = None,
                data_loader: Union[dict, Callable] = None,
                input_columns: List[str] = None,
                output_columns: List[str] = None,
                batch_size: int = 8,
                drop_remainder: bool = True,
                num_parallel_workers: int = 8,
                python_multiprocessing: bool = False,
                repeat: int = 1,
                seed: int = 0,
                prefetch_size: int = 1,
                numa_enable: bool = False,
                eod_reset: bool = False,
                eod_token_id: Optional[int] = None,
                auto_tune: bool = False,
                filepath_prefix: str = './autotune',
                autotune_per_step: int = 10,
                profile: bool = False,
                **kwargs):
        logger.info("Now Create Causal Language Model Dataset.")
        dataset_config = cls.check_dataset_config(dataset_config, locals())
        dataset_config = copy.deepcopy(dataset_config)
        cls.init_dataset_config(dataset_config)
        shard_id, num_shards = cls._generate_shard_info()
        dataset_config.shard_id = shard_id
        dataset_config.num_shards = num_shards

        if isinstance(dataset_config.data_loader, dict):
            if dataset_config.data_loader.type != "MindDataset" and \
                    dataset_config.data_loader.type != "TFRecordDataset":
                dataset = cls._process_raw_text_data(dataset_config)
            else:
                dataset = cls._process_mindrecord_data(dataset_config)
        else:
            dataset = dataset_config.data_loader

        type_cast_op = TypeCast(mstype.int32)
        if dataset_config.eod_reset:
            if cls._is_semi_full_batch() or cls._is_data_parallel():
                shard_id = 0
                dis = dataset_config.batch_size
            else:
                # Each card slice a small batch from the full batch
                dis = dataset_config.batch_size // num_shards
                if dataset_config.batch_size % num_shards != 0:
                    raise ValueError(
                        f"batch size {dataset_config.batch_size} should be a multiple of dataset shard number "
                        f"{num_shards}. You should change the args: per_batch_size.")

            dataset = dataset_batch_func(dataset_config, dataset)
            map_func = lambda input_ids: get_input_data_batch_slice_map(input_ids,
                                                                        eod_token_id=dataset_config.eod_token_id,
                                                                        rank_id=shard_id,
                                                                        dis=dis)
            dataset = get_dataset_map(dataset, map_func,
                                      input_columns=dataset_config.input_columns,
                                      output_columns=dataset_config.output_columns)
            dataset = dataset.project(columns=dataset_config.output_columns)

            for input_arg in dataset_config.output_columns:
                if input_arg in CAST_TO_INT_COLUMNS:
                    dataset = get_dataset_map(dataset, type_cast_op,
                                              input_columns=input_arg)
        else:
            dataset = dataset_batch_func(dataset_config, dataset)

            dataset = dataset.project(columns=dataset_config.input_columns)
            for input_arg in dataset_config.input_columns:
                if input_arg in CAST_TO_INT_COLUMNS:
                    dataset = get_dataset_map(dataset, type_cast_op,
                                              input_columns=input_arg)
        dataset = dataset.repeat(dataset_config.repeat)
        return dataset

    @classmethod
    def _process_raw_text_data(cls, dataset_config):
        """Process the text data"""
        dataset_dir = dataset_config.data_loader.pop("dataset_dir", None)
        dataset = build_dataset_loader(
            dataset_config.data_loader, default_args={'dataset_dir': dataset_dir,
                                                      'num_shards': dataset_config.num_shards,
                                                      'column_names': dataset_config.input_columns,
                                                      'shard_id': dataset_config.shard_id})
        return dataset

    @classmethod
    def _process_mindrecord_data(cls, dataset_config):
        """Process the mindrecord data"""
        dataset_files = []
        mind_compile = re.compile(r"mindrecord\d*$")
        if dataset_config.data_loader.dataset_dir:
            data_dir = dataset_config.data_loader.pop("dataset_dir")
            if os.path.isdir(data_dir):
                for r, _, f in os.walk(data_dir):
                    for file in f:
                        if re.findall(mind_compile, file) or file.endswith(".tfrecord"):
                            dataset_files.append(os.path.join(r, file))
                dataset_files.sort()
            else:
                if re.findall(mind_compile, data_dir) or data_dir.endswith(".tfrecord"):
                    dataset_files = data_dir
        elif dataset_config.data_loader.dataset_files:
            dataset_files = dataset_config.data_loader.dataset_files
            if isinstance(dataset_files, (list, tuple)):
                dataset_files = list(dataset_files)
        else:
            raise ValueError(f"data_loader must contain dataset_dir or dataset_files,"
                             f"but get {dataset_config.data_loader}.")

        if not cls._is_full_batch():
            dataset_config = cls._reset_num_samples(dataset_config)

        dataset = build_dataset_loader(
            dataset_config.data_loader, default_args={'dataset_files': dataset_files,
                                                      'num_shards': dataset_config.num_shards,
                                                      'shard_id': dataset_config.shard_id,
                                                      'columns_list': dataset_config.input_columns})
        return dataset

    @classmethod
    def _reset_num_samples(cls, dataset_config):
        """reset num_samples for full_batch=False"""
        num_samples = dataset_config.data_loader.get('num_samples')
        if num_samples is None:
            return dataset_config

        # dataset_config.device_num is equal to dp
        cur_num_samples = num_samples // dataset_config.num_shards
        logger.info(f"If set full_batch=False and num_samples, "
                    f"num_samples will reset from {num_samples} to {cur_num_samples}.")
        dataset_config.data_loader.num_samples = cur_num_samples
        return dataset_config
