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
import threading
import csv
from functools import partial
from typing import Union, Optional, Callable, List
import numpy as np

import mindspore.common.dtype as mstype
from mindspore.dataset.transforms.transforms import TypeCast
from mindspore.communication import get_rank

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
    elif isinstance(pad_token_id, int):
        pad_token_id = [pad_token_id] * len(cols)
    elif not isinstance(pad_token_id, list):
        raise ValueError("pad_token_id should be list or int.")

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
        per_batch_map_func = partial(
            asl_batch_wrapper,
            micro_batch_num=config.micro_batch_num
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
        token_monitor (bool, optional):
            Whether to enable token monitor function.  Default: ``False``.
        token_monitor_config (dict, optional):
            Config for token monitor function, When set to None, use deault value. Default: ``None``.

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
                token_monitor: bool = False,
                token_monitor_config: Optional[dict] = None,
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

            def map_func(inputs_ids):
                """Mapping function for slicing input_ids."""
                return get_input_data_batch_slice_map(inputs_ids,
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
            if dataset_config.output_columns:
                dataset = dataset.project(columns=dataset_config.output_columns)

        if dataset_config.token_monitor:
            kwargs = {}
            # Check if token_monitor_config exists and is a dict
            if isinstance(dataset_config.token_monitor_config, dict):
                kwargs = dataset_config.token_monitor_config.copy()
                if 'max_token_id' in kwargs and kwargs['max_token_id'] == "inf":
                    kwargs['max_token_id'] = np.inf
            else:
                kwargs = {}

            logger.info("token_monitor is TRUE. Saving token counts to output/token_counts_output_csv")
            dataset = get_dataset_map(dataset,
                                      operations=[cls.perform_token_counting(**kwargs)])

        if dataset_config.get('repeat', 1) > 1:
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
            if not dataset_files:
                raise FileNotFoundError(f"No dataset file is found. Please check whether the path "
                                        f"`{data_dir}` indicated by dataloader.dataset_dir is correct.")
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

    @staticmethod
    def perform_token_counting(top_n=10, min_token_id=0, max_token_id=np.inf,
                               save_path="output/token_counts_output_csv/"):
        """count tokenid and save them in csv"""
        token_counter = TokenCounter(top_n, min_token_id, max_token_id, save_path)
        return token_counter.count_tokens


class TokenCounter:
    """
    TokenCounter is a utility class for counting token occurrences during training steps,
    batching the counts, and saving them in a CSV format for later analysis.

    Args:
        top_n (int): Number of top tokens to record based on occurrence count.
        min_token_id (int): Minimum token ID to consider for counting.
        max_token_id (int): Maximum token ID to consider for counting.
        save_path (str): Directory path where token count files are saved.

    Raises:
        ValueError: If `top_n` is not a positive integer.
    """
    def __init__(self, top_n=10, min_token_id=0, max_token_id=np.inf, save_path="output/token_counts_output_csv/"):
        self.top_n = top_n
        if not isinstance(self.top_n, int) or self.top_n <= 0:
            raise ValueError("top_n must be a positive integer.")
        self.min_token_id = min_token_id
        self.max_token_id = max_token_id
        self.step_num = 1
        self.lock = threading.Lock()
        self.saved_directory = save_path
        self.token_count_pairs_header_written = False  # Flag to manage header writing
        self.initialized = False  # Flag to check if initial setup was done

    def initialize_file(self):
        """This method initializes the file for writing by clearing any existing content"""
        rank_id = get_rank()
        os.makedirs(self.saved_directory, exist_ok=True)
        filename = os.path.join(self.saved_directory, f"rank_{rank_id}_token_counts.csv")

        # Clear existing file content
        with open(filename, 'w', newline='') as csvfile:
            _ = csv.writer(csvfile)

        self.initialized = True
        self.token_count_pairs_header_written = False

    def count_tokens(self, input_ids):
        """count tokens and save in csv file"""
        if not self.initialized:
            self.initialize_file()

        tokens = input_ids.flatten()
        unique_tokens, counts = np.unique(tokens, return_counts=True)
        actual_min = np.min(unique_tokens)
        actual_max = np.max(unique_tokens)

        with self.lock:
            if actual_min < self.min_token_id:
                logger.warning("Step %d: Token ID range warning: Min token ID (%d) is below %d.",
                               self.step_num, actual_min, self.min_token_id)

            if actual_max > self.max_token_id:
                logger.warning("Step %d: Token ID range warning: Max token ID (%d) is above %d.",
                               self.step_num, actual_max, self.max_token_id)

            token_count_pairs = np.array(list(zip(unique_tokens, counts)),
                                         dtype=[('token_id', 'int32'), ('count', 'uint16')])

            if self.top_n:
                token_count_pairs = np.sort(token_count_pairs, order='count')[::-1][:self.top_n]

            rank_id = get_rank()
            filename = os.path.join(self.saved_directory, f"rank_{rank_id}_token_counts.csv")

            with open(filename, mode='a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)

                if not self.token_count_pairs_header_written:
                    header = ['step_num', 'min_id', 'max_id']
                    for i in range(len(token_count_pairs)):
                        header.append(f'token_id_{i+1}')
                        header.append(f'count_{i+1}')
                    csv_writer.writerow(header)
                    self.token_count_pairs_header_written = True

                row = [self.step_num, actual_min, actual_max]
                for token_id, count in token_count_pairs:
                    row.append(token_id)
                    row.append(count)
                csv_writer.writerow(row)

            logger.debug(f"RANK {rank_id}: Appended token counts to {filename} with {token_count_pairs}")
            self.step_num += 1

        return input_ids
