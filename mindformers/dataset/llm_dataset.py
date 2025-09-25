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
# pylint: disable=too-many-lines
"""LLM Modeling Dataset."""
import os
import copy
import re
import threading
import csv
from functools import partial
from typing import Union, Optional, Callable, List, Tuple, Any

import numpy as np
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore.dataset.transforms.transforms import TypeCast
from mindspore.communication import get_rank
from mindspore import dataset as dataset_set

from mindformers.tools.utils import get_real_rank, get_real_group_size
from mindformers.tools.register.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.logger import logger
from mindformers.version_control import get_dataset_map
from mindformers.tools.utils import set_safe_mode_for_file_or_dir
from .dataloader.build_dataloader import build_dataset_loader

CAST_TO_INT_COLUMNS = ["input_ids", "labels"]


def _check_compressed_eod_mask(data_loader: Any) -> bool:
    """
    Check if compressed EOD mask is enabled in data loader.

    Args:
        data_loader (Any): Data loader object to check.

    Returns:
        bool: True if compressed EOD mask is enabled, False otherwise.
    """
    if (hasattr(data_loader, 'config') and data_loader.config and
            data_loader.config.create_compressed_eod_mask):  # megatron dataset
        return True
    if (hasattr(data_loader, 'adaptor_config') and data_loader.adaptor_config and
            data_loader.adaptor_config.compress_mask):  # common dataloader
        return True
    return False


def dyn_batch_wrapper(*cols: Tuple[np.ndarray, ...],
                      divisor: int,
                      remainder: int,
                      pad_token_id: Optional[Union[int, List[int]]] = None) -> Tuple[List[np.ndarray], ...]:
    """
    Dynamic batch process function for padding each batch data to uniform length.

    This function applies dynamic padding to ensure all samples in a batch have the same
    sequence length, which is calculated based on the maximum length in the batch
    adjusted by divisor and remainder parameters.

    Args:
        *cols (Tuple[np.ndarray, ...]): Column data arrays to process, typically containing
            input_ids, labels, and other features.
        divisor (int): Divisor for dynamic batch length calculation, used to round up
            the maximum sequence length to the nearest multiple.
        remainder (int): Remainder for dynamic batch length calculation, added after
            rounding with divisor.
        pad_token_id (Optional[Union[int, List[int]]]): Padding token ID(s) for each column.
            If int, same padding value is used for all columns. If List[int], each column
            uses its corresponding padding value. Defaults to [0, -100] for input and label columns.

    Returns:
        Tuple[List[np.ndarray], ...]: List of padded column data arrays, where each array
            contains padded samples of uniform length.

    Raises:
        ValueError: If pad_token_id is provided but is neither int nor list.
    """
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
        max_length = max(len(sample) for sample in col)
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


def asl_batch_wrapper(*cols: Tuple[np.ndarray, ...],
                      micro_batch_num: int) -> Tuple[np.ndarray, ...]:
    """
    Add offset to actual sequence length for gradient accumulation in micro-batches.

    This function processes the actual sequence length data to support gradient accumulation
    across multiple micro-batches by adding appropriate offsets to maintain correct
    positional information.

    Args:
        *cols (Tuple[np.ndarray, ...]): Column data arrays to process, where the last
            column contains the actual sequence lengths.
        micro_batch_num (int): Number of micro batches for gradient accumulation.

    Returns:
        Tuple[np.ndarray, ...]: Processed column data with updated sequence lengths
            that include appropriate offsets for micro-batch processing.
    """
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


# pylint: disable=too-many-arguments,too-many-positional-arguments
def dataset_batch_func(dataset: Any,
                       data_batch_size: int = 1,
                       drop_remainder: bool = True,
                       output_columns: Optional[List[str]] = None,
                       eod_reset: bool = False,
                       dynamic_batch: bool = False,
                       divisor: Optional[int] = None,
                       remainder: Optional[int] = None,
                       pad_token_id: Optional[int] = None,
                       use_compressed_eod_mask: bool = False,
                       micro_batch_num: int = 1,
                       num_parallel_workers: Optional[int] = None) -> Any:
    """
    Dataset batch process function.

    Args:
        dataset (Any): Input dataset to batch.
        data_batch_size (int): Batch size for the dataset. Defaults to 1.
        drop_remainder (bool): Whether to drop remainder samples. Defaults to True.
        output_columns (Optional[List[str]]): Output column names. Defaults to None.
        eod_reset (bool): Whether to reset EOD (End Of Document). Defaults to False.
        dynamic_batch (bool): Whether to use dynamic batching. Defaults to False.
        divisor (Optional[int]): Divisor for dynamic batching. Defaults to None.
        remainder (Optional[int]): Remainder for dynamic batching. Defaults to None.
        pad_token_id (Optional[int]): Padding token ID. Defaults to None.
        use_compressed_eod_mask (bool): Whether to use compressed EOD mask. Defaults to False.
        micro_batch_num (int): Number of micro batches for gradient accumulation. Defaults to 1.
        num_parallel_workers (Optional[int]): Number of parallel workers. Defaults to None.

    Returns:
        Any: Batched dataset.
    """
    # set per_batch_map
    per_batch_map_func = None
    if dynamic_batch and use_compressed_eod_mask:
        raise ValueError("dynamic_batch and use_compressed_eod_mask not supported simultaneously.")

    if dynamic_batch:
        # set dynamic batch wrapper
        per_batch_map_func = partial(
            dyn_batch_wrapper,
            divisor=divisor,
            remainder=remainder,
            pad_token_id=pad_token_id
        )
    elif use_compressed_eod_mask:
        per_batch_map_func = partial(
            asl_batch_wrapper,
            micro_batch_num=micro_batch_num
        )

    # set num_parallel_workers
    if eod_reset:
        # this branch might be abandoned
        num_parallel_workers = None

    dataset = dataset.batch(
        batch_size=data_batch_size,
        drop_remainder=drop_remainder,
        output_columns=output_columns,
        num_parallel_workers=num_parallel_workers,
        per_batch_map=per_batch_map_func)
    return dataset


def get_input_data_batch_slice_map(input_ids: np.ndarray,
                                   eod_token_id: int,
                                   slice_size: int,
                                   rank_id: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate position_id and attention_mask according to input_ids considering eod reset.

    Args:
        input_ids (np.ndarray): The input token ids.
        eod_token_id (int): The id for <EOD> (End Of Document).
        slice_size (int): The size of data slice for each rank.
        rank_id (int): The current rank id. Defaults to 0.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - input_ids: the sliced input token ids
            - position_ids: the position ids considering eod reset
            - attention_mask: the attention mask considering eod reset
    """
    rank = int(rank_id)
    input_ids = input_ids[rank * slice_size: (rank + 1) * slice_size]
    seq_length = input_ids.shape[1] - 1
    # Initialize position_ids and attention_mask
    batch_input_ids = input_ids
    batch_position_ids = np.ones((slice_size, seq_length))
    batch_attention_mask = np.ones((slice_size, seq_length, seq_length))

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
class LLMDataset:
    """Causal Language Model train dataset.
    
    This class provides functionality for loading and processing large language model
    datasets with support for various data loaders and parallel processing configurations.
    
    Args:
        dataset_config (dict, optional): Dataset configuration dictionary containing
            data_loader and other dataset-specific settings. Defaults to None.

    Raises:
        ValueError: If dataset_config is None or invalid.
    """

    def __init__(self, dataset_config: dict = None) -> None:
        if dataset_config is None:
            raise ValueError("dataset_config cannot be None")
        self.dataset_config = copy.deepcopy(dataset_config)
        self.data_loader_config = self.dataset_config.get("data_loader")
        if not isinstance(self.data_loader_config, dict):
            raise ValueError(f"data_loader_config must be a dict, but get {type(self.data_loader_config)}")
        data_loader_type = self.data_loader_config.get("type")
        if data_loader_type is None:
            raise ValueError("data_loader_config must contain 'type' key")
        self.data_loader_type = data_loader_type

        ds_broadcast_level = ms.context.get_context("dataset_broadcast_opt_level")
        if data_loader_type in ['HFDataLoader', 'CommonDataLoader']:
            if self.data_loader_config.get('use_broadcast_data', True) and ds_broadcast_level < 3:
                raise ValueError(
                    "If you are using `HFDataLoader` or `CommonDataLoader` and enable `use_broadcast_data`, "
                    "please set `dataset_broadcast_opt_level: 3` in the `parallel_speed_up.json` file."
                )
            handler = []
            if self.data_loader_config.get("handler") is not None:
                handler = [sub_handler.get('type') for sub_handler in self.data_loader_config.get("handler")]

            if 'PackingHandler' in handler:
                self.data_loader_config.create_attention_mask = True
                logger.info("create_attention_mask is enabled by default when PackingHandler is used")

        if self.data_loader_type == "BlendedMegatronDatasetDataLoader":
            if ds_broadcast_level < 3:
                raise ValueError("If using `BlendedMegatronDatasetDataLoader`, please set "
                                 "`dataset_broadcast_opt_level: 3` in the `parallel_speed_up.json` file.")

        logger.info(f"Current DataLoader is {data_loader_type}")

    def set_megatron_dataset_config_seed(self, dataset_seed: int = 1234) -> None:
        """
        Set the configuration seed for Megatron dataset.

        Args:
            dataset_seed (int): Seed value for the dataset. Defaults to 1234.
        """
        if self.data_loader_type == "BlendedMegatronDatasetDataLoader":
            self.data_loader_config.config.seed = dataset_seed

    def set_ms_dataset_config(self, dataset_seed: int = 1234, prefetch_size: int = 1,
                              numa_enable: bool = False) -> None:
        """
        Set MindSpore dataset configuration.

        Args:
            dataset_seed (int): Dataset seed value. Defaults to 1234.
            prefetch_size (int): Dataset queue prefetch size. Defaults to 1.
            numa_enable (bool): Whether to enable NUMA. Defaults to False.
        """
        dataset_set.set_seed(dataset_seed)
        dataset_set.config.set_prefetch_size(prefetch_size)
        dataset_set.config.set_numa_enable(numa_enable)
        self.set_megatron_dataset_config_seed(dataset_seed)
        logger.info(f"Dataset {self.data_loader_type} seed has been set to {dataset_seed}")

    def get_default_input_columns(self, create_attention_mask: bool = False,
                                  create_compressed_eod_mask: bool = False) -> List[str]:
        """
        Reset column names for the dataset based on configuration and flags.

        Args:
            create_attention_mask (bool):
                Whether to include attention_mask column. Defaults to False.
            create_compressed_eod_mask (bool):
                Whether to include actual_seq_len column for compressed EOD mask. Defaults to False.

        Returns:
            List[str]: List of column names for the dataset.
        """
        if create_compressed_eod_mask:
            column_names = ['input_ids', 'labels', 'loss_mask', 'position_ids', 'actual_seq_len']
        elif create_attention_mask:
            column_names = ['input_ids', 'labels', 'loss_mask', 'position_ids', 'attention_mask']
        elif self.data_loader_type == 'BlendedMegatronDatasetDataLoader':
            column_names = ['input_ids', 'labels', 'loss_mask', 'position_ids']
        else:
            column_names = self.dataset_config.get("input_columns")
        logger.info(
            f"Current create_compressed_eod_mask: {create_compressed_eod_mask},"
            f"create_attention_mask: {create_attention_mask},"
            f"dataset input column_names: {column_names}")
        return column_names

    def generate_shard_info(self, data_parallel_size: int = 1) -> Tuple[Optional[int], Optional[int]]:
        """
        Generate shard info for dataset based on parallel configuration.

        Args:
            data_parallel_size (int): Data parallel size. Defaults to 1.

        Returns:
            Tuple[Optional[int], Optional[int]]: Shard ID and number of shards.
        """
        shard_id = get_real_rank()
        num_shards = get_real_group_size()
        dp = data_parallel_size

        if self._is_semi_full_batch():
            shard_id = None
            num_shards = None
            logger.debug("Using semi full batch mode, shard_id and num_shards set to None")
        elif self._is_semi() and not self._is_full_batch():
            pp = ms.context.get_auto_parallel_context("pipeline_stages")
            mp = num_shards // pp // dp
            shard_id = shard_id % (num_shards // pp) // mp
            num_shards = dp
            logger.debug(f"Using semi parallel mode, shard_id={shard_id}, num_shards={num_shards}")

        logger.info(f"Generated dataset strategy: shard_id: {shard_id}, num_shards: {num_shards}")
        return shard_id, num_shards

    def create_data_loader(self, column_names: list = None, num_shards: int = None, shard_id: int = None) -> Any:
        """
        Create data loader based on dataset configuration.

        Args:
            column_names (list, optional): Column names for the dataset. Defaults to None.
            num_shards (int, optional): Number of dataset shards. Defaults to None.
            shard_id (int, optional): Shard ID for current process. Defaults to None.

        Returns:
            Any: Data loader instance.

        Raises:
            ValueError: If column_names is None.
        """
        if column_names is None:
            raise ValueError("column_names cannot be None")

        if self.data_loader_type != "MindDataset":
            logger.info(f"Creating custom data loader of type {self.data_loader_type}")
            data_loader = self._process_custom_llm_data(column_names, num_shards, shard_id)
        else:
            logger.info("Creating MindDataset data loader")
            data_loader = self._process_mindrecord_data(column_names, num_shards, shard_id)

        logger.info("Data loader created successfully")
        return data_loader

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    def _create_mindrecord_dataset(self, data_loader: Any, data_batch_size: int, drop_remainder: bool,
                                   input_columns: list, output_columns: list, num_shards: int, shard_id: int,
                                   eod_reset: bool, eod_token_id: int, pad_token_id: int,
                                   num_parallel_workers: int, use_compressed_eod_mask: bool) -> Any:
        """
        Create dataset for MindDataset data loader.

        Args:
            data_loader (Any): Data loader instance.
            data_batch_size (int): Batch size for the dataset.
            drop_remainder (bool): Whether to drop remainder samples.
            input_columns (list): Input column names.
            output_columns (list): Output column names.
            num_shards (int): Number of dataset shards.
            shard_id (int): Shard ID for current process.
            eod_reset (bool): Whether to reset EOD.
            eod_token_id (int): EOD token ID.
            pad_token_id (int): Padding token ID.
            num_parallel_workers (int): Number of parallel workers.
            use_compressed_eod_mask (bool): Whether to use compressed EOD mask.

        Returns:
            Any: Configured dataset instance.
        """
        type_cast_op = TypeCast(mstype.int32)

        if self._is_semi_full_batch() or self._is_data_parallel():
            shard_id = 0
            dis = data_batch_size
            logger.debug("Using semi full batch or data parallel mode, shard_id set to 0")
        else:
            # Each card slice a small batch from the full batch
            dis = data_batch_size // num_shards
            if data_batch_size % num_shards != 0:
                raise ValueError(
                    f"Batch size {data_batch_size} should be a multiple of dataset shard number "
                    f"{num_shards}. Please adjust the per_batch_size parameter.")

        dataset = dataset_batch_func(data_loader,
                                     data_batch_size=data_batch_size, drop_remainder=drop_remainder,
                                     output_columns=input_columns, eod_reset=eod_reset,
                                     dynamic_batch=False, pad_token_id=pad_token_id,
                                     use_compressed_eod_mask=use_compressed_eod_mask,
                                     num_parallel_workers=num_parallel_workers)

        def map_func(inputs_ids):
            """Mapping function for slicing input_ids."""
            if eod_token_id is None:
                raise ValueError("eod_token_id cannot be None when using MindDataset with EOD reset")
            return get_input_data_batch_slice_map(inputs_ids,
                                                  eod_token_id=eod_token_id,
                                                  rank_id=shard_id,
                                                  slice_size=dis)

        dataset = get_dataset_map(dataset, map_func,
                                  input_columns=input_columns,
                                  output_columns=output_columns)
        dataset = dataset.project(columns=output_columns)

        for input_arg in output_columns:
            if input_arg in CAST_TO_INT_COLUMNS:
                dataset = get_dataset_map(dataset, type_cast_op,
                                          input_columns=input_arg)
        return dataset

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    def _create_custom_dataset(self, data_loader: Any, data_batch_size: int, drop_remainder: bool,
                               input_columns: list, output_columns: list, micro_batch_num: int,
                               dynamic_batch: bool, divisor: int, remainder: int, pad_token_id: int,
                               num_parallel_workers: int, use_compressed_eod_mask: bool) -> Any:
        """
        Create dataset for custom data loaders (non-MindDataset).

        Args:
            data_loader (Any): Data loader instance.
            data_batch_size (int): Batch size for the dataset.
            drop_remainder (bool): Whether to drop remainder samples.
            input_columns (list): Input column names.
            output_columns (list): Output column names.
            micro_batch_num (int): Number of micro batches.
            dynamic_batch (bool): Whether to use dynamic batching.
            divisor (int): Divisor for dynamic batching.
            remainder (int): Remainder for dynamic batching.
            pad_token_id (int): Padding token ID.
            num_parallel_workers (int): Number of parallel workers.
            use_compressed_eod_mask (bool): Whether to use compressed EOD mask.

        Returns:
            Any: Configured dataset instance.
        """
        type_cast_op = TypeCast(mstype.int32)

        dataset = dataset_batch_func(data_loader,
                                     data_batch_size=data_batch_size, drop_remainder=drop_remainder,
                                     output_columns=input_columns, eod_reset=False,
                                     micro_batch_num=micro_batch_num,
                                     dynamic_batch=dynamic_batch, divisor=divisor, remainder=remainder,
                                     use_compressed_eod_mask=use_compressed_eod_mask, pad_token_id=pad_token_id,
                                     num_parallel_workers=num_parallel_workers)

        dataset = dataset.project(columns=input_columns)
        for input_arg in input_columns:
            if input_arg in CAST_TO_INT_COLUMNS:
                dataset = get_dataset_map(dataset, type_cast_op,
                                          input_columns=input_arg)
        if output_columns:
            dataset = dataset.project(columns=output_columns)
        return dataset

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches
    def create_dataset(self, data_loader: Any, data_batch_size: int = 1, drop_remainder: bool = False,
                       input_columns: list = None, output_columns: list = None,
                       num_shards: int = None, shard_id: int = None, eod_reset: bool = False, eod_token_id: int = None,
                       dynamic_batch: bool = False, divisor: int = None, remainder: int = None,
                       pad_token_id: int = None,
                       micro_batch_num: int = 1, num_parallel_workers: int = None, use_llm_token_profile: bool = False,
                       profile_llm_token_config: dict = None) -> Any:
        """
        Create and configure dataset with specified parameters for language model training.

        This method creates a complete dataset pipeline including data loading, batching,
        padding, and optional EOD reset functionality based on the data loader type.

        Args:
            data_loader (Any): Data loader instance providing the raw data.
            data_batch_size (int): Batch size for the dataset. Defaults to 1.
            drop_remainder (bool): Whether to drop remainder samples that don't fit in a batch.
                Defaults to False.
            input_columns (list, optional): Input column names for the dataset. Defaults to None.
            output_columns (list, optional): Output column names for the dataset. Defaults to None.
            num_shards (int, optional): Number of dataset shards for distributed training.
                Defaults to None.
            shard_id (int, optional): Shard ID for current process in distributed training.
                Defaults to None.
            eod_reset (bool): Whether to reset position IDs and attention masks at EOD tokens.
                Only supported for MindDataset. Defaults to False.
            eod_token_id (int, optional): EOD (End Of Document) token ID. Required when
                eod_reset is True. Defaults to None.
            dynamic_batch (bool): Whether to use dynamic batching with padding. Defaults to False.
            divisor (int, optional): Divisor for dynamic batch length calculation. Defaults to None.
            remainder (int, optional): Remainder for dynamic batch length calculation. Defaults to None.
            pad_token_id (int, optional): Padding token ID for dynamic batching. Defaults to None.
            micro_batch_num (int): Number of micro batches for gradient accumulation. Defaults to 1.
            num_parallel_workers (int, optional): Number of parallel workers for data processing.
                Defaults to None.
            use_llm_token_profile (bool): Whether to enable token profiling for analysis.
                Defaults to False.
            profile_llm_token_config (dict, optional): Configuration dictionary for token profiling.
                Required when use_llm_token_profile is True. Defaults to None.

        Returns:
            Any: Fully configured dataset instance ready for training.

        Raises:
            ValueError: If eod_reset is enabled for non-MindDataset data loaders, or if
                profile_llm_token_config is None when use_llm_token_profile is True.
        """
        logger.info("Creating Causal Language Model Dataset.")

        if input_columns is None:
            logger.warning("input_columns is None, this may cause issues in dataset processing.")

        if output_columns is None:
            output_columns = input_columns
            logger.info("output_columns not specified, using input_columns as output_columns.")

        if eod_reset and self.data_loader_type != 'MindDataset':
            raise ValueError("eod_reset is only supported for MindDataset data loader, please enable it in DataLoader.")

        use_compressed_eod_mask = self._check_compressed_eod_mask_valid()
        logger.info(f"use_compressed_eod_mask: {use_compressed_eod_mask}")

        if self.data_loader_type == 'MindDataset':
            logger.info("Processing MindDataset data loader")
            dataset = self._create_mindrecord_dataset(
                data_loader, data_batch_size, drop_remainder, input_columns, output_columns,
                num_shards, shard_id, eod_reset, eod_token_id, pad_token_id,
                num_parallel_workers, use_compressed_eod_mask)
        else:
            logger.info(f"Processing {self.data_loader_type} data loader")
            dataset = self._create_custom_dataset(
                data_loader, data_batch_size, drop_remainder, input_columns, output_columns,
                micro_batch_num, dynamic_batch, divisor, remainder, pad_token_id,
                num_parallel_workers, use_compressed_eod_mask)

        if use_llm_token_profile:
            logger.info("LLM token profiling is enabled")
            if profile_llm_token_config is None:
                raise ValueError("profile_llm_token_config cannot be None when use_llm_token_profile is True")
            dataset = self._profile_llm_token(
                dataset, profile_token_config=profile_llm_token_config)

        logger.info("Dataset created successfully")
        return dataset

    def _check_compressed_eod_mask_valid(self) -> bool:
        """
        Check if compressed EOD mask is valid for the data loader.

        Returns:
            bool: True if compressed EOD mask is enabled, False otherwise.
        """
        data_loader = self.data_loader_config
        if (hasattr(data_loader, 'config') and data_loader.config and
                data_loader.config.create_compressed_eod_mask):  # megatron dataset
            logger.debug("Compressed EOD mask is enabled for Megatron dataset")
            return True
        if (hasattr(data_loader, 'adaptor_config') and data_loader.adaptor_config and
                data_loader.adaptor_config.compress_mask):  # common dataloader
            logger.debug("Compressed mask is enabled for common dataloader")
            return True

        logger.debug("Compressed EOD mask is not enabled")
        return False

    def _profile_llm_token(self, dataset: Any, profile_token_config: dict) -> Any:
        """
        Profile LLM tokens and save statistics to CSV files.

        Args:
            dataset: Input dataset to profile.
            profile_token_config (dict): Configuration for token profiling.

        Returns:
            Any: Dataset with token counting operations applied.
        """
        kwargs = {}
        # Check if token_monitor_config exists and is a dict
        if isinstance(profile_token_config, dict):
            kwargs = profile_token_config.copy()
            if 'max_token_id' in kwargs and kwargs['max_token_id'] == "inf":
                kwargs['max_token_id'] = np.inf

        logger.info("Token profiling is enabled. Saving token counts to output/token_counts_output_csv")
        dataset = get_dataset_map(dataset, operations=[self.perform_token_counting(**kwargs)])

        return dataset

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def _process_custom_llm_data(self, column_names: List[str], num_shards: int = None, shard_id: int = None,
                                 python_multiprocessing: bool = False, num_parallel_workers: int = 1) -> Any:
        """
        Process custom text data using the specified data loader.

        Args:
            column_names (List[str]): Column names for the dataset.
            num_shards (int, optional): Number of dataset shards. Defaults to None.
            shard_id (int, optional): Shard ID for current process. Defaults to None.
            python_multiprocessing (bool): Whether to use Python multiprocessing. Defaults to False.
            num_parallel_workers (int): Number of parallel workers. Defaults to 1.

        Returns:
            Any: Processed dataset instance.
        """
        logger.info(f"Processing custom LLM data with {self.data_loader_type}")
        dataset_dir = self.data_loader_config.pop("dataset_dir", None)
        dataset = build_dataset_loader(
            self.data_loader_config, default_args={
                'dataset_dir': dataset_dir, 'num_shards': num_shards, 'shard_id': shard_id,
                'column_names': column_names, 'python_multiprocessing': python_multiprocessing,
                'num_parallel_workers': num_parallel_workers})
        logger.info("Custom LLM data processed successfully")
        return dataset

    def _get_mindrecord_files_from_dir(self, data_dir: str) -> Union[str, List[str]]:
        """
        Get MindRecord files from directory.

        Args:
            data_dir (str): Directory path or file path.

        Returns:
            Union[str, List[str]]: Dataset file path(s).

        Raises:
            FileNotFoundError: If no dataset files are found.
        """
        mind_compile = re.compile(r"mindrecord\d*$")
        dataset_files = []

        if os.path.isdir(data_dir):
            for r, _, f in os.walk(data_dir):
                for file in f:
                    if re.findall(mind_compile, file) or file.endswith(".tfrecord"):
                        dataset_files.append(os.path.join(r, file))
            dataset_files.sort()
            logger.info(f"Found {len(dataset_files)} dataset files in directory")
        else:
            if re.findall(mind_compile, data_dir) or data_dir.endswith(".tfrecord"):
                dataset_files = data_dir
                logger.info(f"Using single dataset file: {dataset_files}")

        if not dataset_files:
            raise FileNotFoundError(
                f"No dataset file is found. Please check whether the path "
                f"`{data_dir}` indicated by dataloader.dataset_dir is correct.")
        return dataset_files

    def _get_dataset_files(self, data_loader_config: Any) -> Union[str, List[str]]:
        """
        Get dataset files from configuration.

        Args:
            data_loader_config: Data loader configuration object.

        Returns:
            Union[str, List[str]]: Dataset file path(s).

        Raises:
            ValueError: If data_loader_config is invalid.
        """
        if data_loader_config.dataset_dir:
            data_dir = data_loader_config.pop("dataset_dir")
            return self._get_mindrecord_files_from_dir(data_dir)

        if data_loader_config.dataset_files:
            dataset_files = data_loader_config.dataset_files
            if isinstance(dataset_files, (list, tuple)):
                dataset_files = list(dataset_files)
            logger.info(f"Using {len(dataset_files)} dataset files from configuration")
            return dataset_files

        raise ValueError(
            f"data_loader must contain dataset_dir or dataset_files, "
            f"but get {data_loader_config}.")

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def _process_mindrecord_data(self, column_names: List[str], num_shards: int = None, shard_id: int = None,
                                 python_multiprocessing: bool = False, num_parallel_workers: int = 1) -> Any:
        """
        Process MindRecord data files.

        Args:
            column_names (List[str]): Column names for the dataset.
            num_shards (int, optional): Number of dataset shards. Defaults to None.
            shard_id (int, optional): Shard ID for current process. Defaults to None.
            python_multiprocessing (bool): Whether to use Python multiprocessing. Defaults to False.
            num_parallel_workers (int): Number of parallel workers. Defaults to 1.

        Returns:
            Any: Processed MindRecord dataset instance.

        Raises:
            FileNotFoundError: If no dataset files are found.
            ValueError: If data_loader_config is invalid.
        """
        logger.info("Processing MindRecord data")
        data_loader_config = self.data_loader_config
        dataset_files = self._get_dataset_files(data_loader_config)

        if not self._is_full_batch():
            data_loader_config = self._reset_num_samples(num_shards=num_shards)

        dataset = build_dataset_loader(
            data_loader_config, default_args={
                'dataset_files': dataset_files, 'num_shards': num_shards, 'shard_id': shard_id,
                'columns_list': column_names, 'python_multiprocessing': python_multiprocessing,
                'num_parallel_workers': num_parallel_workers})
        logger.info("MindRecord data processed successfully")
        return dataset

    def _reset_num_samples(self, num_shards: int) -> dict:
        """
        Reset num_samples for full_batch=False mode.

        Args:
            num_shards (int): Number of dataset shards.

        Returns:
            dict: Updated data loader configuration.
        """
        data_loader_config = self.data_loader_config
        num_samples = data_loader_config.get('num_samples')
        if num_samples is None:
            logger.debug("num_samples not set in data_loader_config, returning config as-is")
            return data_loader_config

        # dataset_config.device_num is equal to dp
        cur_num_samples = num_samples // num_shards
        logger.info(f"If set full_batch=False and num_samples, "
                    f"num_samples will reset from {num_samples} to {cur_num_samples}.")
        data_loader_config.num_samples = cur_num_samples
        return data_loader_config

    def _check_device_rank_for_parallel(self, shard_id: int, num_shards: int) -> Tuple[Optional[int], Optional[int]]:
        """
        Check device rank for auto parallel mode and adjust shard info accordingly.

        Args:
            shard_id (int): Current shard ID.
            num_shards (int): Current number of shards.

        Returns:
            Tuple[Optional[int], Optional[int]]: Adjusted shard ID and number of shards.
        """
        if self._is_semi_full_batch():
            shard_id = None
            num_shards = None
            logger.debug("Semi full batch mode detected, setting shard_id and num_shards to None")
        return shard_id, num_shards

    def is_create_compressed_eod_mask(self):
        """
        Check if compressed EOD mask creation is enabled in the data loader configuration.

        Returns:
            bool: True if compressed EOD mask creation is enabled, False otherwise.
        """
        if self.data_loader_type == "BlendedMegatronDatasetDataLoader":
            return self.data_loader_config.config.get("create_compressed_eod_mask", False)
        return self.data_loader_config.get("create_compressed_eod_mask", False)

    def is_create_attention_mask(self):
        """
        Check if attention mask creation is enabled in the data loader configuration.

        Returns:
            bool: True if attention mask creation is enabled, False otherwise.
        """
        if self.data_loader_type == "BlendedMegatronDatasetDataLoader":
            return self.data_loader_config.config.get("create_attention_mask", False)
        return self.data_loader_config.get("create_attention_mask", False)

    @staticmethod
    def _is_semi() -> bool:
        """
        Check if current parallel mode is semi auto parallel.

        Returns:
            bool: True if parallel mode is semi auto parallel or auto parallel.
        """
        is_semi_mode = ms.context.get_auto_parallel_context("parallel_mode") in ['semi_auto_parallel', 'auto_parallel']
        logger.debug(f"Parallel mode check: is_semi={is_semi_mode}")
        return is_semi_mode

    @staticmethod
    def _is_full_batch() -> bool:
        """
        Check if full batch mode is enabled.

        Returns:
            bool: True if full batch mode is enabled.
        """
        is_full_batch_mode = ms.context.get_auto_parallel_context("full_batch")
        logger.debug(f"Full batch mode check: is_full_batch={is_full_batch_mode}")
        return is_full_batch_mode

    def _is_semi_full_batch(self) -> bool:
        """
        Check if current configuration is semi full batch.

        Returns:
            bool: True if both semi parallel and full batch modes are enabled.
        """
        is_semi_full = self._is_semi() and self._is_full_batch()
        logger.debug(f"Semi full batch check: is_semi_full_batch={is_semi_full}")
        return is_semi_full

    @staticmethod
    def _is_data_parallel() -> bool:
        """
        Check if current parallel mode is data parallel.

        Returns:
            bool: True if parallel mode is data parallel.
        """
        is_dp_mode = ms.context.get_auto_parallel_context("parallel_mode") == ms.context.ParallelMode.DATA_PARALLEL
        logger.debug(f"Data parallel mode check: is_data_parallel={is_dp_mode}")
        return is_dp_mode

    @staticmethod
    def perform_token_counting(top_n: int = 10, min_token_id: int = 0, max_token_id: Union[int, float] = np.inf,
                               save_path: str = "output/token_counts_output_csv/") -> Callable:
        """
        Create a token counting function with specified parameters.

        Args:
            top_n (int): Number of top tokens to record. Defaults to 10.
            min_token_id (int): Minimum token ID to consider. Defaults to 0.
            max_token_id (Union[int, float]): Maximum token ID to consider. Defaults to np.inf.
            save_path (str): Directory path to save token count files. Defaults to "output/token_counts_output_csv/".

        Returns:
            Callable: Token counting function.
        """
        logger.debug(f"Creating token counter with top_n={top_n}, min_token_id={min_token_id}, "
                     f"max_token_id={max_token_id}, save_path={save_path}")
        token_counter = TokenCounter(top_n, min_token_id, max_token_id, save_path)
        return token_counter.count_tokens

    def get_data_loader_type(self):
        """
        Get the type of the data loader being used.

        Returns:
            str: The data loader type.
        """
        return self.data_loader_type


# pylint: disable=too-many-instance-attributes
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

    def __init__(self, top_n: int = 10, min_token_id: int = 0, max_token_id: Union[int, float] = np.inf,
                 save_path: str = "output/token_counts_output_csv/") -> None:
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
        logger.debug(f"TokenCounter initialized with top_n={top_n}, min_token_id={min_token_id}, "
                     f"max_token_id={max_token_id}, save_path={save_path}")

    def initialize_file(self) -> None:
        """
        This method initializes the file for writing by clearing any existing content.
        """
        rank_id = get_rank()
        os.makedirs(self.saved_directory, exist_ok=True)
        filename = os.path.join(self.saved_directory, f"rank_{rank_id}_token_counts.csv")
        logger.info(f"Initializing token count file: {filename}")

        # Clear existing file content
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            _ = csv.writer(csvfile)
        set_safe_mode_for_file_or_dir(filename)

        self.initialized = True
        self.token_count_pairs_header_written = False
        logger.debug("Token count file initialized successfully")

    # pylint: disable=too-many-locals
    def count_tokens(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Count tokens and save in csv file.

        Args:
            input_ids (np.ndarray): Input token IDs to count.

        Returns:
            np.ndarray: Original input_ids array.
        """
        if not self.initialized:
            logger.debug("Initializing token counter file")
            self.initialize_file()

        tokens = input_ids.flatten()
        unique_tokens, counts = np.unique(tokens, return_counts=True)
        actual_min = np.min(unique_tokens)
        actual_max = np.max(unique_tokens)
        logger.debug(f"Token count step {self.step_num}: min_id={actual_min}, max_id={actual_max}, "
                     f"unique_tokens={len(unique_tokens)}")

        with self.lock:
            if actual_min < self.min_token_id:
                logger.warning("Step %d: Token ID range warning: Min token ID (%d) is below %d.",
                               self.step_num, actual_min, self.min_token_id)

            if actual_max > self.max_token_id:
                logger.warning("Step %d: Token ID range warning: Max token ID (%d) is above %d.",
                               self.step_num, actual_max, self.max_token_id)

            token_count_pairs = np.array(list(zip(unique_tokens, counts)),
                                         dtype=[('token_id', 'int32'), ('count', 'uint16')])
            logger.debug(f"Created token count pairs with {len(token_count_pairs)} entries")

            if self.top_n:
                token_count_pairs = np.sort(token_count_pairs, order='count')[::-1][:self.top_n]
                logger.debug(f"Filtered to top {min(self.top_n, len(token_count_pairs))} tokens")

            rank_id = get_rank()
            filename = os.path.join(self.saved_directory, f"rank_{rank_id}_token_counts.csv")

            with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)

                if not self.token_count_pairs_header_written:
                    header = ['step_num', 'min_id', 'max_id']
                    for i in range(len(token_count_pairs)):
                        header.append(f'token_id_{i + 1}')
                        header.append(f'count_{i + 1}')
                    csv_writer.writerow(header)
                    self.token_count_pairs_header_written = True
                    logger.debug("Written CSV header")

                row = [self.step_num, actual_min, actual_max]
                for token_id, count in token_count_pairs:
                    row.append(token_id)
                    row.append(count)
                csv_writer.writerow(row)
                logger.debug(f"Written row with {len(row)} columns to CSV")

            logger.debug(f"RANK {rank_id}: Appended token counts to {filename} with {len(token_count_pairs)} entries")
            self.step_num += 1

        return input_ids
