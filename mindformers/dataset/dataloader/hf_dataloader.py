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
"""HF DataLoader"""

import os
import inspect
import json
import itertools
from dataclasses import dataclass
from typing import Optional, Union
from copy import deepcopy
import numpy as np
import datasets
from datasets.distributed import split_dataset_by_node

import mindspore as ms
from mindspore import Tensor, ops
from mindspore.dataset import GeneratorDataset

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.logger import logger
from mindformers.tools.utils import get_real_group_size
from mindformers.version_control import skip_barrier_controller
import mindformers.dataset.handler as custom_process
from mindformers.tools.utils import get_real_rank
from mindformers.tools.utils import FILE_PERMISSION

from .utils import is_dataset_built_on_rank
from .mock_dataloader import BaseMockDataLoader

# Mapping between integer codes and NumPy data type strings
DATASET_DTYPE_MAP = {
    1: 'int32',
    2: 'float32',
    3: 'int64',
    4: 'float16'
}
# Placeholder number set for broadcast dataset info
PLACEHOLDER_ID = 0


def parse_data_shapes(arr, delimiter=-1):
    """Parse data shapes from broadcast data."""
    arr = arr[arr != PLACEHOLDER_ID]
    indices = np.where(arr == delimiter)[0]
    parts = np.split(arr, indices + 1)
    return [part.tolist()[:-1] for part in parts if len(part) > 0 and part[0] != delimiter]


def parse_data_dtypes(arr):
    """Parse data dtypes from broadcast data."""
    arr = arr[arr != PLACEHOLDER_ID]
    return [DATASET_DTYPE_MAP[item] for item in arr]


def _is_pack_mode(config):
    """Determine if the given configuration is set to 'pack mode'."""
    if isinstance(config.process, list):
        for sub_process in config.process:
            if sub_process.get('type') == 'PackingHandler':
                return True
    return bool(getattr(config, "create_attention_mask", False))


def process_legacy_args(**kwargs):
    """Process and adapt legacy configuration arguments into the non-legacy format."""
    replace_args = {}

    # Process handlers
    packing = kwargs.pop('packing', None)
    handler = kwargs.pop('handler', None)
    if handler and isinstance(handler, list):
        cur_handler = []
        for sub_handler in handler:
            if sub_handler.get('type') == 'AlpacaInstructDataHandler':
                # Disable padding if packing or dynamic length is used
                padding = sub_handler.get('padding', True)
                if packing or sub_handler.get('is_dynamic', False):
                    padding = False
                cur_handler.append({
                    'type': 'AlpacaInstructDataHandler',
                    'seq_length': sub_handler.get('seq_length'),
                    'tokenizer': sub_handler.get('tokenizer'),
                    'padding': padding
                })
            elif sub_handler.get('type') == 'PackingHandler':
                pack_strategy = sub_handler.get('pack_strategy', packing)
                cur_handler.append({
                    'type': 'PackingHandler',
                    'seq_length': sub_handler.get('seq_length'),
                    'pack_strategy': pack_strategy
                })
            else:
                cur_handler.append(sub_handler)
        replace_args['handler'] = cur_handler

    # Process adaptor config
    adaptor_config = kwargs.pop('adaptor_config', None)
    if isinstance(adaptor_config, dict):
        replace_args.update({
            'create_compressed_eod_mask': adaptor_config.get('compress_mask', False),
            'compressed_eod_mask_length': adaptor_config.get('eod_pad_length', 128)
        })

    # Merge updated arguments
    kwargs.update(replace_args)
    return kwargs


@dataclass
class HFDataLoaderConfig:
    """
    Configuration object for HFDataLoader.

    This dataclass stores configuration parameters for loading and processing
    HuggingFace datasets, as well as additional options for special mask handling
    and mock data generation.

    Args:
        load (dict): Arguments for loading HuggingFace datasets.
                     Must be provided, otherwise an error will be raised.
        process (dict | list, optional): Processing configuration for HuggingFace datasets.
                                         Can be a single dictionary or a list of dictionaries.
        create_attention_mask (bool, optional): Whether to create an attention mask.
        create_compressed_eod_mask (bool, optional): Whether to enable
            the creation of a compressed end-of-document (EOD) mask.
            This is used in TND layout FlashAttention modules with `actual_seq_len`.
        compressed_eod_mask_length (int, optional): Maximum `actual_seq_len` length
            when `create_compressed_eod_mask` is set to True. Default is 128.
        use_broadcast_data (bool, optional): Whether to enable broadcasted mock data
            in data-parallel groups. Default is True.
        shuffle (bool, optional): Whether to shuffle the dataset (requires random-access input). Default: `False`.
    """

    load: dict = None

    process: Optional[Union[dict, list]] = None

    create_attention_mask: bool = False

    create_compressed_eod_mask: bool = False

    compressed_eod_mask_length: int = 128

    use_broadcast_data: bool = True

    shuffle: bool = False

    def __post_init__(self) -> None:
        """Post-initialization checks and setup."""
        if self.load is None:
            raise ValueError("`load` in HFDataLoader must be a dict, but got None.")


@dataclass
class HFStreamingConfig(HFDataLoaderConfig):
    """
    Extended configuration for streaming Hugging Face datasets.

    Adds state management and checkpointing parameters for large-scale
    streaming dataset loading.

    Attributes:
        streaming (bool): Enable streaming mode.
        size (int): Total dataset size across all shards.
        dataset_state_dir (str): Directory path for saving dataset states.
        save_step (int): Frequency (in steps) to save dataset state.
        resume_step (int): Step number to resume dataset iteration from.
        batch_size (int): Batch size used in iteration (for step calculations).
    """

    streaming: bool = False

    size: int = None

    dataset_state_dir: str = None

    save_step: int = None

    resume_step: int = None

    batch_size: int = 1

    def __post_init__(self):
        """Ensure mandatory streaming parameters are set."""
        if self.size is None or self.dataset_state_dir is None:
            raise ValueError(
                "`size` and `dataset_state_dir` must be provided when `streaming=True`."
            )


@MindFormerRegister.register(MindFormerModuleType.DATASET_LOADER)
class HFDataLoader:
    """
    HFDataLoader is responsible for loading and processing Hugging Face datasets,
    supporting features like sharding, multiprocessing, and shuffling.
    It returns a `mindspore.dataset.GeneratorDataset` instance ready for training.

    Parameters:
        shard_id (Optional[int]): ID of the current shard in distributed mode. Default: `None`.
        num_shards (Optional[int]): Total number of dataset shards. Default: `None`.
        python_multiprocessing (Optional[bool]): Whether to use multiple processes for Python-based operations.
            Useful when the processing workload is computationally heavy. Default: `True`.
        num_parallel_workers (Optional[int]): Number of parallel worker threads/processes for fetching data.
            Default: `1`.
        **kwargs: Additional keyword arguments for legacy compatibility and loader customization.

    Workflow in __new__():
        1. Process legacy arguments to ensure compatibility.
        2. Build a `HFDataLoaderConfig` object.
        3. Optionally use mock datasets in distributed training (non-main ranks).
        4. Load dataset from source (HF Hub or local).
        5. Optionally process dataset (formatting, cleaning, packing, etc.).
        6. Wrap dataset in a MindSpore `GeneratorDataset`.
        7. Optionally synchronize and broadcast dataset info in distributed mode.

    Support `handler` for source datasets process method in **kwargs:
        1. rename_column: original_column_name, new_column_name;
        2. remove_columns: column_names;
        3. sort: column_names, reverse(defaults to False);
        4. shuffle: seed;
        5. flatten;
    For example:
        handler:
            rename_column:
                original_column_name: 'col1'
                new_column_name: 'col2'
            shuffle:
                seed: 42

    Returns:
        mindspore.dataset.GeneratorDataset: Dataset ready for training.
    """

    def __new__(
            cls,
            num_shards: int = None,
            shard_id: int = None,
            python_multiprocessing: bool = True,
            num_parallel_workers: int = 1,
            **kwargs
    ):
        ms_dataset_args = {
            'num_shards': num_shards,
            'shard_id': shard_id,
            'python_multiprocessing': python_multiprocessing,
            'num_parallel_workers': num_parallel_workers,
        }
        kwargs = process_legacy_args(**kwargs)

        # Get distributed training environment info
        world_size = get_real_group_size()

        # 1. Build HFDataLoaderConfig object
        config = cls._build_config(**kwargs)

        # In distributed mode: use mock dataset for non-main ranks to avoid redundant loading
        if world_size > 1 and config.use_broadcast_data and not is_dataset_built_on_rank():
            return cls._build_mock_dataset(config, num_shards, shard_id)

        # 2. Load dataset (supports HF Hub or local files)
        dataset = cls.load_dataset(config)

        # 3. Process dataset (if processing configuration is provided)
        if config.process is not None:
            dataset, config = cls.process_dataset(config, dataset)

        # 4. Wrap dataset into a MindSpore GeneratorDataset
        dataset = cls.wrap_dataset(config, dataset, **ms_dataset_args)

        # 5. In distributed mode, synchronize all ranks after dataset initialization
        if world_size > 1:
            logger.info(" > start barrier for all dataset init ... ")
            skip_barrier_controller()  # Ensure all ranks reach the same point before proceeding

            if config.use_broadcast_data:
                cls._broadcast_dataset_info(dataset.source)

        return dataset

    @classmethod
    def load_dataset(cls, config: HFDataLoaderConfig):
        """Load datasets, support HF dataset loading methods now."""
        load_func_name = config.load.pop('load_func', 'load_dataset')
        logger.info(
            f" > using `datasets.{load_func_name}` to load dataset. "
            f"Pass 'load_func' in config.load to change this behavior."
        )

        load_func = getattr(datasets, load_func_name)
        dataset = load_func(**config.load)

        if config.load.get('split') is None and not isinstance(dataset, datasets.Dataset):
            logger.info(" > `split` not provided for datasets, use 'train' default.")
            dataset = dataset.get('train')

        return dataset

    @classmethod
    def process_dataset(cls, config: HFDataLoaderConfig, dataset):
        """Process datasets, support HF dataset processing methods now."""
        if isinstance(config.process, dict):
            config.process = [config.process]

        for process_args in config.process:
            sub_process = deepcopy(process_args)
            if not isinstance(sub_process, dict) or 'type' not in sub_process:
                raise ValueError("`process` in HFDataLoader must be a dict or list, "
                                 "and dict should contain the 'type' key.")

            process_type = sub_process.pop('type')
            # 1. processing dataset with methods in `handler` module if existed
            if hasattr(custom_process, process_type):
                # In streaming mode, skip packing handler (packing requires random access).
                if process_type == 'PackingHandler' and getattr(config, 'streaming', False):
                    logger.info(" > skipping PackingHandler because streaming mode is enabled.")
                    continue
                process_func = getattr(custom_process, process_type)
                logger.info(f" > processing `{process_type}` in `handler` module ...")
                dataset = cls._process_custom(process_func, dataset, **sub_process)

            # 2. processing dataset with methods with `datasets` source methods if local methods not existed
            elif hasattr(dataset, process_type):
                logger.info(f" > processing `{process_type}` in `datasets` source module ...")
                dataset = cls._process_source(process_type, dataset, **sub_process)

            else:
                raise ValueError(
                    f"process.type {process_type} not found in `handler` module or "
                    f"`datasets` open source processing methods.")

        return dataset, config

    @classmethod
    def wrap_dataset(cls,
                     config: HFDataLoaderConfig,
                     dataset,
                     shard_id=None,
                     num_shards=None,
                     python_multiprocessing=False,
                     num_parallel_workers=1):
        """Wrap source dataset with Mindspore Dataset."""
        if getattr(config, 'streaming', False):
            hf_dataset = HFIterableDataset(config, dataset, num_shards, shard_id)
            if num_parallel_workers > 1:
                num_parallel_workers = 1
                logger.warning(
                    "Streaming mode only supports 'num_parallel_workers=1'. Automatically resetting the value.")
        else:
            hf_dataset = HFDataset(config, dataset)
        dataset = GeneratorDataset(
            hf_dataset,
            column_names=hf_dataset.column_names,
            num_parallel_workers=num_parallel_workers,
            python_multiprocessing=python_multiprocessing,
            shard_id=shard_id,
            num_shards=num_shards,
            shuffle=config.shuffle,
        )
        return dataset

    @staticmethod
    def _process_custom(process_func, dataset, **kwargs):
        """Apply custom data processing method."""
        if inspect.isclass(process_func):
            process_func = process_func(**kwargs)
            dataset = process_func(dataset)
        elif inspect.isfunction(process_func):
            dataset = process_func(dataset, **kwargs)
        else:
            raise ValueError(
                f"process.type should be Class or Function in `handler` module, "
                f"but got {type(process_func)}.")
        return dataset

    @staticmethod
    def _process_source(process_name, dataset, **kwargs):
        """Apply source datasets processing method."""
        dataset = getattr(dataset, process_name)(**kwargs)
        return dataset

    @staticmethod
    def _build_config(**kwargs):
        """Build dataloader config from input args."""
        create_attention_mask = kwargs.pop('create_attention_mask', False)
        create_compressed_eod_mask = kwargs.pop('create_compressed_eod_mask', False)
        compressed_eod_mask_length = kwargs.pop('compressed_eod_mask_length', 128)
        use_broadcast_data = kwargs.pop('use_broadcast_data', True)
        handler = kwargs.pop('handler', None)
        shuffle = kwargs.pop('shuffle', False)

        streaming = kwargs.get('streaming', False)
        size = kwargs.pop('size', None)
        dataset_state_dir = kwargs.pop('dataset_state_dir', None)
        save_step = kwargs.pop('save_step', None)
        resume_step = kwargs.pop('resume_step', None)
        batch_size = kwargs.pop('batch_size', 1)

        # filter out invalid parameters passed from upper-level interfaces.
        invalid_args = ['dataset_dir', 'column_names', 'type']
        for args in invalid_args:
            kwargs.pop(args, None)

        if streaming:
            return HFStreamingConfig(
                load=kwargs,
                process=handler,
                create_attention_mask=create_attention_mask,
                create_compressed_eod_mask=create_compressed_eod_mask,
                compressed_eod_mask_length=compressed_eod_mask_length,
                use_broadcast_data=use_broadcast_data,
                shuffle=shuffle,
                streaming=streaming,
                size=size,
                dataset_state_dir=dataset_state_dir,
                save_step=save_step,
                resume_step=resume_step,
                batch_size=batch_size,
            )

        return HFDataLoaderConfig(
            load=kwargs,
            process=handler,
            create_attention_mask=create_attention_mask,
            create_compressed_eod_mask=create_compressed_eod_mask,
            compressed_eod_mask_length=compressed_eod_mask_length,
            use_broadcast_data=use_broadcast_data,
            shuffle=shuffle,
        )

    @staticmethod
    def _build_mock_dataset(config, num_shards=None, shard_id=None):
        """Build a mock dataset for distributed training (non-main ranks)."""
        logger.info(" > start barrier for all dataset init ... ")
        skip_barrier_controller()  # mock dataset only support parallel mode

        logger.info(" > start receive dataset info from main rank.")
        # Initialize placeholder tensors for broadcast reception
        received_data = (
            Tensor([PLACEHOLDER_ID], dtype=ms.int32),
            Tensor([PLACEHOLDER_ID], dtype=ms.int32),
            Tensor([PLACEHOLDER_ID], dtype=ms.int32),
        )

        # Receive dataset metadata from main rank (rank 0)
        dataset_size, num_columns, seq_length = ops.Broadcast(0)(received_data)
        mock_data = {
            'dataset_size': dataset_size.numpy()[0],  # Total number of samples
            'num_columns': num_columns.numpy()[0],  # Number of dataset columns
            'seq_length': seq_length.numpy()[0]  # Sequence length of each sample
        }

        logger.info(f"\n > received dataset info: \n"
                    f"   size:        {mock_data.get('dataset_size')} \n"
                    f"   num_columns: {mock_data.get('num_columns')} \n"
                    f"   seq_length:  {mock_data.get('seq_length')}")

        mock_dataloader = MockHFDataLoader(config, **mock_data)
        mock_dataloader = GeneratorDataset(
            mock_dataloader,
            column_names=mock_dataloader.mock_columns,
            num_shards=num_shards,
            shard_id=shard_id)
        return mock_dataloader

    @classmethod
    def _broadcast_dataset_info(cls, dataset):
        """
        Broadcast dataset metadata from the main rank to all other ranks
        in a distributed training environment.
        This method is typically called after the real dataset has been
        built on the main rank. Other ranks will receive this metadata
        and use it to construct a mock dataset with the same structure.
        """
        # iter dataset
        sample = next(iter(itertools.tee(dataset, 1)[0]))
        sample_columns = dataset.column_names

        dataset_size = len(dataset)
        num_columns = len(sample_columns)
        seq_length = len(sample[0])  # Assuming the first column is `input_ids`

        logger.info(f"\n > build real dataset completed, broadcast dataset info: \n"
                    f"   size:       {dataset_size} \n"
                    f"   columns:    {sample_columns} \n"
                    f"   seq_length: {seq_length}")
        dataset_size = Tensor(dataset_size, dtype=ms.int32)
        num_columns = Tensor(num_columns, dtype=ms.int32)
        seq_length = Tensor(seq_length, dtype=ms.int32)

        logger.info(" > start broadcast dataset info to other rank.")
        ops.Broadcast(0)((dataset_size, num_columns, seq_length))


class HFDataset:
    """
    A dataset wrapper for Hugging Face datasets that supports both regular and packed data formats.

    This class provides iteration, indexing, and length support, and it can optionally
    generate compressed EOD (end-of-document) masks for packed datasets.

    Args:
        config (HFDataLoaderConfig): Configuration object defining dataset handling,
                                         including packed mode and mask options.
        dataset (dict-like): Loaded dataset, typically from Hugging Face or a mock loader.
    """

    def __init__(self, config, dataset):
        self.use_packed_data = _is_pack_mode(config)
        self.create_compressed_eod_mask = config.create_compressed_eod_mask
        self.compressed_eod_mask_length = config.compressed_eod_mask_length

        self.dataset = dataset

        # Define column names based on whether packed mode is used
        if self.use_packed_data:
            if config.create_compressed_eod_mask:
                self.column_names = ['input_ids', 'labels', 'loss_mask', 'position_ids', 'actual_seq_len']
            else:
                self.column_names = ['input_ids', 'labels', 'loss_mask', 'position_ids', 'attention_mask']
        else:
            self.column_names = list(next(iter(deepcopy(dataset))).keys())
            self._check_columns(self.column_names)

        self.dataset_size = len(dataset)

    def __getitem__(self, idx):
        """
        Retrieve a dataset sample by index.

        Args:
            idx (int): Sample index.

        Returns:
            tuple or generator: Dataset sample either in packed or regular format.
        """
        if self.use_packed_data:
            return self._iter_packed_data(int(idx))
        return self._iter_data(int(idx))

    def __len__(self):
        """Return total number of samples in the dataset."""
        return self.dataset_size

    def _iter_data(self, idx):
        """Retrieve a single sample from a regular (non-packed) dataset."""
        sample = self.dataset[idx]
        sample = tuple(sample[col] for col in self.column_names)
        return sample

    def _iter_packed_data(self, idx):
        """
        Retrieve a single sample from a packed dataset.

        This handles sequences that are concatenated together and may include
        compressed EOD masks, position IDs, and attention masks.

        Args:
            idx (int): Sample index.

        Returns:
            tuple: Tokens, labels, loss mask, position IDs, and attention mask.
        """
        sample = self.dataset[idx]
        input_ids = sample.get('input_ids')
        labels = sample.get('labels')
        actual_seq_len = sample.get('actual_seq_len')

        if input_ids is None or labels is None or actual_seq_len is None:
            raise ValueError("Packed dataset sample missing required keys: 'input_ids','labels' or 'actual_seq_len'.")

        return _get_packed_data(
            input_ids,
            labels,
            actual_seq_len,
            self.create_compressed_eod_mask,
            self.compressed_eod_mask_length,
        )

    @staticmethod
    def _check_columns(columns):
        supported_columns = [
            'input_ids', 'labels', 'loss_mask', 'position_ids', 'attention_mask', 'actual_seq_len'
        ]
        invalid_columns = list(set(columns) - set(supported_columns))
        if invalid_columns:
            raise ValueError(
                f"Currently, only columns `{supported_columns}` are supported. The processed data contains "
                f"unsupported column(s): `{invalid_columns}`. If you need to use these columns to train a custom "
                f"model, you may remove this restriction and disable the `use_broadcast_data` feature. "
                f"This will not affect normal dataset iteration."
            )


class MockHFDataLoader(BaseMockDataLoader):
    """
    Mock version of HFDataLoader used for distributed training or testing
    without requiring the real dataset.

    This class simulates a dataset with the same column structure, shapes, and
    data types as the real dataset, enabling non-main ranks to work with mock
    data in parallel or to test dataset pipelines.

    Args:
        config (HFDataLoaderConfig): Configuration object containing packed mode and mask settings.
        dataset_size (int): Number of samples in the mock dataset.
        num_columns (int): Number of columns in the mock dataset (used for non-packed mode).
        seq_length (int): Sequence length of the mock dataset.
    """

    def __init__(self, config: HFDataLoaderConfig, dataset_size, num_columns, seq_length):
        if _is_pack_mode(config):
            # Define mock column names for packed datasets
            mock_columns = ['input_ids', 'labels', 'loss_mask', 'position_ids']

            # Set data shapes for each packed column
            data_shapes = [
                [seq_length],  # input_ids
                [seq_length],  # labels
                [seq_length],  # loss_mask
                [seq_length]  # position_ids
            ]
            if config.create_compressed_eod_mask:
                mock_columns.append('actual_seq_len')
                data_shapes.append([config.compressed_eod_mask_length])
            else:
                mock_columns.append('attention_mask')
                data_shapes.append([1, seq_length, seq_length])

        elif num_columns == 1:
            mock_columns = ['input_ids']
            data_shapes = [[seq_length]]
        elif num_columns == 2:
            mock_columns = ['input_ids', 'labels']
            data_shapes = [[seq_length], [seq_length]]
        else:
            raise ValueError("For `MockHFDataLoader`, currently only the two columns `input_ids` and `labels` "
                             "are supported, but more than 2 columns were received.")

        # All columns use int32 dtype
        data_dtypes = ['int32'] * len(mock_columns)

        super().__init__(mock_columns, data_shapes, data_dtypes, dataset_size)


def _resume_hf_iterable_dataset(dataset, step):
    """
    Resume a Hugging Face iterable dataset from a given training step.
    """
    inner_dataset = dataset
    max_depth = 20  # Prevent infinite recursion when traversing nested structures
    cur_depth = 0

    # Traverse into nested dataset wrappers to find the actual HF dataset source
    while not hasattr(inner_dataset, 'source'):
        if inner_dataset and isinstance(inner_dataset, list):
            inner_dataset = inner_dataset[0]
        elif hasattr(inner_dataset, 'children'):
            inner_dataset = inner_dataset.children

        cur_depth += 1
        if cur_depth >= max_depth:
            # Safety check: stop if nesting is unexpectedly deep
            return

    source = inner_dataset.source
    # Ensure the dataset supports `_load_state` before assigning resume step
    if not hasattr(source, '_load_state'):
        return

    logger.info(f"Set HFIterableDataset resume_step={step}")
    # Set the resume step
    source.resume_step = step


class HFIterableDataset:
    """
    Iterable wrapper for streaming HF datasets used with MindSpore GeneratorDataset.

    This class supports:
      - sharding via `datasets.distributed.split_dataset_by_node` when `num_shards` is provided;
      - packing multiple samples into a fixed `seq_length` when PackingHandler is enabled;
      - saving and loading dataset iterator state for resumable streaming.

    Notes on __getitem__:
      - MindSpore may call __getitem__ with arbitrary indices; we treat each call as "advance once"
        and manage an internal iterator to produce the next sample or packed batch.
      - `idx` is not treated as a strict sample index; it's used only for a few control actions
        (e.g., first-call resume). This keeps the iterator behavior stable under MindSpore's expectations.
    """

    def __init__(self, config, dataset, num_shards, shard_id):
        self.size = config.size
        self.state_dir = config.dataset_state_dir
        self.save_step = config.save_step
        self.resume_step = config.resume_step

        self.create_compressed_eod_mask = config.create_compressed_eod_mask
        self.compressed_eod_mask_length = config.compressed_eod_mask_length
        self.pack_config = self._get_pack_config(config)

        # Define column names based on whether packed mode is used
        if self.pack_config:
            if config.create_compressed_eod_mask:
                self.column_names = ['input_ids', 'labels', 'loss_mask', 'position_ids', 'actual_seq_len']
            else:
                self.column_names = ['input_ids', 'labels', 'loss_mask', 'position_ids', 'attention_mask']
        else:
            self.column_names = list(next(iter(deepcopy(dataset))).keys())

        self.shard_id = shard_id if shard_id is not None else 0
        if num_shards:
            dataset = split_dataset_by_node(dataset, shard_id, num_shards)
            per_shard_size = config.size // num_shards
        else:
            per_shard_size = config.size
        self.batch_size = config.batch_size
        self.max_iters = per_shard_size - (per_shard_size % self.batch_size)
        logger.info(f"HFIterableDataset: per_shard_size={per_shard_size}, "
                    f"batch_size={self.batch_size}, max_iters={self.max_iters}")

        self.source = deepcopy(dataset)
        self.dataset = None
        self.dataset_iter = None

        # iteration state
        self.cur_epoch = 1
        self.cur_iter = 0
        self.cur_step = 0
        self.global_step = 0
        self.total_steps = self.max_iters // self.batch_size

        # initialize iterator state
        self._init_state()

        self.local_rank = get_real_rank()

    def __getitem__(self, idx):
        """
        Return the next sample (or packed sample) for the generator.

        Important:
            - `idx` is not treated as absolute index. MindSpore may call __getitem__ with
              many worker-specific indices; we only use idx for first-call resume behaviour.
        """
        if int(idx) == self.shard_id or int(idx) == 0:
            self.cur_iter = 0
            self.cur_step = 0
            self._init_state()

        if self.resume_step and int(idx) > self.shard_id:
            self._load_state(self.resume_step)
            self.resume_step = None

        if self.pack_config:
            sample = self._query_packed_samples()
        else:
            sample = self._query_single_sample()

        # update internal counters
        self.cur_iter += 1
        if (self.cur_iter - 1) % self.batch_size == 0:
            self.cur_step += 1
        self.global_step = (self.cur_epoch - 1) * self.total_steps + self.cur_step
        if self.cur_iter >= self.max_iters:
            self.cur_epoch += 1

        if self.save_step and self.global_step % self.save_step == 0:
            self._save_state(self.global_step)

        return sample

    def _query_single_sample(self):
        """
        Fetch one sample from the underlying iterator, re-initializing if iterator exhausted.
        Returns a tuple of column values (ordered by self.column_names).
        """
        sample = None
        if self.cur_iter >= self.max_iters:
            is_init_state = True
        else:
            is_init_state = False
            try:
                sample = tuple(next(self.dataset_iter).values())
            except StopIteration:
                is_init_state = True

        if is_init_state:
            self._init_state()
            sample = tuple(next(self.dataset_iter).values())
        return sample

    def _load_state(self, step):
        """
        Load dataset iterator state and training counters from JSON saved by `_save_state`.
        """
        state_path = f"{self.state_dir}/step_{step}/dataset_state_rank{self.local_rank}.json"
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"Dataset state file not found: {state_path}")
        logger.info(f'Load dataset state form {state_path}.')

        with open(state_path, 'r', encoding='utf-8') as fp:
            state_dict = json.load(fp)
        train_state = state_dict.pop('_train_state')
        self.cur_iter = train_state.get('cur_iter')
        self.cur_epoch = train_state.get('cur_epoch')
        self.cur_step = train_state.get('cur_step')
        self.global_step = train_state.get('global_step')

        # reinitialize dataset and load internal state if available
        self._init_state()
        self.dataset.load_state_dict(state_dict)

    def _save_state(self, step):
        """
        Save the dataset state and current training counters to JSON.
        The saved file layout:
            <state_dir>/step_<step>/dataset_state_rank<rank>.json
        """
        state_path = f"{self.state_dir}/step_{step}/dataset_state_rank{self.local_rank}.json"
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        state_dict = self.dataset.state_dict()
        state_dict['_train_state'] = {
            'cur_iter': self.cur_iter,
            'cur_epoch': self.cur_epoch,
            'cur_step': self.cur_step,
            'global_step': self.global_step,
        }
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        file = os.open(state_path, flags, FILE_PERMISSION)
        with os.fdopen(file, 'w', encoding='utf-8') as fp:
            json.dump(state_dict, fp, indent=2)
        logger.info(f'Save dataset state to {state_path}.')

    def _init_state(self):
        """Deepcopy the source and create a fresh iterator for iteration."""
        logger.info(
            "Initialize the iterator state, which may occur either at the start of training "
            "or when the iterator is exhausted.")
        self.dataset = deepcopy(self.source)
        self.dataset_iter = iter(self.dataset)

    def _query_packed_samples(self):
        """
        Build a packed sequence by concatenating multiple samples until `seq_length` tokens are reached.
        """
        seq_length = int(self.pack_config.get('seq_length'))
        pack_iter = itertools.tee(self.dataset_iter, 1)[0]
        input_ids = []
        labels = []
        actual_seq_len = [0]

        while len(input_ids) < seq_length:
            try:
                candidate = next(pack_iter)
            except StopIteration:
                self._init_state()
                break

            cur_input_ids = candidate.get('input_ids')
            cur_labels = candidate.get('labels')
            cur_seq_length = len(cur_input_ids) + actual_seq_len[-1]
            if cur_seq_length > seq_length:
                # cannot accept candidate with overflow; stop packing
                break

            # accept candidate: append content and advance real iterator
            input_ids.extend(cur_input_ids)
            labels.extend(cur_labels)
            actual_seq_len.append(cur_seq_length)
            next(self.dataset_iter)

        # padding if necessary
        pad_length = seq_length - len(input_ids)
        if pad_length > 0:
            input_ids.extend([0] * pad_length)
            labels.extend([-100] * pad_length)
        del pack_iter

        data = _get_packed_data(
            input_ids,
            labels,
            actual_seq_len,
            self.create_compressed_eod_mask,
            self.compressed_eod_mask_length,
        )
        return data

    def __len__(self):
        """Return total number of samples in the dataset."""
        return self.size

    @staticmethod
    def _get_pack_config(config):
        """
        Extract the configuration dictionary for `PackingHandler`.
        """
        if isinstance(config.process, list):
            for sub_process in config.process:
                if sub_process.get('type') == 'PackingHandler':
                    return deepcopy(sub_process)
        return {}


def _get_packed_data(
        tokens,
        labels,
        actual_seq_len,
        create_compressed_eod_mask: bool = False,
        compressed_eod_mask_length: int = None,
):
    """
    Convert packed sequence components into final arrays required by training.

    Args:
        tokens (list or np.ndarray): flattened token ids for concatenated subsequences (length == seq_length).
        labels (list or np.ndarray): flattened labels for concatenated subsequences (length == seq_length).
        actual_seq_len (list or np.ndarray): cumulative end positions for subsequences, starting from 0.
            Example: [0, 5, 12] means first subseq length 5, second subseq length 7 (cumulative 12).
        create_compressed_eod_mask (bool): if True, return `attention_mask` as compressed EOD mask array.
        compressed_eod_mask_length (int): maximum number of subsequences to store in compressed mask.

    Returns:
        tuple(tokens_np, labels_np, loss_mask_np, position_ids_np, attention_mask_np_or_actual_seq_len)
        All returned arrays are numpy arrays with dtype `int32`.
    """
    tokens = np.array(tokens)
    labels = np.array(labels)
    actual_seq_len = np.array(actual_seq_len)

    seq_length = len(tokens)

    # loss mask
    loss_mask = labels != -100

    # position ids and attention mask
    position_ids = []
    if create_compressed_eod_mask:
        attention_mask = None
    else:
        attention_mask = np.expand_dims(np.tril(np.ones((seq_length, seq_length))), axis=0)
    pre_seq = 0
    for seq in actual_seq_len:
        sub_pos = np.arange(seq - pre_seq, dtype=np.float32)
        position_ids.append(sub_pos)
        pre_seq = seq
        if attention_mask is not None:
            attention_mask[0, seq:, : seq] = 0

    position_ids.append(np.arange(seq_length - actual_seq_len[-1], dtype=np.float32))
    position_ids = np.concatenate(position_ids)

    if create_compressed_eod_mask:
        if compressed_eod_mask_length < len(actual_seq_len):
            raise ValueError(
                f"The actual_seq_len: {len(actual_seq_len)} in the dataset exceeds the "
                f"compressed_eod_mask_length: {compressed_eod_mask_length}, please check data or "
                f"increase the compressed_eod_mask_length.")

        actual_seq_len = np.pad(
            actual_seq_len, (0, compressed_eod_mask_length - len(actual_seq_len)),
            mode='constant',
            constant_values=seq_length)
        attention_mask = actual_seq_len
    else:
        # reverse attention mask
        attention_mask = attention_mask < 0.5

    return (
        tokens.astype(np.int32),
        labels.astype(np.int32),
        loss_mask.astype(np.int32),
        position_ids.astype(np.int32),
        attention_mask.astype(np.int32)
    )
