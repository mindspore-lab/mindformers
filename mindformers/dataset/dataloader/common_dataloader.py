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
"""Common DataLoader"""
import types
from typing import Optional
import numpy as np

import mindspore as ms
from mindspore.dataset import GeneratorDataset
from mindspore import Tensor, ops

from mindformers.version_control import skip_barrier_controller
from mindformers.dataset.handler import build_data_handler
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.utils import get_real_group_size

from .base_dataloader import BaseDataLoader
from .ms_ds_convertor import to_ms_dataset
from .utils import is_dataset_built_on_rank
from .mock_dataloader import BaseMockDataLoader


DATASET_DTYPE_MAP = {
    1: 'int32',
    2: 'float32',
    3: 'int64',
    4: 'float16'
}
PLACEHOLDER = 256
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


@MindFormerRegister.register(MindFormerModuleType.DATASET_LOADER)
class CommonDataLoader(BaseDataLoader):
    """Common Dataloader"""
    _support_parameters = ["path", "name", "data_dir", "data_files", "split", "cache_dir", "features",
                           "download_config", "download_mode", "verification_mode", "ignore_verifications",
                           "keep_in_memory", "save_infos", "revision", "token", "use_auth_token", "task",
                           "streaming", "num_proc", "storage_options", "trust_remote_code"]

    # pylint: disable=W0102
    def __new__(cls,
                shard_id: Optional[int] = None,
                num_shards: Optional[int] = None,
                column_names: list = ["input_ids", "labels"],
                shuffle: bool = False,
                path: Optional[str] = None,
                load_func: str = 'load_dataset',
                handler: Optional[list] = None,
                packing: str = None,
                adaptor_config: dict = None,
                **kwargs):

        mock_data = kwargs.get('mock_data')
        if get_real_group_size() > 1 and mock_data and not is_dataset_built_on_rank():
            logger.info(" > start barrier for all dataset init ... ")
            skip_barrier_controller()  # mock dataset only support parallel mode

            logger.info(" > start receive dataset info from main rank.")
            received_data = (
                Tensor([PLACEHOLDER_ID], dtype=ms.int32),
                Tensor([PLACEHOLDER_ID], dtype=ms.int32),
                Tensor([PLACEHOLDER_ID] * PLACEHOLDER, dtype=ms.int32),
                Tensor([PLACEHOLDER_ID] * PLACEHOLDER, dtype=ms.int32)
            )
            dataset_size, num_columns, data_shapes, data_dtypes = ops.Broadcast(0)(received_data)

            mock_data = dict(
                dataset_size=dataset_size.numpy()[0],
                num_columns=num_columns.numpy()[0],
                data_shapes=parse_data_shapes(data_shapes.numpy()),
                data_dtypes=parse_data_dtypes(data_dtypes.numpy())
            )

            logger.info(f" > received dataset info: \n"
                        f"   size:        {mock_data.get('dataset_size')} \n"
                        f"   num_columns: {mock_data.get('num_columns')} \n"
                        f"   data_shapes: {mock_data.get('data_shapes')} \n"
                        f"   data_dtypes: {mock_data.get('data_dtypes')}")
            mock_dataloader = GeneratorDataset(
                MockCommonDataLoader(**mock_data),
                column_names=column_names,
                num_shards=num_shards,
                shard_id=shard_id)
            return mock_dataloader

        if path is None or path.strip() == "":
            raise ValueError(f"path should not be empty.")

        if "split" not in kwargs:
            kwargs["split"] = "train"

        kwargs = cls._filter_params(kwargs=kwargs)
        dataset = cls.load_dataset(path=path, load_func=load_func, **kwargs)

        if handler:  # data preprocess
            if not isinstance(handler, list):
                raise ValueError(f"handler in config should be set as 'list', but got {type(handler)}.")
            for per_handler in handler:
                data_handler = build_data_handler(per_handler, packing=packing)
                dataset = data_handler.handle(dataset)
        dataset_info = cls._get_dataset_info(dataset, packing=packing, adaptor_config=adaptor_config)

        # set `to_ms_dataset` as dataset class method
        setattr(dataset, "to_ms_dataset", types.MethodType(to_ms_dataset, dataset))

        dataset = dataset.to_ms_dataset(columns=column_names,
                                        num_shards=num_shards,
                                        shard_id=shard_id,
                                        shuffle=shuffle,
                                        packing=packing,
                                        adaptor_config=adaptor_config)

        if get_real_group_size() > 1:  # process dataset in parallel mode
            logger.info(" > start barrier for all dataset init ... ")
            skip_barrier_controller()

            if mock_data:
                # broadcast dataset info
                logger.info(f" > build real dataset completed, broadcast dataset info: \n"
                            f"   size:        {dataset_info.get('size')} \n"
                            f"   num_columns: {dataset_info.get('num_columns')} \n"
                            f"   data_shapes: {dataset_info.get('src_shapes')}")
                dataset_size = Tensor(dataset_info.get('size'), dtype=ms.int32)
                num_columns = Tensor(dataset_info.get('num_columns'), dtype=ms.int32)
                data_shapes = Tensor(dataset_info.get('data_shapes'), dtype=ms.int32)
                data_dtypes = Tensor(dataset_info.get('data_dtypes'), dtype=ms.int32)

                logger.info(" > start broadcast dataset info to other rank.")
                ops.Broadcast(0)((dataset_size, num_columns, data_shapes, data_dtypes))

        return dataset

    @classmethod
    def _filter_params(cls, kwargs):
        result = {}
        for key in kwargs:
            if key not in cls._support_parameters:
                logger.info(f"dataset load_dataset not support params: {key}")
                continue
            result[key] = kwargs[key]
        return result

    @classmethod
    def _get_dataset_info(cls, dataset, packing=None, adaptor_config=None):
        """Get processed dataset info."""
        # iter dataset
        sample = next(iter(dataset))
        sample_columns = list(sample.keys())

        if packing:
            logger.info("collect packed dataset info.")
            # cols: "input_ids", "labels", "loss_mask", "position_ids", "attention_mask"
            num_columns = 5
            data_dtype_map = {v: k for k, v in DATASET_DTYPE_MAP.items()}
            sample_dtypes = [data_dtype_map['int32']] * num_columns
            seq_length = len(sample['input_ids'])
            src_shapes = [[seq_length]] * (num_columns - 1)
            if adaptor_config.get('compress_mask'):
                src_shapes.append([adaptor_config.get('eod_pad_length', 128)])
            else:
                src_shapes.append([1, seq_length, seq_length])

            sample_shapes = []
            for data_shape in src_shapes:
                sample_shapes.extend(data_shape)
                sample_shapes.append(-1)

        else:
            logger.info("collect source dataset info.")
            # collect data dtype from each column
            sample_dtypes = []
            for idx, data_dtype in enumerate(dataset.features.values()):
                data_dtype = str(data_dtype).lower()
                dtype_id = [k for k, v in DATASET_DTYPE_MAP.items() if v in data_dtype]
                if not dtype_id:
                    raise ValueError(
                        f"Get column {sample_columns[idx]} dtype: {data_dtype} in dataset, "
                        f"but `int` or `float` should in dtype. Please check data dtype in each column, "
                        f"or remove unexpected columns first."
                    )
                sample_dtypes.append(dtype_id[0])

            sample_shapes, src_shapes = [], []
            for data in sample.values():
                cur_shape = list(np.array(data).shape)
                src_shapes.append(cur_shape)
                sample_shapes.extend(cur_shape)
                sample_shapes.append(-1)  # split each data shape with -1
            num_columns = len(sample_columns)

        # pad dataset info to placeholder length
        sample_dtypes.extend([PLACEHOLDER_ID] * (PLACEHOLDER - len(sample_dtypes)))
        sample_shapes.extend([PLACEHOLDER_ID] * (PLACEHOLDER - len(sample_shapes)))

        dataset_info = dict(
            src_shapes=src_shapes,
            size=len(dataset),
            num_columns=num_columns,
            data_shapes=sample_shapes,
            data_dtypes=sample_dtypes,
        )
        return dataset_info


class MockCommonDataLoader(BaseMockDataLoader):
    """The mock CommonDataLoader

    Args:
        dataset_size (int): The mock dataset size.
        num_columns (int): The number of mock dataset columns.
        data_shapes (list[list]): Data shapes of each column.
        data_dtypes (list[str]): Data dtypes of each column.
    """

    def __init__(self, dataset_size, num_columns, data_shapes, data_dtypes):
        mock_columns = [f"col{col_id}" for col_id in range(num_columns)]
        super().__init__(mock_columns, data_shapes, data_dtypes, dataset_size)
