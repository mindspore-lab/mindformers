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

from mindformers.dataset.handler import build_data_handler
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

from .base_dataloader import BaseDataLoader
from .ms_ds_convertor import to_ms_dataset


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

        # set `to_ms_dataset` as dataset class method
        setattr(dataset, "to_ms_dataset", types.MethodType(to_ms_dataset, dataset))

        dataset = dataset.to_ms_dataset(columns=column_names,
                                        num_shards=num_shards,
                                        shard_id=shard_id,
                                        shuffle=shuffle,
                                        packing=packing,
                                        adaptor_config=adaptor_config)

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
