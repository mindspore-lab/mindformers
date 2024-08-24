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
import importlib

from typing import Optional
from packaging import version
from datasets import config
from mindformers.dataset.handler import build_data_handler
from mindformers.tools.logger import logger
from ...tools.register import MindFormerRegister, MindFormerModuleType
from .base_dataloader import BaseDataLoader


@MindFormerRegister.register(MindFormerModuleType.DATASET_LOADER)
class CommonDataLoader(BaseDataLoader):
    """Common Dataloader"""
    # pylint: disable=W0102
    def __new__(cls,
                num_shards: Optional[int] = None,
                shuffle: bool = True,
                handler: Optional[dict] = None,
                shard_id: Optional[int] = None,
                path: Optional[str] = None,
                input_columns: list = ["input_ids", "labels"],
                **kwargs):
        if path is None or path.strip() == "":
            raise ValueError(f"dataset_path is empty.")

        if "split" not in kwargs:
            kwargs["split"] = "train"

        dataset = cls.load_dataset(path, **kwargs)

        if handler:  # data preprocess
            data_handler = build_data_handler(handler)
            dataset = data_handler.handle(dataset)

        dataset = dataset.to_ms_dataset(columns=input_columns,
                                        num_shards=num_shards,
                                        shard_id=shard_id,
                                        shuffle=shuffle
                                        )

        return dataset

def ms_adaptor_execution():
    """ms adaptor execution"""
    try:
        ms_version = version.parse(importlib.metadata.version("mindspore"))
        config.MS_VERSION = ms_version
        logger.info(f"Mindspore version {ms_version} available.")

        from datasets import Dataset
        from .ms_ds_convertor import to_ms_dataset

        setattr(Dataset, "to_ms_dataset", to_ms_dataset)
    except importlib.metadata.PackageNotFoundError:
        logger.error(f"to_ms_dataset adaptor failed.")


ms_adaptor_execution()
