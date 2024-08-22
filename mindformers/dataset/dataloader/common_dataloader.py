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
"""Common DataLoader"""
from typing import Optional, List, Union
from mindformers.dataset.handler import build_data_handler
from ...tools.register import MindFormerRegister, MindFormerModuleType
from .base_dataloader import BaseDataLoader


@MindFormerRegister.register(MindFormerModuleType.DATASET_LOADER)
class CommonDataLoader(BaseDataLoader):
    """Common Dataloader"""
    # pylint: disable=W0102
    def __new__(cls,
                num_shards: Optional[int] = None,
                split: Optional[str] = None,
                shuffle: bool = True,
                handler: Optional[dict] = None,
                shard_id: Optional[int] = None,
                dataset_path: Optional[str] = None,
                input_columns: list = ["input_ids", "labels"],
                token: Optional[str] = None,
                data_files: Optional[Union[List[str], str]] = None):
        if dataset_path is None or dataset_path.strip() == "":
            raise ValueError(f"dataset_path is empty.")

        dataset = cls.load_dataset(dataset_path, data_files=data_files, split=split, token=token)

        if handler:  # 离线预处理脚本逻辑
            data_handler = build_data_handler(handler)
            dataset = data_handler.handle(dataset)

        dataset = dataset.to_ms_dataset(columns=input_columns,
                                        num_shards=num_shards,
                                        shard_id=shard_id,
                                        shuffle=shuffle
                                        )

        return dataset
