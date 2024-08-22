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
"""Base DataLoader"""
import os
import datasets

from mindformers.tools.logger import logger


class BaseDataLoader:
    """Base Dataloader"""
    @classmethod
    def load_dataset(cls, dataset_path: str, data_files=None, split=None, token=None):
        """load dataset"""
        if dataset_path.startswith("local:"):
            return cls._load_local_dataset(dataset_path[6:], data_files, split, token)
        if dataset_path.startswith("hf:"):
            return cls._load_hugging_face_dataset(dataset_path[3:], data_files, split, token)
        if dataset_path.startswith("om:"):
            return cls._load_open_mind_dataset(dataset_path[3:], data_files, split, token)
        return cls._load_open_mind_dataset(dataset_path, data_files, split, token)

    @classmethod
    def _load_open_mind_dataset(cls, dataset_path, data_files=None, split=None, token=None):
        """open mind dataset"""
        os.environ["USE_OM"] = "AUTO"
        # pylint: disable=W0611
        import openmind_datasets
        logger.info(f"_load_open_mind_dataset : {dataset_path}, {data_files}, {split}")
        dataset = datasets.load_dataset(dataset_path,
                                        data_files=data_files,
                                        split=split,
                                        token=token,
                                        )
        return dataset

    @classmethod
    def _load_hugging_face_dataset(cls, dataset_path, data_files=None, split=None, token=None):
        """huggingFace dataset"""
        logger.info(f"_load_hugging_face_dataset : {dataset_path}, {data_files}, {split}")
        os.environ["USE_OM"] = "OFF"
        # pylint: disable=W0611
        import openmind_datasets
        dataset = datasets.load_dataset(dataset_path,
                                        data_files=data_files,
                                        split=split,
                                        token=token,
                                        )
        return dataset

    @classmethod
    def _load_local_dataset(cls, path, data_files=None, split=None, token=None):
        """local dataset"""
        logger.info(f"_load_local_dataset : {path}, {data_files}, {split}")
        os.environ["USE_OM"] = "OFF"
        # pylint: disable=W0611
        import openmind_datasets
        dataset = datasets.load_dataset(path,
                                        data_dir=data_dir,
                                        data_files=data_files,
                                        split=split,
                                        token=token,
                                        )
        return dataset
