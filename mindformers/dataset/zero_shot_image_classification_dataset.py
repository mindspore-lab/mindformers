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
"""Zero Shot Image Classification Dataset."""
from .dataloader import build_dataset_loader
from .transforms import build_transforms
from .base_dataset import BaseDataset
from ..tools import logger
from ..tools.register import MindFormerRegister, MindFormerModuleType
from ..models.build_tokenizer import build_tokenizer

@MindFormerRegister.register(MindFormerModuleType.DATASET)
class ZeroShotImageClassificationDataset(BaseDataset):
    """
    Zero Shot Image Classification Dataset API.
    output image, text, and label columns
    """
    def __new__(cls, dataset_config: dict = None):
        """new method"""
        logger.info("Now Create Zero Shot Image Classification Dataset.")
        cls.init_dataset_config(dataset_config)
        dataset = build_dataset_loader(dataset_config.data_loader)

        transforms = build_transforms(dataset_config.transforms)
        tokenizer = build_tokenizer(dataset_config.tokenizer)
        text_transforms = build_transforms(dataset_config.text_transforms,
                                           default_args={"tokenizer": tokenizer})

        if transforms is not None:
            dataset = dataset.map(
                input_columns="image",
                operations=transforms,
                num_parallel_workers=dataset_config.num_parallel_workers,
                python_multiprocessing=dataset_config.python_multiprocessing
            )

        if text_transforms is not None:
            dataset = dataset.map(
                input_columns="text",
                operations=text_transforms,
                num_parallel_workers=dataset_config.num_parallel_workers,
                python_multiprocessing=dataset_config.python_multiprocessing
            )

        dataset = dataset.batch(dataset_config.batch_size,
                                drop_remainder=dataset_config.drop_remainder,
                                num_parallel_workers=dataset_config.num_parallel_workers)
        dataset = dataset.repeat(dataset_config.repeat)
        return dataset
