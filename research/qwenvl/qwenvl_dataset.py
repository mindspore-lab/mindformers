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
"""Dataset for QwenVL"""

from typing import Optional, Union, Callable

from mindformers.version_control import get_dataset_map
from mindformers.dataset.dataloader import build_dataset_loader
from mindformers.dataset.transforms import build_transforms
from mindformers.dataset.sampler import build_sampler
from mindformers.dataset.base_dataset import BaseDataset
from mindformers.tools import logger
from mindformers.models.build_tokenizer import build_tokenizer
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.DATASET)
class QwenVLDataset(BaseDataset):
    """Dataset for QwenVL"""

    def __new__(cls,
                dataset_config: Optional[dict] = None,
                transforms: Union[dict, list] = None,
                text_transforms: Union[dict, list] = None,
                tokenizer: Union[dict, Callable] = None,
                sampler: Union[dict, Callable] = None):
        """new method"""
        logger.info("Now Create QwenVL Pretrain Dataset.")
        dataset_config = cls.check_dataset_config(dataset_config, locals())
        cls.init_dataset_config(dataset_config)
        rank_id, device_num = cls._generate_shard_info()
        if isinstance(dataset_config.data_loader, dict):
            dataset = build_dataset_loader(dataset_config.data_loader,
                                           default_args={'num_shards': device_num, 'shard_id': rank_id})
        else:
            dataset = dataset_config.data_loader

        if isinstance(dataset_config.tokenizer, dict):
            tokenizer = build_tokenizer(dataset_config.tokenizer)
        else:
            tokenizer = dataset_config.tokenizer

        if (isinstance(dataset_config.transforms, list) and isinstance(dataset_config.transforms[0], dict)) \
                or isinstance(dataset_config.transforms, dict):
            transforms = build_transforms(dataset_config.transforms)
        else:
            transforms = dataset_config.transforms

        if (isinstance(dataset_config.text_transforms, list) and isinstance(dataset_config.text_transforms[0], dict)) \
                or isinstance(dataset_config.text_transforms, dict):
            text_transforms = build_transforms(dataset_config.text_transforms, default_args={"tokenizer": tokenizer})
        else:
            text_transforms = dataset_config.text_transforms

        if isinstance(dataset_config.sampler, dict):
            sampler = build_sampler(dataset_config.sampler)
        else:
            sampler = dataset_config.sampler

        if sampler is not None:
            dataset = dataset.use_sampler(sampler)

        if transforms is not None:
            dataset = get_dataset_map(dataset, transforms,
                                      input_columns="image",
                                      num_parallel_workers=dataset_config.num_parallel_workers,
                                      python_multiprocessing=dataset_config.python_multiprocessing)

        if text_transforms is not None:
            output_columns = dataset_config.get('output_columns', ["text", "img_start_pos"])
            dataset = get_dataset_map(dataset, text_transforms,
                                      input_columns=["text"],
                                      output_columns=output_columns,
                                      num_parallel_workers=dataset_config.num_parallel_workers,
                                      python_multiprocessing=dataset_config.python_multiprocessing)

        column_order = dataset_config.get('column_order')
        if column_order is not None:
            dataset = dataset.project(columns=column_order)

        dataset = dataset.batch(dataset_config.batch_size,
                                drop_remainder=dataset_config.drop_remainder,
                                num_parallel_workers=dataset_config.num_parallel_workers)
        dataset = dataset.repeat(dataset_config.repeat)
        return dataset
