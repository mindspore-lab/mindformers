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
"""Token classification Dataset."""
import os

import mindspore
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.logger import logger
from mindformers.tools.utils import is_version_ge

from .dataloader import build_dataset_loader
from ..models.build_tokenizer import build_tokenizer
from .transforms import build_transforms
from .sampler import build_sampler
from .base_dataset import BaseDataset


@MindFormerRegister.register(MindFormerModuleType.DATASET)
class TokenClassificationDataset(BaseDataset):
    """
    Token classification Dataset.

    Examples:
        >>> from mindformers.tools.register import MindFormerConfig
        >>> from mindformers import MindFormerBook
        >>> from mindformers.dataset import TokenClassificationDataset
        >>> from mindformers.dataset import build_dataset, check_dataset_config
        >>> config_dict_list = MindFormerBook.get_trainer_support_task_list()
        >>> config_path = config_dict_list['token_classification']['tokcls_bert_base_chinese']
        >>> # Initialize a MindFormerConfig instance with a specific config file of yaml.
        >>> config = MindFormerConfig(config_path)
        >>> config.train_dataset.data_loader.dataset_dir = "The required task dataset path"
            Note:
                The detailed data setting could refer to
                https://gitee.com/mindspore/mindformers/blob/r0.3/docs/task_cards/token_classification.md
        >>> check_dataset_config(config)
        >>> # 1) use config dict to build dataset
        >>> dataset_from_config = build_dataset(config.train_dataset_task)
        >>> # 2) use class name to build dataset
        >>> dataset_from_name = build_dataset(class_name='TokenClassificationDataset',
        ...                                   dataset_config=config.train_dataset_task.dataset_config)
        >>> # 3) use class to build dataset
        >>> dataset_from_class = TokenClassificationDataset(config.train_dataset_task.dataset_config)
    """
    def __new__(cls, dataset_config: dict = None):
        """new method"""
        logger.info("Now Create Token classification Dataset.")
        cls.init_dataset_config(dataset_config)
        rank_id = int(os.getenv("RANK_ID", "0"))
        device_num = int(os.getenv("RANK_SIZE", "1"))

        dataset = build_dataset_loader(
            dataset_config.data_loader, default_args={'num_shards': device_num, 'shard_id': rank_id})

        tokenizer = build_tokenizer(dataset_config.tokenizer)

        text_transforms = build_transforms(dataset_config.text_transforms,
                                           default_args={"tokenizer": tokenizer})

        label_transforms = build_transforms(dataset_config.label_transforms)

        sampler = build_sampler(dataset_config.sampler)

        if sampler is not None:
            dataset = dataset.use_sampler(sampler)

        if text_transforms is not None:
            if is_version_ge(mindspore.__version__, '2.0.0'):
                dataset = dataset.map(
                    input_columns=dataset_config.input_columns,
                    operations=text_transforms,
                    output_columns=dataset_config.output_columns,
                    num_parallel_workers=dataset_config.num_parallel_workers,
                    python_multiprocessing=dataset_config.python_multiprocessing
                )
            else:
                dataset = dataset.map(
                    input_columns=dataset_config.input_columns,
                    operations=text_transforms,
                    column_order=dataset_config.output_columns,
                    output_columns=dataset_config.output_columns,
                    num_parallel_workers=dataset_config.num_parallel_workers,
                    python_multiprocessing=dataset_config.python_multiprocessing
                )

        if label_transforms is not None:
            dataset = dataset.map(
                input_columns=dataset_config.input_columns[1],
                operations=label_transforms,
                num_parallel_workers=dataset_config.num_parallel_workers,
                python_multiprocessing=dataset_config.python_multiprocessing
            )

        dataset = dataset.batch(dataset_config.batch_size,
                                drop_remainder=dataset_config.drop_remainder,
                                num_parallel_workers=dataset_config.num_parallel_workers)
        dataset = dataset.repeat(dataset_config.repeat)

        return dataset
