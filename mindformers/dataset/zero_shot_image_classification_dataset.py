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
import os
from mindformers.version_control import get_dataset_map
from .dataloader import build_dataset_loader
from .transforms import build_transforms
from .base_dataset import BaseDataset
from ..tools import logger
from ..tools.register import MindFormerRegister, MindFormerModuleType
from ..models.build_tokenizer import build_tokenizer

@MindFormerRegister.register(MindFormerModuleType.DATASET)
class ZeroShotImageClassificationDataset(BaseDataset):
    r"""
    Zero Shot Image Classification Dataset API.
    output image, text, and label columns

    Args:
        dataset_config (dict): Config for dataset.

    Returns:
        A dataset for ZeroShotImageClassificationDataset.

    Examples:
        >>> from mindformers.tools.register import MindFormerConfig
        >>> from mindformers import MindFormerBook
        >>> from mindformers.dataset import ZeroShotImageClassificationDataset
        >>> from mindformers.dataset import build_dataset, check_dataset_config
        >>> config_dict_list = MindFormerBook.get_trainer_support_task_list()
        >>> config_path = config_dict_list['zero_shot_image_classification']['clip_vit_b_32']
        >>> # Initialize a MindFormerConfig instance with a specific config file of yaml.
        >>> config = MindFormerConfig(config_path)
        >>> config.eval_dataset.data_loader.dataset_dir = "The required task dataset path"
        >>> # Note:
        >>> #     The detailed data setting could refer to
        >>> #     https://gitee.com/mindspore/mindformers/blob/r0.8/docs/model_cards/clip.md
        >>> check_dataset_config(config)
        >>> # use class to build dataset
        >>> dataset_from_class = ZeroShotImageClassificationDataset(config.eval_dataset_task.dataset_config)
    """
    def __new__(cls, dataset_config: dict = None):
        """New method"""
        logger.info("Now Create Zero Shot Image Classification Dataset.")
        cls.init_dataset_config(dataset_config)
        rank_id = int(os.getenv("RANK_ID", "0"))
        device_num = int(os.getenv("RANK_SIZE", "1"))

        dataset = build_dataset_loader(
            dataset_config.data_loader, default_args={'num_shards': device_num, 'shard_id': rank_id})

        transforms = build_transforms(dataset_config.transforms)
        tokenizer = build_tokenizer(dataset_config.tokenizer)
        text_transforms = build_transforms(dataset_config.text_transforms,
                                           default_args={"tokenizer": tokenizer})

        if transforms is not None:
            dataset = get_dataset_map(dataset,
                                      input_columns="image",
                                      operations=transforms,
                                      num_parallel_workers=dataset_config.num_parallel_workers,
                                      python_multiprocessing=dataset_config.python_multiprocessing)

        if text_transforms is not None:
            dataset = get_dataset_map(dataset,
                                      input_columns="text",
                                      operations=text_transforms,
                                      num_parallel_workers=dataset_config.num_parallel_workers,
                                      python_multiprocessing=dataset_config.python_multiprocessing)

        dataset = dataset.batch(dataset_config.batch_size,
                                drop_remainder=dataset_config.drop_remainder,
                                num_parallel_workers=dataset_config.num_parallel_workers)
        dataset = dataset.repeat(dataset_config.repeat)
        return dataset
