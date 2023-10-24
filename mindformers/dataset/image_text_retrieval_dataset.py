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
"""Image-text Retrieval Dataset."""
import os
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
from mindspore.dataset.transforms import TypeCast
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.logger import logger
from mindformers.version_control import get_dataset_map
from .base_dataset import BaseDataset
from .transforms import build_transforms


@MindFormerRegister.register(MindFormerModuleType.DATASET)
class ImageToTextRetrievalDataset(BaseDataset):
    """
    Image-text Retrieval Dataset API.

    Args:
        dataset_config (dict): Config for dataset.

    Returns:
        A dataset for ImageToTextRetrievalDataset.

    Examples:
        >>> from mindformers.tools.register import MindFormerConfig
        >>> from mindformers import MindFormerBook
        >>> from mindformers.dataset import ImageToTextRetrievalDataset
        >>> from mindformers.dataset import build_dataset, check_dataset_config
        >>> config_dict_list = MindFormerBook.get_trainer_support_task_list()
        >>> config_path = config_dict_list['image_to_text_retrieval']['blip2_stage1_vit_g']
        >>> # Initialize a MindFormerConfig instance with a specific config file of yaml.
        >>> config = MindFormerConfig(config_path)
        >>> config.train_dataset.data_loader.dataset_dir = "The required task dataset path"
        >>> # Note:
        >>> #     The detailed data setting could refer to
        >>> #     https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/blip2.md
        >>> check_dataset_config(config)
        >>> # use class to build dataset
        >>> dataset_from_class = ImageToTextRetrievalDataset(config.train_dataset_task.dataset_config)
    """
    def __new__(cls, dataset_config: dict = None):
        logger.info("Now Create Image-text Retrieval Dataset.")
        cls.init_dataset_config(dataset_config)
        rank_id = int(os.getenv("RANK_ID", "0"))
        device_num = int(os.getenv("RANK_SIZE", "1"))
        dataset = ds.MindDataset(dataset_config.data_loader.dataset_dir,
                                 shuffle=dataset_config.data_loader.shuffle,
                                 num_shards=device_num,
                                 shard_id=rank_id)
        transforms = build_transforms(dataset_config.transforms)
        if transforms is not None:
            dataset = get_dataset_map(dataset, transforms,
                                      num_parallel_workers=dataset_config.num_parallel_workers,
                                      python_multiprocessing=dataset_config.python_multiprocessing,
                                      input_columns="image", output_columns=['image'])

        type_cast_op = TypeCast(mstype.float32)
        dataset = get_dataset_map(dataset, type_cast_op,
                                  input_columns="image",
                                  output_columns=['image'])

        dataset = dataset.project(["image", "token"])
        dataset = dataset.batch(dataset_config.batch_size, drop_remainder=dataset_config.drop_remainder)
        return dataset
