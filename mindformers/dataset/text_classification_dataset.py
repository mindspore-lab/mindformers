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
"""Text Classification Dataset."""
import os
import copy
import mindspore.common.dtype as mstype
import mindspore.dataset.transforms.c_transforms as C
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.logger import logger
from .dataloader import build_dataset_loader
from .base_dataset import BaseDataset


@MindFormerRegister.register(MindFormerModuleType.DATASET)
class TextClassificationDataset(BaseDataset):
    """
    Text Classification dataset.

    Examples:
        >>> from mindformers.tools.register import MindFormerConfig
        >>> from mindformers import MindFormerBook
        >>> from mindformers.dataset import TextClassificationDataset
        >>> from mindformers.dataset import build_dataset, check_dataset_config
        >>> config_dict_list = MindFormerBook.get_trainer_support_task_list()
        >>> config_path = config_dict_list['text_classification']['txtcls_bert_base_uncased']
        >>> # Initialize a MindFormerConfig instance with a specific config file of yaml.
        >>> config = MindFormerConfig(config_path)
        >>> config.train_dataset.data_loader.dataset_dir = "The required task dataset path"
            Note:
                The detailed data setting could refer to
                https://gitee.com/mindspore/mindformers/blob/r0.3/docs/task_cards/text_classification.md
        >>> check_dataset_config(config)
        >>> # 1) use config dict to build dataset
        >>> dataset_from_config = build_dataset(config.train_dataset_task)
        >>> # 2) use class name to build dataset
        >>> dataset_from_name = build_dataset(class_name='TextClassificationDataset',
        ...                                   dataset_config=config.train_dataset_task.dataset_config)
        >>> # 3) use class to build dataset
        >>> dataset_from_class = TextClassificationDataset(config.train_dataset_task.dataset_config)
    """
    def __new__(cls, dataset_config: dict = None):
        logger.info("Now Create Text Classification Dataset.")
        cls.init_dataset_config(dataset_config)
        rank_id = int(os.getenv("RANK_ID", "0"))
        device_num = int(os.getenv("RANK_SIZE", "1"))
        dataset_config = copy.deepcopy(dataset_config)
        if "data_files" not in dataset_config.data_loader \
            and dataset_config.data_loader.dataset_dir:
            dataset_files = []
            data_dir = dataset_config.data_loader.dataset_dir
            if os.path.isdir(data_dir):
                for r, _, f in os.walk(data_dir):
                    for file in f:
                        if file.endswith(".tfrecord"):
                            dataset_files.append(os.path.join(r, file))
            else:
                if data_dir.endswith(".tfrecord"):
                    dataset_files.append(data_dir)
        else:
            dataset_files = list(dataset_config.data_loader.dataset_files)
        dataset_config.data_loader.pop("dataset_dir")
        dataset = build_dataset_loader(
            dataset_config.data_loader, default_args={'dataset_files': dataset_files,
                                                      'num_shards': device_num, 'shard_id': rank_id,
                                                      'shard_equal_rows': True})
        dataset = dataset.batch(dataset_config.batch_size,
                                drop_remainder=dataset_config.drop_remainder,
                                output_columns=dataset_config.input_columns,
                                num_parallel_workers=dataset_config.num_parallel_workers)
        dataset = dataset.project(columns=dataset_config.input_columns)
        dataset = dataset.repeat(dataset_config.repeat)
        type_cast_op = C.TypeCast(mstype.int32)
        for input_arg in dataset_config.input_columns:
            dataset = dataset.map(operations=type_cast_op, input_columns=input_arg)
        return dataset
