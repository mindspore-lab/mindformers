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
"""T5 Dataset."""
import os
import copy
import numpy as np


import mindspore
import mindspore.common.dtype as mstype
import mindspore.dataset.transforms.c_transforms as C

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.logger import logger
from mindformers.tools.utils import is_version_ge
from .dataloader import build_dataset_loader
from .base_dataset import BaseDataset
from ..auto_class import AutoTokenizer

__all__ = ['TranslationDataset']

@MindFormerRegister.register(MindFormerModuleType.DATASET)
class TranslationDataset(BaseDataset):
    """Translation dataset.

    Examples:
        >>> from mindformers.tools.register import MindFormerConfig
        >>> from mindformers import MindFormerBook
        >>> from mindformers.dataset import TranslationDataset
        >>> from mindformers.dataset import build_dataset, check_dataset_config
        >>> config_dict_list = MindFormerBook.get_trainer_support_task_list()
        >>> config_path = config_dict_list['translation']['t5_small']
        >>> # Initialize a MindFormerConfig instance with a specific config file of yaml.
        >>> config = MindFormerConfig(config_path)
        >>> config.train_dataset.data_loader.dataset_dir = "The required task dataset path"
            Note:
                The detailed data setting could refer to
                https://gitee.com/mindspore/mindformers/blob/r0.3/docs/model_cards/t5.md
        >>> check_dataset_config(config)
        >>> # 1) use config dict to build dataset
        >>> dataset_from_config = build_dataset(config.train_dataset_task)
        >>> # 2) use class name to build dataset
        >>> dataset_from_name = build_dataset(class_name='TranslationDataset',
        ...                                   dataset_config=config.train_dataset_task.dataset_config)
        >>> # 3) use class to build dataset
        >>> dataset_from_class = TranslationDataset(config.train_dataset_task.dataset_config)
    """
    def __new__(cls, dataset_config: dict = None):
        logger.info("Now Create Translation Dataset.")
        cls.init_dataset_config(dataset_config)
        if dataset_config.data_loader.type != 'MindDataset':
            dataset = cls._process_raw_text_data(dataset_config)
        else:
            dataset = cls._process_mindrecord_data(dataset_config)

        dataset = dataset.batch(dataset_config.batch_size,
                                drop_remainder=dataset_config.drop_remainder,
                                num_parallel_workers=dataset_config.num_parallel_workers)
        dataset = dataset.repeat(dataset_config.repeat)
        type_cast_op = C.TypeCast(mstype.int32)
        for input_arg in dataset_config.input_columns:
            dataset = dataset.map(operations=type_cast_op, input_columns=input_arg)
        return dataset

    @classmethod
    def _tokenizer_map(cls, dataset, tokenizer_config):
        """Maps the tokenizer on the source and the output"""
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.type)
        prefix = tokenizer_config.prefix
        src_max_length = tokenizer_config.src_max_length
        tgt_max_length = tokenizer_config.tgt_max_length

        logger.info("Start tokenize on the dataset using tokenizer: %s", tokenizer_config)
        def pad_max_function(src, tgt):
            src = src.tolist()
            if isinstance(src, bytes):
                src = src.decode()
            output = tokenizer(prefix + src, padding='max_length', max_length=src_max_length, truncation=True)

            tgt = tgt.tolist()
            if isinstance(tgt, bytes):
                tgt = tgt.decode()
            tgt_output = tokenizer(tgt, padding='max_length', max_length=tgt_max_length, truncation=True)

            input_ids = np.array(output['input_ids'], np.int32)
            attention_mask = np.array(output['attention_mask'], np.float32)
            labels = np.array(tgt_output['input_ids'], np.int32)
            return input_ids, attention_mask, labels

        if is_version_ge(mindspore.__version__, "2.0.0"):
            dataset = dataset.map(pad_max_function,
                                  input_columns=['source', 'target'],
                                  output_columns=['input_ids', 'attention_mask', 'labels'])
            dataset = dataset.project(columns=['input_ids', 'attention_mask', 'labels'])

        else:
            dataset = dataset.map(pad_max_function,
                                  input_columns=['source', 'target'],
                                  output_columns=['input_ids', 'attention_mask', 'labels'],
                                  column_order=['input_ids', 'attention_mask', 'labels'])
        return dataset

    @classmethod
    def _process_raw_text_data(cls, dataset_config):
        """Process the text data"""
        rank_id = int(os.getenv("RANK_ID", "0"))
        device_num = int(os.getenv("RANK_SIZE", "1"))
        dataset_dir = dataset_config.data_loader.pop("dataset_dir")
        dataset = build_dataset_loader(
            dataset_config.data_loader, default_args={'dataset_dir': dataset_dir,
                                                      'num_shards': device_num, 'shard_id': rank_id})

        dataset = cls._tokenizer_map(dataset, dataset_config.tokenizer)
        return dataset

    @classmethod
    def _process_mindrecord_data(cls, dataset_config):
        """Process the mindrecord data"""
        rank_id = int(os.getenv("RANK_ID", "0"))
        device_num = int(os.getenv("RANK_SIZE", "1"))
        dataset_config = copy.deepcopy(dataset_config)

        dataset_files = []
        if dataset_config.data_loader.dataset_dir:
            data_dir = dataset_config.data_loader.pop("dataset_dir")
            if os.path.isdir(data_dir):
                for r, _, f in os.walk(data_dir):
                    for file in f:
                        if file.endswith(".mindrecord"):
                            dataset_files.append(os.path.join(r, file))
            else:
                if data_dir.endswith(".mindrecord"):
                    dataset_files = data_dir
        elif dataset_config.data_loader.dataset_files:
            dataset_files = dataset_config.data_loader.dataset_files
            if isinstance(dataset_files, (list, tuple)):
                dataset_files = list(dataset_files)
        else:
            raise ValueError(f"data_loader must contain dataset_dir or dataset_files,"
                             f"but get {dataset_config.data_loader}.")

        logger.info("Using args %s to instance the dataset.", dataset_config.data_loader)
        dataset = build_dataset_loader(
            dataset_config.data_loader, default_args={'dataset_files': dataset_files,
                                                      'num_shards': device_num, 'shard_id': rank_id,
                                                      'columns_list': dataset_config.input_columns})
        return dataset
