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
"""Causal Image Modeling Dataset."""
import os
import copy
import re
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.dataset.transforms.c_transforms as C
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.logger import logger
from mindformers.version_control import get_dataset_map
from mindformers.models.build_tokenizer import build_tokenizer
from .dataloader import build_dataset_loader
from .base_dataset import BaseDataset


def get_input_data_batch_slice_map(chosen_input_ids, chosen_attention_mask,
                                   rejected_input_ids, rejected_attention_mask,
                                   position_id, loss_mask, end_ind,
                                   rank: int = 0, dis: int = 1, pad_id: int = 2):
    """
    Generate position_id and attention_mask according to input_ids considering eos reset
    """
    rank = int(rank)
    chosen_input_ids = chosen_input_ids[rank*dis: (rank + 1)*dis]
    rejected_input_ids = rejected_input_ids[rank*dis: (rank + 1)*dis]
    chosen_attention_mask = chosen_attention_mask[rank*dis: (rank + 1)*dis]
    rejected_attention_mask = rejected_attention_mask[rank*dis: (rank + 1)*dis]
    position_id = position_id[rank*dis: (rank + 1)*dis]
    loss_mask = loss_mask[rank*dis: (rank + 1)*dis]

    input_ids = np.concatenate((chosen_input_ids, rejected_input_ids))
    attention_mask = np.concatenate((chosen_attention_mask, rejected_attention_mask))
    position_id = np.concatenate((position_id, position_id))
    end_ind = np.sum(input_ids != pad_id, axis=1)
    return input_ids, position_id, attention_mask, loss_mask, end_ind

@MindFormerRegister.register(MindFormerModuleType.DATASET)
class RewardModelDataset(BaseDataset):
    """Causal Language Model pretrain dataset.

    Examples:
        >>> from mindformers.tools.register import MindFormerConfig
        >>> from mindformers import MindFormerBook
        >>> from mindformers.dataset import RewardModelDataset
        >>> from mindformers.dataset import build_dataset, check_dataset_config
        >>> config_dict_list = MindFormerBook.get_trainer_support_task_list()
        >>> config_path = config_dict_list['text_generation']['gpt2']
        >>> # Initialize a MindFormerConfig instance with a specific config file of yaml.
        >>> config = MindFormerConfig(config_path)
        >>> config.train_dataset.data_loader.dataset_dir = "The required task dataset path"
            Note:
                The detailed data setting could refer to
                https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/gpt2.md
        >>> check_dataset_config(config)
        >>> # 1) use config dict to build dataset
        >>> dataset_from_config = build_dataset(config.train_dataset_task)
        >>> # 2) use class name to build dataset
        >>> dataset_from_name = build_dataset(class_name='RewardModelDataset',
        ...                                   dataset_config=config.train_dataset_task.dataset_config)
        >>> # 3) use class to build dataset
        >>> dataset_from_class = RewardModelDataset(config.train_dataset_task.dataset_config)
    """
    def __new__(cls, dataset_config: dict = None):
        logger.info("Now Create Reward Model Dataset.")
        print("dataset_config", dataset_config)
        rank_id = int(os.getenv("RANK_ID", "0"))
        device_num = int(os.getenv("RANK_SIZE", "1"))
        dataset_config = copy.deepcopy(dataset_config)
        cls.init_dataset_config(dataset_config)
        logger.info("Now Create Reward Model Dataset1.")
        rank_id, device_num = cls._check_device_rank_for_parallel(rank_id, device_num)
        dataset_config.rank_id = rank_id
        dataset_config.device_num = device_num
        if dataset_config.data_loader.type != "MindDataset" and \
                dataset_config.data_loader.type != "TFRecordDataset":
            dataset = cls._process_raw_text_data(dataset_config)
        else:
            dataset = cls._process_mindrecord_data(dataset_config)
        logger.info("Now Create Reward Model Dataset1.5")
        # type_cast_op = C.TypeCast(mstype.int32)
        if cls._is_semi_full_batch() or cls._is_data_parallel():
            rank_id = 0
            dis = dataset_config.batch_size
        else:
            # Each card slice a small batch from the full batch
            dis = dataset_config.batch_size // device_num
            if dataset_config.batch_size % device_num != 0:
                raise ValueError(
                    f"batch size {dataset_config.batch_size} should be a multiple of device number {device_num}."
                    " You should change the args: per_batch_size.")
        logger.info("Now Create Reward Model Dataset1.6")
        dataset = dataset.batch(dataset_config.batch_size,
                                drop_remainder=dataset_config.drop_remainder,
                                output_columns=dataset_config.input_columns)
        pad_id = dataset_config.pad_token_id if dataset_config.pad_token_id is not None else 2
        logger.info("Now Create Reward Model Dataset2.")
        map_func = (lambda chosen_input_ids, chosen_attention_mask,
                           rejected_input_ids, rejected_attention_mask,
                           position_id, loss_mask, end_ind: \
                        get_input_data_batch_slice_map(chosen_input_ids, chosen_attention_mask,
                                                       rejected_input_ids, rejected_attention_mask,
                                                       position_id, loss_mask, end_ind, rank_id, dis=dis,
                                                       pad_id=pad_id))

        dataset = get_dataset_map(dataset, map_func,
                                  input_columns=dataset_config.input_columns,
                                  output_columns=dataset_config.output_columns)
        dataset = dataset.project(columns=dataset_config.output_columns)

        type_cast_op = C.TypeCast(mstype.int32)
        type_cast_op_float = C.TypeCast(mstype.float16)
        dataset = get_dataset_map(dataset, input_columns="input_ids", operations=type_cast_op)
        dataset = get_dataset_map(dataset, input_columns="position_id", operations=type_cast_op)
        dataset = get_dataset_map(dataset, input_columns="attention_mask", operations=type_cast_op_float)
        dataset = get_dataset_map(dataset, input_columns="loss_mask", operations=type_cast_op_float)
        dataset = get_dataset_map(dataset, input_columns="end_ind", operations=type_cast_op)
        dataset = dataset.repeat(dataset_config.repeat)
        return dataset

    @classmethod
    def _prepare_for_model(cls, dataset, dataset_config):
        """Preprocess data for gpt2 model"""
        tokenizer_config = dataset_config.tokenizer
        tokenizer = build_tokenizer(tokenizer_config)
        max_length = tokenizer_config.max_length

        def map_func(input_data):
            input_data = input_data.tolist()
            input_ids = tokenizer(input_data, padding='max_length', max_length=max_length, truncation=True,
                                  add_special_tokens=False)
            return input_ids.get('input_ids')

        dataset = get_dataset_map(dataset, map_func,
                                  input_columns=dataset_config.input_columns,
                                  output_columns=dataset_config.input_columns)
        return dataset

    @classmethod
    def _process_raw_text_data(cls, dataset_config):
        """Process the text data"""
        dataset_dir = dataset_config.data_loader.pop("dataset_dir")
        dataset = build_dataset_loader(
            dataset_config.data_loader, default_args={'dataset_dir': dataset_dir,
                                                      'num_shards': dataset_config.device_num,
                                                      'shard_id': dataset_config.rank_id})

        dataset = cls._prepare_for_model(dataset, dataset_config)
        return dataset

    @classmethod
    def _process_mindrecord_data(cls, dataset_config):
        """Process the mindrecord data"""
        dataset_files = []
        mind_compile = re.compile("mindrecord0*$")
        if dataset_config.data_loader.dataset_dir:
            data_dir = dataset_config.data_loader.pop("dataset_dir")
            if os.path.isdir(data_dir):
                for r, _, f in os.walk(data_dir):
                    for file in f:
                        if re.findall(mind_compile, file) or file.endswith(".tfrecord"):
                            dataset_files.append(os.path.join(r, file))
                dataset_files.sort()
            else:
                if re.findall(mind_compile, data_dir) or data_dir.endswith(".tfrecord"):
                    dataset_files = data_dir
        elif dataset_config.data_loader.dataset_files:
            dataset_files = dataset_config.data_loader.dataset_files
            if isinstance(dataset_files, (list, tuple)):
                dataset_files = list(dataset_files)
        else:
            raise ValueError(f"data_loader must contain dataset_dir or dataset_files,"
                             f"but get {dataset_config.data_loader}.")

        dataset = build_dataset_loader(
            dataset_config.data_loader, default_args={'dataset_files': dataset_files,
                                                      'num_shards': dataset_config.device_num,
                                                      'shard_id': dataset_config.rank_id,
                                                      'columns_list': dataset_config.input_columns})
        return dataset
