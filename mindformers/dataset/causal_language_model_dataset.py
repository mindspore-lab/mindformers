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
import mindspore
import mindspore.common.dtype as mstype
import mindspore.dataset.transforms.c_transforms as C
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.logger import logger
from mindformers.models.build_tokenizer import build_tokenizer
from mindformers.tools.utils import is_version_ge
from .dataloader import build_dataset_loader
from .base_dataset import BaseDataset


def get_input_data_batch_slice_map(input_ids, eos_token_id, dis, rank_id: int = 0):
    """
    Generate position_id and attention_mask according to input_ids considering eos reset

    Args:
        input_ids: the input token ids
        eos_token_id: the id for <EOS>
        dis: the slice value for each rank
        rank_id: the current rank id
    Returns:
        batch_input_ids: the input token ids
        batch_position_ids: the position ids cosidering eos reset
        batch_attention_mask: the attention mask considering eos reset
    """
    rank = int(rank_id)
    input_ids = input_ids[rank*dis: (rank + 1)*dis]
    seq_length = input_ids.shape[1] - 1
    # Initialize position_ids and attention_mask
    batch_input_ids = input_ids
    batch_position_ids = np.ones((dis, seq_length))
    batch_attention_mask = np.ones((dis, seq_length, seq_length))

    # Loop through batches
    for bs_i in range(len(input_ids)):
        # Get normal position_ids and attention_mask
        local_ids = input_ids[bs_i]
        batch_attention_mask[bs_i] = np.tril(np.ones(shape=(seq_length, seq_length)))
        batch_position_ids[bs_i] = np.arange(seq_length)
        # Find the index of <EOS>
        eos_index = batch_position_ids[bs_i, local_ids[:-1] == eos_token_id].astype(np.int32)
        prev_index = 0
        for i in range(eos_index.size):
            # Reset position_ids and attention_mask considering <EOS>
            index = eos_index[i]
            batch_attention_mask[bs_i, (index + 1):, :(index + 1)] = 0
            batch_position_ids[bs_i, (index + 1):] -= (index + 1 - prev_index)
            prev_index = index + 1
    return batch_input_ids, batch_position_ids, batch_attention_mask


@MindFormerRegister.register(MindFormerModuleType.DATASET)
class CausalLanguageModelDataset(BaseDataset):
    """Causal Language Model pretrain dataset.

    Examples:
        >>> from mindformers.tools.register import MindFormerConfig
        >>> from mindformers import MindFormerBook
        >>> from mindformers.dataset import CausalLanguageModelDataset
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
        >>> dataset_from_name = build_dataset(class_name='CausalLanguageModelDataset',
        ...                                   dataset_config=config.train_dataset_task.dataset_config)
        >>> # 3) use class to build dataset
        >>> dataset_from_class = CausalLanguageModelDataset(config.train_dataset_task.dataset_config)
    """
    def __new__(cls, dataset_config: dict = None):
        logger.info("Now Create Causal Language Model Dataset.")
        rank_id = int(os.getenv("RANK_ID", "0"))
        device_num = int(os.getenv("RANK_SIZE", "1"))
        dataset_config = copy.deepcopy(dataset_config)
        cls.init_dataset_config(dataset_config)
        rank_id, device_num = cls._check_device_rank_for_parallel(rank_id, device_num)
        dataset_config.rank_id = rank_id
        dataset_config.device_num = device_num
        if dataset_config.data_loader.type != "MindDataset" and \
                dataset_config.data_loader.type != "TFRecordDataset":
            dataset = cls._process_raw_text_data(dataset_config)
        else:
            dataset = cls._process_mindrecord_data(dataset_config)

        type_cast_op = C.TypeCast(mstype.int32)
        if dataset_config.eos_reset:
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

            dataset = dataset.batch(dataset_config.batch_size,
                                    drop_remainder=dataset_config.drop_remainder,
                                    output_columns=dataset_config.input_columns)
            map_func = lambda input_ids: get_input_data_batch_slice_map(input_ids,
                                                                        eos_token_id=dataset_config.eos_token_id,
                                                                        rank_id=rank_id,
                                                                        dis=dis)
            if is_version_ge(mindspore.__version__, '2.0.0'):
                dataset = dataset.map(operations=map_func, input_columns=dataset_config.input_columns,
                                      output_columns=dataset_config.output_columns)
            else:
                dataset = dataset.map(operations=map_func, input_columns=dataset_config.input_columns,
                                      output_columns=dataset_config.output_columns,
                                      column_order=dataset_config.output_columns)
            dataset = dataset.project(columns=dataset_config.output_columns)

            for input_arg in dataset_config.output_columns:
                dataset = dataset.map(operations=type_cast_op, input_columns=input_arg)
        else:
            dataset = dataset.batch(dataset_config.batch_size,
                                    drop_remainder=dataset_config.drop_remainder,
                                    output_columns=dataset_config.input_columns,
                                    num_parallel_workers=dataset_config.num_parallel_workers)
            dataset = dataset.project(columns=dataset_config.input_columns)
            for input_arg in dataset_config.input_columns:
                dataset = dataset.map(operations=type_cast_op, input_columns=input_arg)

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

        dataset = dataset.map(map_func, input_columns=dataset_config.input_columns,
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
