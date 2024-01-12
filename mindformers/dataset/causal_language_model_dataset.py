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
from typing import Union, Optional, Callable
import numpy as np
import mindspore.common.dtype as mstype
from mindspore.dataset.transforms import TypeCast
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.logger import logger
from mindformers.version_control import get_dataset_map
from .dataloader import build_dataset_loader
from .base_dataset import BaseDataset


def get_input_data_batch_slice_map(input_ids, eod_token_id, dis, rank_id: int = 0):
    """
    Generate position_id and attention_mask according to input_ids considering eod reset

    Args:
        input_ids: the input token ids
        eod_token_id: the id for <EOD>
        dis: the slice value for each rank
        rank_id: the current rank id
    Returns:
        batch_input_ids: the input token ids
        batch_position_ids: the position ids cosidering eod reset
        batch_attention_mask: the attention mask considering eod reset
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
        eod_index = batch_position_ids[bs_i, local_ids[:-1] == eod_token_id].astype(np.int32)
        prev_index = 0
        for i in range(eod_index.size):
            # Reset position_ids and attention_mask considering <EOS>
            index = eod_index[i]
            batch_attention_mask[bs_i, (index + 1):, :(index + 1)] = 0
            batch_position_ids[bs_i, (index + 1):] -= (index + 1 - prev_index)
            prev_index = index + 1
    return batch_input_ids, batch_position_ids, batch_attention_mask


@MindFormerRegister.register(MindFormerModuleType.DATASET)
class CausalLanguageModelDataset(BaseDataset):
    """
    Causal Language Model pretrain dataset.
    output input_ids columns

    Args:
        dataset_config (Optional[dict]):
            Config for dataset.
        data_loader (Union[dict, Callable]):
            Config for data loader or a data loader object.
        input_columns (list):
            Column name before the map function.
        output_columns (list):
            Column name after the map function.
        batch_size (int):
            Size of each batch. Default: 8.
        drop_remainder (bool):
            Whether to discard the last batch when the number of data items contained in the last batch is smaller
            than batch_size. Default: True.
        num_parallel_workers (int):
            Specifies the number of concurrent processes or threads for map operations
            to accelerate processing. Default: 8.
        python_multiprocessing (bool):
            Enabling the Python Multi-Process Mode to Accelerate Map Operations. Default: False.
        repeat (int):
            Number of times this dataset is repeated. Default: 1.
        seed (int):
            Random seed number. Default: 0.
        prefetch_size (int):
            Buffer queue size of each data processing operation in the pipeline. Default: 1.
        numa_enable (bool):
            Indicates whether to use the NUMA binding function. Default: False.
        eod_reset (bool):
            Specifies whether to reset the EOD. Default: False.
        eod_token_id (int):
            Indicates the token id of the EOD.
        auto_tune (bool):
            Indicates whether to enable automatic optimization of data processing parameters. Default: False.
        autotune_per_step (int):
            Specifies the interval for adjusting the configuration step of automatic data acceleration. Default: 10.
        filepath_prefix (str):
            Path for saving optimized parameter configurations. Default: './autotune'.
        profile (bool):
            Whether to enable data collection. Default: False.

    Returns:
        A dataset for CausalLanguageModelDataset.

    Examples:
        >>> # 1) Create an instance using a MindFormerConfig.
        >>> from mindformers.tools.register import MindFormerConfig
        >>> from mindformers import MindFormerBook
        >>> from mindformers.dataset import CausalLanguageModelDataset
        >>> from mindformers.dataset import check_dataset_config
        >>> config_dict_list = MindFormerBook.get_trainer_support_task_list()
        >>> config_path = config_dict_list['text_generation']['gpt2']
        >>> # Initialize a MindFormerConfig instance with a specific config file of yaml.
        >>> config = MindFormerConfig(config_path)
        >>> config.train_dataset.data_loader.dataset_dir = "The required task dataset path"
        >>> # Note:
        >>> #     The detailed data setting could refer to
        >>> #     https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/gpt2.md
        >>> check_dataset_config(config)
        >>> # use class to build dataset
        >>> dataset_from_class = CausalLanguageModelDataset(config.train_dataset_task.dataset_config)
        >>>
        >>> # 2) Creating an instance using other parameters.
        >>> from mindspore.dataset import MindDataset
        >>> from mindformers.dataset import CausalLanguageModelDataset
        >>> data_loader = MindDataset(dataset_files="The required task dataset path", shuffle=True)
        >>> dataset_from_param = CausalLanguageModelDataset(data_loader=data_loader,
        ...                                                 input_columns=["input_ids", "attention_mask"])
    """

    # pylint: disable=W0613
    def __new__(cls,
                dataset_config: Optional[dict] = None,
                data_loader: Union[dict, Callable] = None,
                input_columns: list = None,
                output_columns: list = None,
                batch_size: int = 8,
                drop_remainder: bool = True,
                num_parallel_workers: int = 8,
                python_multiprocessing: bool = False,
                repeat: int = 1,
                seed: int = 0,
                prefetch_size: int = 1,
                numa_enable: bool = False,
                eod_reset: bool = False,
                eod_token_id: Optional[int] = None,
                auto_tune: bool = False,
                filepath_prefix: str = './autotune',
                autotune_per_step: int = 10,
                profile: bool = False,
                **kwargs):
        logger.info("Now Create Causal Language Model Dataset.")
        dataset_config = cls.check_dataset_config(dataset_config, locals())
        dataset_config = copy.deepcopy(dataset_config)
        cls.init_dataset_config(dataset_config)
        rank_id, device_num = cls._generate_shard_info()
        dataset_config.rank_id = rank_id
        dataset_config.device_num = device_num

        if isinstance(dataset_config.data_loader, dict):
            if dataset_config.data_loader.type != "MindDataset" and \
                    dataset_config.data_loader.type != "TFRecordDataset":
                dataset = cls._process_raw_text_data(dataset_config)
            else:
                dataset = cls._process_mindrecord_data(dataset_config)
        else:
            dataset = dataset_config.data_loader

        type_cast_op = TypeCast(mstype.int32)
        if dataset_config.eod_reset:
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
                                                                        eod_token_id=dataset_config.eod_token_id,
                                                                        rank_id=rank_id,
                                                                        dis=dis)
            dataset = get_dataset_map(dataset, map_func,
                                      input_columns=dataset_config.input_columns,
                                      output_columns=dataset_config.output_columns)
            dataset = dataset.project(columns=dataset_config.output_columns)

            for input_arg in dataset_config.output_columns:
                dataset = get_dataset_map(dataset, type_cast_op,
                                          input_columns=input_arg)
        else:
            dataset = dataset.batch(dataset_config.batch_size,
                                    drop_remainder=dataset_config.drop_remainder,
                                    output_columns=dataset_config.input_columns,
                                    num_parallel_workers=dataset_config.num_parallel_workers)
            dataset = dataset.project(columns=dataset_config.input_columns)
            for input_arg in dataset_config.input_columns:
                dataset = get_dataset_map(dataset, type_cast_op,
                                          input_columns=input_arg)
        dataset = dataset.repeat(dataset_config.repeat)
        return dataset

    @classmethod
    def _process_raw_text_data(cls, dataset_config):
        """Process the text data"""
        dataset_dir = dataset_config.data_loader.pop("dataset_dir")
        dataset = build_dataset_loader(
            dataset_config.data_loader, default_args={'dataset_dir': dataset_dir,
                                                      'num_shards': dataset_config.device_num,
                                                      'shard_id': dataset_config.rank_id})
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
