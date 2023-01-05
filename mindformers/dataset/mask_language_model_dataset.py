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
"""Masked Image Modeling Dataset."""
import os
import mindspore.common.dtype as mstype
import mindspore.dataset.transforms.c_transforms as C
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.logger import logger
from .dataloader import build_dataset_loader
from .base_dataset import BaseDataset


@MindFormerRegister.register(MindFormerModuleType.DATASET)
class MaskLanguageModelDataset(BaseDataset):
    """Bert pretrain dataset."""
    def __new__(cls, dataset_config: dict = None):
        logger.info("Now Create Masked Image Modeling Dataset.")
        cls.init_dataset_config(dataset_config)
        rank_id = int(os.getenv("RANK_ID", "0"))
        device_num = int(os.getenv("RANK_SIZE", "1"))
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
                                                      'num_shards': device_num, 'shard_id': rank_id})
        dataset = dataset.batch(dataset_config.batch_size,
                                drop_remainder=dataset_config.drop_remainder,
                                column_order=dataset_config.input_columns,
                                output_columns=dataset_config.input_columns,
                                num_parallel_workers=dataset_config.num_parallel_workers)
        dataset = dataset.repeat(dataset_config.repeat)
        type_cast_op = C.TypeCast(mstype.int32)
        for input_arg in dataset_config.input_columns:
            dataset = dataset.map(operations=type_cast_op, input_columns=input_arg)
        return dataset
