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
import mindspore.dataset.transforms.c_transforms as C
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.logger import logger
from .base_dataset import BaseDataset
from .transforms import build_transforms


@MindFormerRegister.register(MindFormerModuleType.DATASET)
class ImageToTextRetrievalDataset(BaseDataset):
    """Image-text Retrieval Dataset for filip fine-tuning and evaluation."""
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
            dataset = dataset.map(
                operations=transforms,
                num_parallel_workers=dataset_config.num_parallel_workers,
                python_multiprocessing=dataset_config.python_multiprocessing,
                input_columns="image", output_columns=['image']
            )

        type_cast_op = C.TypeCast(mstype.float32)
        dataset = dataset.map(operations=type_cast_op, input_columns="image", output_columns=['image'])

        dataset = dataset.project(["image", "token"])
        dataset = dataset.batch(dataset_config.batch_size, drop_remainder=dataset_config.drop_remainder)
        return dataset
