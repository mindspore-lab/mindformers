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
        A dataset for ZeroShotImageClassificationTrainer.

    Examples:
        >>> import os
        >>> from mindformers import MindFormerBook, MindFormerConfig, build_dataset
        >>> project_path = MindFormerBook.get_project_path()
        >>> config_path = os.path.join(project_path, "configs", "clip",
        >>>                     "run_clip_vit_b_32_zero_shot_image_classification_cifar100.yaml")
        >>> config = MindFormerConfig(config_path)
            Note:
                Put cifar100 dataset to ./
                The detailed data setting could refer to ./configs/clip/clip.md
        >>> config.eval_dataset_task.dataset_config.batch_size = 1
        >>> dataset = build_dataset(config.eval_dataset_task)
        >>> for item in dataset:
        >>>     print(item)
        >>>     break
        [Tensor(shape=[1, 3, 224, 224], dtype=Float32, value=
        [[[[1.11282456e+000, 1.11282456e+000, ... 1.47778523e+000, 1.47778523e+000],
        [1.11282456e+000, 1.11282456e+000, ... 1.47778523e+000, 1.47778523e+000],
        [1.11282456e+000, 1.11282456e+000, ... 1.47778523e+000, 1.47778523e+000],
        ...
        [1.97748125e-001, 1.97748125e-001, ... 1.12205243e+000, 1.12205243e+000],
        [1.97748125e-001, 1.97748125e-001, ... 1.12205243e+000, 1.12205243e+000],
        [1.97748125e-001, 1.97748125e-001, ... 1.12205243e+000, 1.12205243e+000]]]]),
        Tensor(shape=[1, 100, 77], dtype=Int32, value=
        [[[49406,   320,  1674 ...     0,     0,     0],
        [49406,   320,  1674 ...     0,     0,     0],
        [49406,   320,  1674 ...     0,     0,     0],
        ...
        [49406,   320,  1674 ...     0,     0,     0],
        [49406,   320,  1674 ...     0,     0,     0],
        [49406,   320,  1674 ...     0,     0,     0]]]),
        Tensor(shape=[1], dtype=Int32, value= [49])]
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
            dataset = dataset.map(
                input_columns="image",
                operations=transforms,
                num_parallel_workers=dataset_config.num_parallel_workers,
                python_multiprocessing=dataset_config.python_multiprocessing
            )

        if text_transforms is not None:
            dataset = dataset.map(
                input_columns="text",
                operations=text_transforms,
                num_parallel_workers=dataset_config.num_parallel_workers,
                python_multiprocessing=dataset_config.python_multiprocessing
            )

        dataset = dataset.batch(dataset_config.batch_size,
                                drop_remainder=dataset_config.drop_remainder,
                                num_parallel_workers=dataset_config.num_parallel_workers)
        dataset = dataset.repeat(dataset_config.repeat)
        return dataset
