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
"""Contrastive Language Image Pretrain Dataset."""
import os

from .dataloader import build_dataset_loader
from .transforms import build_transforms
from .sampler import build_sampler
from .base_dataset import BaseDataset
from ..tools import logger
from ..models.build_tokenizer import build_tokenizer
from ..tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.DATASET)
class ContrastiveLanguageImagePretrainDataset(BaseDataset):
    r"""
    Contrastive Language Image Pretrain Dataset API.
    output image and text columns

    Args:
        dataset_config (dict): Config for dataset.

    Returns:
        A dataset for ContrastiveLanguageImagePretrainTrainer.

    Examples:
        >>> import os
        >>> from mindformers import MindFormerBook, MindFormerConfig, build_dataset
        >>> project_path = MindFormerBook.get_project_path()
        >>> config_path = os.path.join(project_path, "configs", "clip",
        >>>                     "run_clip_vit_b_32_pretrain_flickr8k.yaml")
        >>> config = MindFormerConfig(config_path)
            Note:
                Put flickr8k dataset to ./checkpoint_download
                The detailed data setting could refer to ./configs/clip/clip.md
        >>> config.train_dataset_task.dataset_config.batch_size = 1
        >>> dataset = build_dataset(config.train_dataset_task)
        >>> for item in dataset:
        >>>     print(item)
        >>>     break
            [Tensor(shape=[1, 3, 224, 224], dtype=Float32, value=
            [[[[4.99690473e-001, 6.74871564e-001, ... 3.68304640e-001, 2.36918822e-001],
            [7.91658998e-001, 7.62462139e-001, ... -2.01033935e-001, -1.13443382e-001],
            ...
            [-5.98575652e-001, -6.12795711e-001, ... 1.47755420e+000, 1.46333420e+000],
            [-3.85274649e-001, -6.27015769e-001, ... 1.42067397e+000, 1.43489408e+000],
            [-7.97656536e-001, -1.01095748e+000, ... 9.37191546e-001, 9.08751369e-001]]]]),
             Tensor(shape=[1, 77], dtype=Int32, value=
            [[49406,  1237, 18250 ...     0,     0,     0]])]
    """
    def __new__(cls, dataset_config: dict = None):
        """new method"""
        logger.info("Now Create Contrastive Language Image Pretrain Dataset.")
        cls.init_dataset_config(dataset_config)
        rank_id = int(os.getenv("RANK_ID", "0"))
        device_num = int(os.getenv("RANK_SIZE", "1"))

        dataset = build_dataset_loader(
            dataset_config.data_loader, default_args={'num_shards': device_num, 'shard_id': rank_id})

        transforms = build_transforms(dataset_config.transforms)

        tokenizer = build_tokenizer(dataset_config.tokenizer)
        text_transforms = build_transforms(dataset_config.text_transforms,
                                           default_args={"tokenizer": tokenizer})

        sampler = build_sampler(dataset_config.sampler)

        if sampler is not None:
            dataset = dataset.use_sampler(sampler)

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
