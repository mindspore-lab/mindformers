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
"""Build Dataset API."""
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType, MindFormerConfig


def build_dataset(
        config: dict = None, default_args: dict = None,
        module_type: str = 'dataset', class_name: str = None, **kwargs):
    r"""Build dataset For MindFormer.
    Instantiate the dataset from MindFormerRegister's registry.

    Args:
        config (dict): The task dataset's config. Default: None.
        default_args (dict): The default argument of dataset API. Default: None.
        module_type (str): The module type of MindFormerModuleType. Default: 'dataset'.
        class_name (str): The class name of dataset API. Default: None.

    Return:
        The function instance of dataset API.

    Examples:
        >>> from mindformers import build_dataset
        >>> from mindformers.dataset import check_dataset_config
        >>> from mindformers.tools.register import MindFormerConfig
        >>> config = MindFormerConfig('configs/vit/run_vit_base_p16_224_100ep.yaml')
        >>> check_dataset_config(config)
        >>> # 1) use config dict to build dataset
        >>> dataset_from_config = build_dataset(config.train_dataset_task)
        >>> # 2) use class name to build dataset
        >>> dataset_class_name = build_dataset(class_name='ImageCLSDataset', dataset_config=config.train_dataset_task)
    """
    if config is None and class_name is None:
        return None
    if config is not None:
        if isinstance(config, dict) and not isinstance(config, MindFormerConfig):
            config = MindFormerConfig(**config)
        return MindFormerRegister.get_instance_from_cfg(
            config, MindFormerModuleType.DATASET, default_args=default_args)
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)
