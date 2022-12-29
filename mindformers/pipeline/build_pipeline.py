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
"""Build Pipeline API."""
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


def build_pipeline(
        config: dict = None, default_args: dict = None,
        module_type: str = 'pipeline', class_name: str = None, **kwargs):
    r"""Build pipeline For MindFormer.
    Instantiate the pipeline from MindFormerRegister's registry.

    Args:
        config (dict): The task pipeline's config. Default: None.
        default_args (dict): The default argument of pipeline API. Default: None.
        module_type (str): The module type of MindFormerModuleType. Default: 'pipeline'.
        class_name (str): The class name of pipeline API. Default: None.

    Return:
        The function instance of pipeline API.

    Examples:
        >>> from mindformers import build_pipeline
        >>> pipeline_from_class_name = build_pipeline(class_name='image_classification', model='vit_base_p16')
    """
    if config is None and class_name is None:
        return None
    if config is not None:
        return MindFormerRegister.get_instance_from_cfg(
            config, MindFormerModuleType.PIPELINE, default_args=default_args)
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)
