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

"""Build Processor API."""
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from .build_feature_extractor import build_feature_extractor
from .build_tokenizer import build_tokenizer


def build_processor(
        config: dict = None, default_args: dict = None,
        module_type: str = 'processor', class_name: str = None, **kwargs):
    """Build processor API."""
    if config is None and class_name is None:
        return None
    if config is not None:
        if config.feature_extractor is not None:
            if config.feature_extractor.image_processor is not None:
                config.feature_extractor.image_processor = build_processor(
                    config.feature_extractor.image_processor)
            config.feature_extractor = build_feature_extractor(config.feature_extractor)
        if config.tokenizer is not None:
            config.tokenizer = build_tokenizer(config.tokenizer, **kwargs)

        return MindFormerRegister.get_instance_from_cfg(
            config, MindFormerModuleType.PROCESSOR, default_args=default_args)
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)
