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

"""
FilipConfig class, which consists FilipTextConfig and FilipVisionConfig
All configs here are inherited from BaseConfig
"""
from ..base_config import BaseConfig
from ...tools.register import MindFormerRegister, MindFormerModuleType
from ...mindformer_book import MindFormerBook
from ...tools import logger


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class FilipTextConfig(BaseConfig):
    """Config for filip text module"""
    def __init__(self, hidden_size=768, vocab_size=21128,
                 max_position_embeddings=32, num_hidden_layers=12, **kwargs):
        super(FilipTextConfig, self).__init__(hidden_size=hidden_size,
                                              vocab_size=vocab_size,
                                              max_position_embeddings=max_position_embeddings,
                                              num_hidden_layers=num_hidden_layers,
                                              **kwargs)


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class FilipVisionConfig(BaseConfig):
    """Config for filip vision module"""
    def __init__(self, hidden_size=1024, image_size=224,
                 patch_size=14, num_hidden_layers=24, **kwargs):
        super(FilipVisionConfig, self).__init__(hidden_size=hidden_size,
                                                image_size=image_size,
                                                patch_size=patch_size,
                                                num_hidden_layers=num_hidden_layers,
                                                **kwargs)


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class FilipConfig(BaseConfig):
    """Config for filip model"""
    _support_list = MindFormerBook.get_model_support_list()['filip']

    def __init__(self, text_config=None, vision_config=None, projection_dim=256, ratio=64,
                 checkpoint_name_or_path="", dtype="float16", **kwargs):
        if text_config is None:
            text_config = FilipTextConfig()
            logger.info("text_config is None. "
                        "Initializing the FilipTextConfig with default values.")
        elif isinstance(text_config, FilipTextConfig):
            pass
        else:
            raise TypeError(f"text_config should be a "
                            f"FiLipTextConfig class, but got {type(FilipTextConfig)}")

        if vision_config is None:
            vision_config = FilipVisionConfig()
            logger.info("vision_config is None. "
                        "Initializing the FilipTextConfig with default values.")
        elif isinstance(vision_config, FilipVisionConfig):
            pass
        else:
            raise TypeError("text_config should be a FilipVisionConfig"
                            f"class, but got {type(FilipVisionConfig)}")
        super(FilipConfig, self).__init__(text_config=text_config,
                                          vision_config=vision_config,
                                          projection_dim=projection_dim,
                                          ratio=ratio,
                                          checkpoint_name_or_path=checkpoint_name_or_path,
                                          dtype=dtype,
                                          **kwargs)
