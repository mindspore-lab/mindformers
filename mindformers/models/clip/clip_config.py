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

"""
ClipConfig class, which consists of ClipTextConfig and ClipVisionConfig
All configs here are inherited from BaseConfig
"""
from typing import Optional

from ..base_config import BaseConfig
from ...tools.register import MindFormerRegister, MindFormerModuleType
from ...mindformer_book import MindFormerBook
from ...tools import logger

@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class ClipTextConfig(BaseConfig):
    r"""
    Config For Clip Text Module

    Args:
        hidden_size (Optional[int]): The dims of text features.
        vocab_size (Optional[int]): The size of vocab.
        max_position_embeddings (Optional[int]): Length of input_ids.
        num_hidden_layers (Optional[int]): The number of transformer layers in text encoder.

    Examples:
        >>> from mindformers import ClipTextConfig
        >>> ClipTextConfig(hidden_size=256, vocab_size=40000)
            {'hidden_size': 256, 'vocab_size': 40000,
             'max_position_embeddings': 77, 'num_hidden_layers': 12}
    """
    def __init__(self, hidden_size: Optional[int] = 512,
                 vocab_size: Optional[int] = 49408,
                 max_position_embeddings: Optional[int] = 77,
                 num_hidden_layers: Optional[int] = 12,
                 **kwargs):
        super(ClipTextConfig, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.num_hidden_layers = num_hidden_layers


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class ClipVisionConfig(BaseConfig):
    r"""
    Config For Clip Vision Module

    Args:
        hidden_size (Optional[int]): The dims of image features.
        image_size (Optional[int]): The shape of input image.
        patch_size (Optional[int]): The size of image patch.
        num_hidden_layers (Optional[int]): The number of transformer layers.

    Examples:
        >>> from mindformers import ClipVisionConfig
        >>> ClipVisionConfig(hidden_size=512, image_size=256)
            {'hidden_size': 512, 'image_size': 256,
             'patch_size': 32, 'num_hidden_layers': 12}
    """
    def __init__(self, hidden_size: Optional[int] = 768,
                 image_size: Optional[int] = 224,
                 patch_size: Optional[int] = 32,
                 num_hidden_layers: Optional[int] = 12,
                 **kwargs):
        super(ClipVisionConfig, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_hidden_layers = num_hidden_layers


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class ClipConfig(BaseConfig):
    r"""
    Config For Clip Model

    Args:
        text_config (Optional[ClipTextConfig]): The config of text transformer.
        vision_config (Optional[ClipVisionConfig]): The config of vision transformer.
        projection_dim (Optional[int]): The dims of projected featrues.
        ratio (Optional[int]): The ratio of attentions heads and feature size
        checkpoint_name_or_path (Optional[str]): The path of checkpoint(.ckpt)
            or a support model name in ClipConfig.show_support_list()
        dtype (Optional[str]): The type of tensors in model, ["float16", "float32"].

    Raises:
         TypeError: If the type of text_config is not ClipTextConfig or the type of vision_config
            is not ClipVisionConfig

    Examples:
        >>> from mindformers import ClipConfig
        >>> ClipConfig.show_support_list()
            INFO - support list of ClipConfig is:
            INFO -    ['clip_vit_b_32']
            INFO - -------------------------------------
        >>> config = ClipConfig.from_pretrained('clip_vit_b_32')
        >>> config
            {'text_config': {'hidden_size': 512, 'vocab_size': 49408, 'max_position_embeddings': 77,
             'num_hidden_layers': 12}, 'vision_config': {'hidden_size': 768, 'image_size': 224,
              'patch_size': 32, 'num_hidden_layers': 12}, 'projection_dim': 512, 'ratio': 64,
               'checkpoint_name_or_path': 'clip_vit_b_32', 'dtype': 'float16'}
        >>> config.save_pretrained(save_directory="./", save_name="clip_config")
            INFO - config saved successfully!
    """
    _support_list = MindFormerBook.get_config_support_list()['clip']

    def __init__(self, text_config=None, vision_config=None, projection_dim=512, ratio=64,
                 checkpoint_name_or_path="clip_vit_b_32", dtype="float16", **kwargs):
        if text_config is None:
            text_config = ClipTextConfig()
            logger.info("text_config is None. Initializing the CLIPTextConfig with default values.")
        elif isinstance(text_config, ClipTextConfig):
            pass
        else:
            raise TypeError(f"text_config should be a "
                            f"CLipTextConfig class, but got {type(ClipTextConfig)}")

        if vision_config is None:
            vision_config = ClipVisionConfig()
            logger.info("vision_config is None."
                        " Initializing the CLIPTextConfig with default values.")
        elif isinstance(vision_config, ClipVisionConfig):
            pass
        else:
            raise TypeError("text_config should be a CLipVisionConfig"
                            f" class, but got {type(ClipVisionConfig)}")

        super(ClipConfig, self).__init__(**kwargs)

        self.text_config = text_config
        self.vision_config = vision_config
        self.projection_dim = projection_dim
        self.ratio = ratio
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self.dtype = dtype
