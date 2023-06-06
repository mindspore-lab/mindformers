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

"""models init"""
from .bert import *
from .mae import *
from .vit import *
from .swin import *
from .clip import *
from .t5 import *
from .gpt2 import *
from .glm import *
from .llama import *
from .pangualpha import *
from .filip import *
from .bloom import *
from .base_tokenizer import *
from .base_config import BaseConfig
from .base_model import BaseModel
from .base_processor import BaseProcessor, BaseImageProcessor, BaseAudioProcessor
from .build_tokenizer import build_tokenizer
from .build_processor import build_processor
from .build_model import build_model_config, build_head, \
    build_model, build_encoder

__all__ = ['BaseConfig', 'BaseModel', 'BaseProcessor', 'BaseImageProcessor',
           'BaseAudioProcessor']

__all__.extend(bert.__all__)
__all__.extend(mae.__all__)
__all__.extend(vit.__all__)
__all__.extend(swin.__all__)
__all__.extend(clip.__all__)
__all__.extend(t5.__all__)
__all__.extend(gpt2.__all__)
__all__.extend(glm.__all__)
__all__.extend(llama.__all__)
__all__.extend(pangualpha.__all__)
__all__.extend(filip.__all__)
__all__.extend(bloom.__all__)
__all__.extend(base_tokenizer.__all__)
