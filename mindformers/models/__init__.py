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

'''models init'''
from mindformers.models import mae, build_model, build_config, clip
from .mae import *
from .build_model import *
from .base_config import BaseConfig
from .clip import *
from .base_model import BaseModel
from .build_tokenizer import build_tokenizer
from .build_feature_extractor import build_feature_extractor
from .build_processor import build_processor
from .base_feature_extractor import BaseFeatureExtractor
from .base_tokenizer import BaseTokenizer
from .base_processor import BaseProcessor

__all__ = []
__all__.extend(mae.__all__)
__all__.extend(clip.__all__)
