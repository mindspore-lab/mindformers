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
Note: The config module of Parameter Efficient Tuning module.
"""
from mindformers.models.base_config import BaseConfig
from mindformers.models.utils import convert_mstype
from mindformers.pet.constants import PetType


__all__ = ['LoraConfig']


class PetConfig(BaseConfig):
    """
    Pet model config class which defines the tuning algorithm and pretrained for tuning.
    """
    def __init__(self,
                 pet_type=PetType.LORA,
                 base_mode=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.pet_type = pet_type
        self.base_mode = base_mode


class LoraConfig(PetConfig):
    """
    Lora tuning algorithm config.
    """
    def __init__(self,
                 lora_rank: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.01,
                 lora_a_init: str = 'normal',
                 lora_b_init: str = 'zero',
                 param_init_type: str = 'float16',
                 compute_dtype: str = 'float16',
                 **kwargs):
        super().__init__(pet_type=PetType.LORA, **kwargs)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_a_init = lora_a_init
        self.lora_b_init = lora_b_init
        self.param_init_type = convert_mstype(param_init_type)
        self.compute_dtype = convert_mstype(compute_dtype)
