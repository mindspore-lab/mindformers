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
"""Mae Encoder API."""
from mindformers.models.base_model import BaseModel
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from .mae_config import MaeConfig


@MindFormerRegister.register(MindFormerModuleType.ENCODER)
class MaeEncoder(BaseModel):
    """vision encoder for mae"""

    def __init__(self, config=MaeConfig()):
        super().__init__()
        self.config = config
