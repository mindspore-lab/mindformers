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
"""Mae Model."""
from mindspore import Tensor, Parameter

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.base_model import BaseModel
from .mae_config import MaeConfig


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class MaeModel(BaseModel):
    """Pretrain MAE Module."""

    def __init__(self, config=MaeConfig()):
        super(MaeModel, self).__init__(config)
        self.config = config
        self.param_test = Parameter(Tensor([1.0]))

    def construct(self, image):
        loss = 2.0 * self.param_test * image
        return loss.mean()
