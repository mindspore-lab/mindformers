# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Glm4 models' APIs."""

import os

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from .utils import Glm4PreTrainedModel
from .modeling_glm4_infer import InferenceGlm4ForCausalLM


@MindFormerRegister.register(MindFormerModuleType.MODELS, legacy=False)
class Glm4ForCausalLM(Glm4PreTrainedModel):
    r"""
    Provide Glm4 Model for training and inference.
    Args:
        config (Glm4Config): The config of Glm4 model.

    Returns:
        Tensor, the loss or logits of the network.
    """

    def __new__(cls, config):
        r"""
        get run mode to init different model.

        Args:
            config (Glm4Config): The config of Glm4 model.

        Raises:
            NotImplementedError: Train mode is not supported for Glm4 model.

        Returns:
            Tensor, the loss or logits of the network

        """
        if os.environ.get("RUN_MODE") == "predict":
            return InferenceGlm4ForCausalLM(config=config)
        raise NotImplementedError("Train mode is not supported for Glm4 model.")
