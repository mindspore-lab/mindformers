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
"""Glm4Moe models' APIs."""
__all__ = [
    'Glm4MoeForCausalLM',
]

import os

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.glm4_moe.utils import Glm4MoePreTrainedModel
from mindformers.models.glm4_moe.modeling_glm4_moe_infer import InferenceGlm4MoeForCausalLM


@MindFormerRegister.register(MindFormerModuleType.MODELS, legacy=False)
class Glm4MoeForCausalLM(Glm4MoePreTrainedModel):
    r"""
    Provide Glm4Moe Model for training and inference.
    Args:
        config (Glm4MoeConfig): The config of Glm4Moe model.

    Returns:
        Tensor, the loss or logits of the network.
    """

    def __new__(cls, config):
        r"""
        get run mode to init different model.

        Args:
            config (Glm4MoeConfig): The config of Glm4Moe model.

        Raises:
            NotImplementedError: Train mode is not supported for Glm4Moe model.

        Returns:
            Tensor, the loss or logits of the network

        """
        if os.environ.get("RUN_MODE", "predict") == "predict":
            return InferenceGlm4MoeForCausalLM(config=config)
        raise NotImplementedError("Train mode is not supported for Glm4Moe model.")
