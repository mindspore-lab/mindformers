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
"""Qwen3Moe models' APIs."""
__all__ = [
    'Qwen3MoeForCausalLM',
]

import os

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.qwen3_moe.utils import Qwen3MoePreTrainedModel
from mindformers.models.qwen3_moe.modeling_qwen3_moe_infer import InferenceQwen3MoeForCausalLM


@MindFormerRegister.register(MindFormerModuleType.MODELS, legacy=False)
class Qwen3MoeForCausalLM(Qwen3MoePreTrainedModel):
    r"""
    Provide Qwen3Moe Model for training and inference.
    Args:
        config (Qwen3MoeConfig): The config of Qwen3Moe model.

    Returns:
        Tensor, the loss or logits of the network.
    """

    def __new__(cls, config):
        r"""
        get run mode to init different model.

        Args:
            config (Qwen3MoeConfig): The config of Qwen3Moe model.

        Raises:
            NotImplementedError: Train mode is not supported for Qwen3Moe model.

        Returns:
            Tensor, the loss or logits of the network

        """
        if os.environ.get("RUN_MODE", "predict") == "predict":
            return InferenceQwen3MoeForCausalLM(config=config)
        raise NotImplementedError("Train mode is not supported for Qwen3Moe model.")
