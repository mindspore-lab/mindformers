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
"""Telechat2 models' APIs."""
__all__ = [
    'TeleChat2ForCausalLM',
]
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from .utils import Telechat2PreTrainedModel
from .modeling_telechat2_infer import InferenceTelechat2ForCausalLM


@MindFormerRegister.register(MindFormerModuleType.MODELS, legacy=False)
class TeleChat2ForCausalLM(Telechat2PreTrainedModel):
    r"""
    Provide Qwen3 Model for training and inference.
    Args:
        config (Qwen3Config): The config of Qwen3 model.

    Returns:
        Tensor, the loss or logits of the network.
    """

    def __new__(cls, config):
        r"""
        get run mode to init different model.

        Args:
            config (Qwen3Config): The config of qwen3 model.

        Raises:
            NotImplementedError: Train mode is not supported for Qwen3 model.

        Returns:
            Tensor, the loss or logits of the network

        """
        return InferenceTelechat2ForCausalLM(config=config)
