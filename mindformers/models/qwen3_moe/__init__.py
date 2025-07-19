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
"""qwen3 model"""
from .utils import Qwen3MoePreTrainedModel
from .configuration_qwen3_moe import Qwen3MoeConfig
from .modeling_qwen3_moe import Qwen3MoeForCausalLM
from .modeling_qwen3_moe_infer import InferenceQwen3MoeForCausalLM
from .modeling_qwen3_moe_train import TrainingQwen3MoeForCausalLM

__all__ = [
    "Qwen3MoeConfig",
    "Qwen3MoeForCausalLM",
    "TrainingQwen3MoeForCausalLM",
    "InferenceQwen3MoeForCausalLM",
    "Qwen3MoePreTrainedModel",
]
