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
"""Mindspore quantization utils."""
from .a8w8_mindspore import A8W8LinearMethod
from .a8dynw8_mindspore import A8W8DynamicLinearMethod

QUANTIZATION_METHOD_MAPPING = {
    "W8A8": A8W8LinearMethod,
    "W8A8_DYNAMIC": A8W8DynamicLinearMethod
}

mapping_rules = {
    '.linear_q_down_proj': ('.linear_qkv_down_proj', '.linear_q_down_proj', 'q_down'),
    '.linear_kv_down_proj': ('.linear_qkv_down_proj', '.linear_kv_down_proj', 'kv_down'),
    '.linear_q': ('.linear_qkv', '.linear_q', 'q'),
    '.linear_k': ('.linear_qkv', '.linear_k', 'k'),
    '.linear_v': ('.linear_qkv', '.linear_v', 'v'),
    '.linear_kv': ('.linear_qkv', '.linear_kv', 'kv'),
    '.mlp.gating': ('.mlp.linear_fc1', '.mlp.gating', 'gating'),
    '.mlp.hidden': ('.mlp.linear_fc1', '.mlp.hidden', 'hidden'),
    '.mlp.shared_experts.gating': ('.mlp.shared_experts.linear_fc1', '.mlp.shared_experts.gating', 'gating'),
    '.mlp.shared_experts.hidden': ('.mlp.shared_experts.linear_fc1', '.mlp.shared_experts.hidden', 'hidden'),
    '.mlp.experts.gating': ('.mlp.experts.linear_fc1', '.mlp.experts.gating', 'gating'),
    '.mlp.experts.hidden': ('.mlp.experts.linear_fc1', '.mlp.experts.hidden', 'hidden')
}
