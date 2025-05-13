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
"""Generate data for test mlp."""
import numpy as np


def get_init_params(input_size, ffn_hidden_size):
    """Generate initialization parameters"""
    np.random.seed(2025)
    fc1_add_gate_weight_shape = (ffn_hidden_size * 2, input_size)
    fc1_no_gate_weight_shape = (ffn_hidden_size, input_size)
    fc2_weight_shape = (input_size, ffn_hidden_size)
    return {
        "input": np.random.normal(loc=0, scale=0.01, size=(2, 2, input_size)),
        "fc1_gate_weight": np.random.normal(loc=0, scale=0.01, size=fc1_add_gate_weight_shape),
        "fc1_no_gate_weight": np.random.normal(loc=0, scale=0.01, size=fc1_no_gate_weight_shape),
        "fc2_weight": np.random.normal(loc=0, scale=0.01, size=fc2_weight_shape),
        "fc1_gate_bias": np.random.normal(loc=0, scale=0.01, size=(ffn_hidden_size * 2,)),
        "fc1_no_gate_bias": np.random.normal(loc=0, scale=0.01, size=(ffn_hidden_size,)),
        "fc2_bias": np.random.normal(loc=0, scale=0.01, size=(input_size,))
    }
