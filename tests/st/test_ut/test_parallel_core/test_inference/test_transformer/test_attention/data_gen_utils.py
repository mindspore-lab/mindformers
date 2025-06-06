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
"""Generate data for mcore attention."""
import numpy as np


NUM_BLOCKS = 128
BLOCK_SIZE = 64
BATCH_SIZE = 2
SEQ_LENGTH = 2
HIDDEN_SIZE = 32
NUM_HEADS = 2


def get_init_params(is_prefill):
    """Generate initialization parameters"""
    if is_prefill:
        input_sa = np.random.normal(
            0, 0.01,
            [1, BATCH_SIZE * SEQ_LENGTH, HIDDEN_SIZE])
    else:
        input_sa = np.random.normal(
            0, 1,
            [BATCH_SIZE, 1, HIDDEN_SIZE])
    return {
        "input_sa": input_sa
    }
