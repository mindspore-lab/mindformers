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
"""Generate data for mcore flash_attn UT of inference."""
import numpy as np

NUM_BLOCKS = 128
BLOCK_SIZE = 64
BATCH_SIZE = 2
SEQ_LENGTH = 2
HIDDEN_SIZE = 32
NUM_HEADS = 2


def get_init_params(is_prefill, n_kv_heads):
    """Generate initialization parameters"""
    if is_prefill:
        q_shape = (1, BATCH_SIZE * SEQ_LENGTH, HIDDEN_SIZE)
        query = np.random.normal(0, 0.01, q_shape)

        kv_shape = (
            1, BATCH_SIZE * SEQ_LENGTH,
            int(n_kv_heads * HIDDEN_SIZE / NUM_HEADS)
        )
        key = np.random.normal(0, 0.01, kv_shape)
        value = np.random.normal(0, 0.01, kv_shape)

    else:
        q_shape = (BATCH_SIZE, 1, HIDDEN_SIZE)
        query = np.random.normal(0, 0.01, q_shape)

        kv_shape = (
            BATCH_SIZE, 1,
            int(n_kv_heads * HIDDEN_SIZE / NUM_HEADS)
        )
        key = np.random.normal(0, 0.01, kv_shape)
        value = np.random.normal(0, 0.01, kv_shape)
    return {
        "query": query,
        "key": key,
        "value": value
    }
