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
"""Data generation utilities for VocabParallelCrossEntropy tests with random data"""
import numpy as np

def get_init_params(batch_size, seq_length, vocab_size):
    """
    Generates random initial parameters (inputs) for VocabParallelCrossEntropy.
    """
    np.random.seed(42)

    logits_shape = (batch_size * seq_length, vocab_size)
    logits = 0.01 * np.random.randn(*logits_shape).astype(np.float32)

    target_shape = (batch_size * seq_length,)
    target = np.random.randint(0, vocab_size, size=target_shape).astype(np.int32)
    input_mask = np.random.randint(0, 2, size=target_shape).astype(np.float32)

    if np.sum(input_mask) == 0 and input_mask.size > 0:
        input_mask[0] = 1.0

    return {
        "logits": logits,
        "target": target,
        "input_mask": input_mask,
    }



GOLDEN_DATA = {
    "numerator": np.array(41.57823944091796875),
    "denominator": np.array(6.),
}

GPU_DATA = {
    "numerator": np.array(41.57823944091796875),
    "denominator": np.array(6.),
}
