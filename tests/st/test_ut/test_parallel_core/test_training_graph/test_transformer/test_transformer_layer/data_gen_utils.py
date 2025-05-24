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
"""Data generation utilities for TransformerLayer test"""
import numpy as np
import mindspore as ms

# Default configuration for data generation
# These can be overridden by arguments in the test script if needed
DEFAULT_SEQ_LENGTH = 16
DEFAULT_BATCH_SIZE = 16
DEFAULT_HIDDEN_SIZE = 32
DEFAULT_FFN_HIDDEN_SIZE = 64 # Typically 4 * hidden_size, but smaller for faster tests
DEFAULT_NUM_HEADS = 4


def get_init_params(seq_length=DEFAULT_SEQ_LENGTH,
                    batch_size=DEFAULT_BATCH_SIZE,
                    hidden_size=DEFAULT_HIDDEN_SIZE,
                    compute_dtype=ms.bfloat16):
    """
    Generates initial parameters (inputs) for the TransformerLayer model.
    """
    np.random.seed(42)
    hidden_states_shape = (seq_length, batch_size, hidden_size)
    hidden_states_np = 0.01 * np.random.randn(*hidden_states_shape).astype(np.float32) # Initial data in fp32
    attention_mask_np = np.random.choice([True, False], size=(batch_size, 1, seq_length, seq_length)).astype(np.int32)
    init_params = {
        "hidden_states": ms.Tensor(hidden_states_np, dtype=compute_dtype),
        "attention_mask": ms.Tensor(attention_mask_np, dtype=compute_dtype), # Usually bool or float
    }
    return init_params


_common_output_shape = (DEFAULT_SEQ_LENGTH, DEFAULT_BATCH_SIZE, DEFAULT_HIDDEN_SIZE)
_random_output_data = np.random.rand(*_common_output_shape).astype(np.float32)
_random_extra_loss_data = np.array(np.random.rand() * 0.1, dtype=np.float32) # scalar loss

GOLDEN_DATA = {
    "output_default": _random_output_data,
    "extra_loss_default": _random_extra_loss_data,
}

GPU_DATA = {
    "output_default": _random_output_data.copy(), # Use .copy() to avoid modifying the same array
    "extra_loss_default": _random_extra_loss_data.copy(),
}

if __name__ == '__main__':
    # Example of how to generate and save data if needed for external use
    params = get_init_params()
    print("Generated hidden_states shape:", params["hidden_states"].shape)
    print("Generated attention_mask shape:", params["attention_mask"].shape)
    print("Data generation utilities ready.")
