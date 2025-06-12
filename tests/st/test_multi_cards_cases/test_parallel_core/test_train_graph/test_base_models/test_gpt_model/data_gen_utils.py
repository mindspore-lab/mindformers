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

DEFAULT_SEQ_LENGTH = 16
DEFAULT_BATCH_SIZE = 16
DEFAULT_HIDDEN_SIZE = 32
DEFAULT_FFN_HIDDEN_SIZE = 64
DEFAULT_NUM_HEADS = 4


def get_init_params(seq_length=DEFAULT_SEQ_LENGTH, batch_size=DEFAULT_BATCH_SIZE):
    """
    Generates initial parameters (inputs) for the TransformerLayer model.
    """
    np.random.seed(42)
    input_ids = np.zeros(shape=(batch_size, seq_length))
    loss_mask = np.ones(shape=(batch_size, seq_length))
    init_params = {
        "input_ids": ms.Tensor(input_ids, dtype=ms.int32),
        "loss_mask": ms.Tensor(loss_mask, dtype=ms.int32)
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
    "output_default": _random_output_data.copy(),
    "extra_loss_default": _random_extra_loss_data.copy(),
}
