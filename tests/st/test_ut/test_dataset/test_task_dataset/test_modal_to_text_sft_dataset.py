# Copyright 2024 Huawei Technologies Co., Ltd
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
"""test others"""
import pytest
import numpy as np
from mindformers.dataset.modal_to_text_sft_dataset import batch_add


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_batch_add():
    """
    Feature: modal_to_text_sft_dataset.batch_add
    Description: test batch_add function
    Expectation: success
    """
    input_col = np.zeros((2, 2), dtype=np.int32)
    res = batch_add(input_col, None)
    assert res[0].shape == (2, 1, 2, 2)
    assert res[0].tolist() == [[[[0, 0], [0, 0]]], [[[1, 0], [1, 0]]]]
