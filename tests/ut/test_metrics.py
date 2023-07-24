# Copyright 2023 Huawei Technologies Co., Ltd
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
"""test lr schedule."""

import pytest

import mindspore as ms
from mindspore import Tensor

from mindformers.core.metric import PromptAccMetric


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_promptacc_metric():
    """
    Feature: Test PromptAccMetric
    Description: Test PromptAccMetric
    Expectation: ValueError
    """
    logits = Tensor(np.array([[[[0.4, -0.9],
                                [0.2, 0.7]],
                               [[0.9, 0.09],
                                [-0.4, 0.4]],
                               [[-0.2, 0.05],
                                [0.6, -0.8]],
                               [[0.6, -0.4],
                                [0.18, -0.56]]]]), ms.float16)
    input_ids = Tensor(np.array([[0, 1], [3, 7], [6, 2], [4, 4]]), ms.int32)
    input_mask = Tensor(np.array([[0, 1], [0, 1], [0, 1], [0, 1]]), ms.float32)
    labels = Tensor(np.array([[0]]), ms.int32)

    prompt_acc_std = 0

    metric = PromptAccMetric()
    metric.clear()
    metric.update(logits, input_ids, input_mask, labels)
    prompt_acc = metric.eval()

    error = 1e-8
    assert abs(prompt_acc - prompt_acc_std) < error
