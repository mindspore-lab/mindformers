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
""" test modules in inference"""
import pytest
import numpy as np
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore.common.api import _cell_graph_executor
from mindformers.inference.postprocess_sampler import Sampler


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_postprocess_sampler():
    """
    Feature: Test Sampler construct
    Description: Test the forward
    Expectation: No exception
    """
    logits = ms.Tensor(np.ones((8, 512)), dtype=mstype.float16)
    temperature = ms.Tensor([0.8], dtype=mstype.float16)
    sample_model = Sampler()
    sample_model.set_train(False)
    _cell_graph_executor.compile(sample_model, logits, temperature)
