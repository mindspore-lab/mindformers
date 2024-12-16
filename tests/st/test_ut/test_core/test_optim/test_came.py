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
"""test came.py"""

import pytest
from mindformers import GPT2Config, GPT2LMHeadModel
from mindformers.core.optim import Came


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_came():
    """
    Feature: Came.__init__
    Description: test Came __init__
    Expectation: success
    """
    gpt2_config = GPT2Config(num_layers=1)
    net = GPT2LMHeadModel(gpt2_config)
    assert isinstance(Came(params=net.trainable_params(), learning_rate=0.1), Came)
