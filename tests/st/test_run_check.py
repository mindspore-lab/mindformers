#  Copyright 2025 Huawei Technologies Co., Ltd
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ============================================================================
"""Test for run_check function"""
import pytest
from mindformers import run_check


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_run_check():
    """
    Feature: Test run_check function
    Description: Call run_check to check if MindSpore, MindFormers, CANN and driver versions are compatible
    Expectation: No exceptions raised, all checks pass
    """
    run_check()
