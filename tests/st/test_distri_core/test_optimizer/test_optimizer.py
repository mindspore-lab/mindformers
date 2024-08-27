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
"""Test Optimizer"""
import os
import pytest


class TestAdamWeightDecayZeRO2:
    """A test class for testing AdamWeightDecayZeRO2."""

    @pytest.mark.run(order=1)
    def test_adamwzero2(self):
        """
        Feature: AdamWeigthDecayZeRO2
        Description: Test AdamWeightDecayZeRO2 opt parallel
        Exception: AssertionError
        """
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        ret = os.system(f"bash {sh_path}/msrun_launch_optimizer.sh 8")
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log/worker_0.log -C 3")
        assert ret == 0
