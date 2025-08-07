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
"""Test module for testing fused SwiGlu in activation used for mindformers."""
import pytest
from mindformers.parallel_core.training_graph.activation import FusedSwiGlu, SwiGlu
from tests.utils.double_benchmark import DoubleBenchmarkComparator, DoubleBenchmarkStandard

from .data_gen_utils import get_input


class TestFusedSwiGlu:
    """A test class for testing FusedSwiGlu kernel"""

    def setup_method(self):
        """Setup method to prepare test environment"""
        self.input = get_input()

    def run_test(self):
        """Helper function to run test"""
        actual = FusedSwiGlu()(self.input).asnumpy()
        golden = SwiGlu()(self.input).asnumpy()
        standard = DoubleBenchmarkStandard(dtype="float32")
        assert DoubleBenchmarkComparator.check_pass_or_not(actual, golden, golden, standard), (
            f"FusedSwiGlu test failed.\n"
            f"FusedSwiGlu output:\n{actual}\n\n"
            f"SwiGlu output:\n{golden}\n\n"
        )

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_fused_swiglu(self):
        """
        Feature: Activation
        Description: Test Fused SwiGlu,
        Exception: AssertionError
        """
        self.run_test()
