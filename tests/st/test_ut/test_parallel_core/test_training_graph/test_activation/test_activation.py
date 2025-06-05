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
"""Test module for testing activation used for mindformers."""
import pytest
from mindformers.parallel_core.training_graph.activation import GELU, SiLU, SwiGlu
from tests.utils.double_benchmark import DoubleBenchmarkComparator
from .data_gen_utils import get_input, get_output


class TestActivation:
    """A test class for testing activation function"""

    def setup_method(self):
        """Setup method to prepare test environment"""
        self.input = get_input()

    def run_test(self, activation):
        """Helper function to run test and check results"""
        output = activation(self.input)
        npu_output = output.asnumpy()
        gpu_output, golden_output = get_output(activation)
        assert DoubleBenchmarkComparator.check_pass_or_not(npu_output, gpu_output, golden_output), (
            f"Activation test failed.\n"
            f"NPU output:\n{npu_output}\n\n"
            f"GPU output:\n{gpu_output}\n\n"
            f"Golden output:\n{golden_output}"
        )

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_gelu(self):
        """
        Feature: Activation
        Description: Test Case: activation_func='gelu',
        Exception: AssertionError
        """
        self.run_test(activation=GELU())

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_gelu_unapproximate(self):
        """
        Feature: Activation
        Description: Test Case: activation_func='gelu', approximate=False
        Exception: AssertionError
        """
        self.run_test(activation=GELU(approximate=False))

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_silu(self):
        """
        Feature: Activation
        Description: Test Case: activation_func='silu',
        Exception: AssertionError
        """
        self.run_test(activation=SiLU())

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_swiglu(self):
        """
        Feature: Activation
        Description: Test Case: activation_func='swiglu',
        Exception: AssertionError
        """
        self.run_test(activation=SwiGlu())
