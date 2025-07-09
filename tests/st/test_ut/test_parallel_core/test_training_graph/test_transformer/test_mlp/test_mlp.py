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
"""Test module for testing MLP used for mindformers."""
import os
import subprocess
import pytest

cur_dir = os.path.dirname(os.path.abspath(__file__))


class TestMLP:
    """A test class for testing MLP"""

    def setup_method(self):
        """Setup method to prepare test environment"""
        self.run_file = 'run_mlp.py'

    def run_test(self, args: str = ""):
        """Helper function to run test and check results"""
        cmd = f"python {cur_dir}/{self.run_file} {args}"
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, (
            f"Test script failed with non-zero exit code: "
            f"{result.returncode}.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"
        )

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_backward_case(self):
        """
        Feature: MLP
        Description: Test All False Case: input_size: None, add_bias_linear: False, gated_linear_unit: False,
                                          enable_backward: True
        Exception: AssertionError
        """
        args = "--enable_backward"
        self.run_test(args)

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_gated_linear_unit_case(self):
        """
        Feature: MLP
        Description: Test gated_linear_unit Case: input_size: None, add_bias_linear: False, gated_linear_unit: True
        Exception: AssertionError
        """
        args = "--gated_linear_unit"
        self.run_test(args)

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_add_bias_linear_case(self):
        """
        Feature: MLP
        Description: Test add_bias_linear Case: input_size: None, add_bias_linear: True, gated_linear_unit: True
        Exception: AssertionError
        """
        args = "--add_bias_linear --gated_linear_unit"
        self.run_test(args)

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_input_size_case(self):
        """
        Feature: MLP
        Description: Test input_size Case: input_size: 32, add_bias_linear: True, gated_linear_unit: True
        Exception: AssertionError
        """
        args = "--input_size 32 --add_bias_linear --gated_linear_unit"
        self.run_test(args)
