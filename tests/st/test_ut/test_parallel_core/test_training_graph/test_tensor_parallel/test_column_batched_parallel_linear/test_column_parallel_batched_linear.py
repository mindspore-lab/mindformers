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
"""Test module for testing ColumnParallelBatchedLinear used for mindformers."""
import os
import subprocess
import pytest

cur_dir = os.path.dirname(os.path.abspath(__file__))


class TestColumnParallelBatchedLinear:
    """A test class for testing ColumnBatchedParallelLinear"""

    def setup_method(self):
        """Setup method to prepare test environment"""
        self.run_file = 'run_column_parallel_batched_linear.py'

    def run_test(self, args: str = "", expected_error: str = None):
        """Helper function to run test and check results"""
        cmd = f"python {cur_dir}/{self.run_file} {args}"
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True
        )
        if not expected_error:
            assert result.returncode == 0, (
                f"Test script failed with non-zero exit code: "
                f"{result.returncode}.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"
            )
        else:
            assert result.returncode != 0
            assert expected_error in result.stderr, (
                f"Expected error message not found.\nExpected:\n{expected_error}\nStderr:\n{result.stderr}"
            )

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_skip_weight_false_case(self):
        """
        Feature: ColumnParallelBatchedLinear
        Description: Test skip_weight False Case: skip_weight_param_allocation=False, num_moe_experts=2, weight=None
        Exception: AssertionError
        """
        args = ""
        self.run_test(args)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_skip_weight_true_case(self):
        """
        Feature: ColumnParallelBatchedLinear
        Description: Test skip_weight True Case: skip_weight_param_allocation=True, num_moe_experts=2, weight=None
        Exception: AssertionError
        """
        args = "--skip_weight_param_allocation"
        expected_error = ("ValueError: 'For ColumnParallelBatchedLinear, when `skip_weight_param_allocation` "
                          "is enabled, `weight` is required, but got None'")
        self.run_test(args, expected_error)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_weight_true_case(self):
        """
        Feature: ColumnParallelBatchedLinear
        Description: Test weight True Case: skip_weight_param_allocation=False, num_moe_experts=2, weight=Tensor
        Exception: AssertionError
        """
        args = "--weight"
        self.run_test(args)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_all_true_case(self):
        """
        Feature: ColumnParallelBatchedLinear
        Description: Test All True Case: skip_weight_param_allocation=True, num_moe_experts=2, weight=Tensor
        Exception: AssertionError
        """
        args = "--skip_weight_param_allocation --weight"
        self.run_test(args)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_num_moe_experts_error_case(self):
        """
        Feature: ColumnParallelBatchedLinear
        Description: Test num_moe_experts Error Case: num_moe_experts<=1
        Exception: AssertionError
        """
        args = "--num_moe_experts 1"
        expected_error = ("ValueError: For ColumnParallelBatchedLinear, `is_expert` should be True and "
                          "`num_moe_experts` should be larger than 1")
        self.run_test(args, expected_error)
