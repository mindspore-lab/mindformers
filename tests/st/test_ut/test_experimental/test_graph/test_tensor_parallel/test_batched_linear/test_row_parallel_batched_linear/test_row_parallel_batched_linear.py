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
"""
Test module for testing the MLP used for mindformers.
How to run this:
    pytest tests/st/test_experimental/test_graph/test_batched_linear/test_column_batched_parallel_linear/test_row.py
"""
import os
import subprocess
import pytest

cur_dir = os.path.dirname(os.path.abspath(__file__))


class TestColumnParallelBatchedLinear:
    """A test class for testing ColumnBatchedParallelLinear"""

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_skip_bias_add_true_case(self):
        """
        Feature: RowParallelBatchedLinear
        Description: Test Base Case: bias=True, skip_bias_add=True, num_moe_experts=2
        Exception: AssertionError
        """
        result = subprocess.run(
            f"python {cur_dir}/run_row_parallel_batched_linear.py --bias --skip_bias_add",
            shell=True,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_skip_bias_add_false_case(self):
        """
        Feature: RowParallelBatchedLinear
        Description: Test skip_bias_add False Case: bias=True, skip_bias_add=False, num_moe_experts=2
        Exception: AssertionError
        """
        result = subprocess.run(
            f"python {cur_dir}/run_row_parallel_batched_linear.py --bias",
            shell=True,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_bias_false_case(self):
        """
        Feature: RowParallelBatchedLinear
        Description: Test bias False Case: bias=False, skip_bias_add=True, num_moe_experts=2
        Exception: AssertionError
        """
        result = subprocess.run(
            f"python {cur_dir}/run_row_parallel_batched_linear.py --bias",
            shell=True,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_all_false_case(self):
        """
        Feature: RowParallelBatchedLinear
        Description: Test All False Case: bias=False, skip_bias_add=False, num_moe_experts=2
        Exception: AssertionError
        """
        result = subprocess.run(
            ["python", f"{cur_dir}/run_row_parallel_batched_linear.py"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_num_moe_experts_error_case(self):
        """
        Feature: RowParallelBatchedLinear
        Description: Test num_moe_experts Error Case: num_moe_experts<=1
        Exception: AssertionError
        """
        result = subprocess.run(
            f"python {cur_dir}/run_row_parallel_batched_linear.py --num_moe_experts 1",
            shell=True,
            capture_output=True,
            text=True
        )
        assert result.returncode != 0
        assert ("ValueError: For RowParallelBatchedLinear, `is_expert` should be True and "
                "`num_moe_experts` should be larger than 1") in result.stderr
