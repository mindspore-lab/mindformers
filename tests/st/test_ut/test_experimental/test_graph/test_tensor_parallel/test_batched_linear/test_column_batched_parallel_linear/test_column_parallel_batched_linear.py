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

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_skip_bias_add_true_case(self):
        """
        Feature: ColumnParallelBatchedLinear
        Description: Test skip_bias_add True Case: bias=True, skip_bias_add=True, skip_weight_param_allocation=False,
                                     num_moe_experts=2, weight=None
        Exception: AssertionError
        """
        result = subprocess.run(
            f"python {cur_dir}/run_column_parallel_batched_linear.py --bias --skip_bias_add",
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
        Feature: ColumnParallelBatchedLinear
        Description: Test skip_bias_add False Case: bias=True, skip_bias_add=False, skip_weight_param_allocation=False,
                                                    num_moe_experts=2, weight=None
        Exception: AssertionError
        """
        result = subprocess.run(
            f"python {cur_dir}/run_column_parallel_batched_linear.py --bias",
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
        Feature: ColumnParallelBatchedLinear
        Description: Test bias False Case: bias=False, skip_bias_add=True, skip_weight_param_allocation=False,
                                           num_moe_experts=2, weight=None
        Exception: AssertionError
        """
        result = subprocess.run(
            f"python {cur_dir}/run_column_parallel_batched_linear.py --bias",
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
        Feature: ColumnParallelBatchedLinear
        Description: Test All False Case: bias=False, skip_bias_add=False, skip_weight_param_allocation=False,
                                          num_moe_experts=2, weight=None
        Exception: AssertionError
        """
        result = subprocess.run(
            f"python {cur_dir}/run_column_parallel_batched_linear.py",
            shell=True,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_skip_weight_true_case(self):
        """
        Feature: ColumnParallelBatchedLinear
        Description: Test skip_weight True Case: bias=True, skip_bias_add=True, skip_weight_param_allocation=True,
                                                 num_moe_experts=2, weight=None
        Exception: AssertionError
        """
        result = subprocess.run(
            f"python {cur_dir}/run_column_parallel_batched_linear.py --bias --skip_bias_add "
            f"--skip_weight_param_allocation",
            shell=True,
            capture_output=True,
            text=True
        )
        assert result.returncode != 0
        assert ("ValueError: 'For ColumnParallelBatchedLinear, when `skip_weight_param_allocation` is enabled,"
                " `weight` is required, but got None'") in result.stderr

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_weight_true_case(self):
        """
        Feature: ColumnParallelBatchedLinear
        Description: Test weight True Case: bias=True, skip_bias_add=True, skip_weight_param_allocation=False,
                                            num_moe_experts=2, weight=Tensor
        Exception: AssertionError
        """
        result = subprocess.run(
            f"python {cur_dir}/run_column_parallel_batched_linear.py --bias --skip_bias_add --weight",
            shell=True,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_all_true_case(self):
        """
        Feature: ColumnParallelBatchedLinear
        Description: Test All True Case: bias=True, skip_bias_add=True, skip_weight_param_allocation=True,
                                         num_moe_experts=2, weight=Tensor
        Exception: AssertionError
        """
        result = subprocess.run(
            f"python {cur_dir}/run_column_parallel_batched_linear.py --bias --skip_bias_add "
            f"--skip_weight_param_allocation --weight",
            shell=True,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_num_moe_experts_error_case(self):
        """
        Feature: ColumnParallelBatchedLinear
        Description: Test num_moe_experts Error Case: num_moe_experts<=1
        Exception: AssertionError
        """
        result = subprocess.run(
            f"python {cur_dir}/run_column_parallel_batched_linear.py --num_moe_experts 1",
            shell=True,
            capture_output=True,
            text=True
        )
        assert result.returncode != 0
        assert ("ValueError: For ColumnParallelBatchedLinear, `is_expert` should be True and "
                "`num_moe_experts` should be larger than 1") in result.stderr
