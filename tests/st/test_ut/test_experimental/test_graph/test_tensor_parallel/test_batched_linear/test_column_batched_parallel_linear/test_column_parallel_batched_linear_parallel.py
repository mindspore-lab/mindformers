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
    @pytest.mark.env_single
    def test_parallel_case(self):
        """
        Feature: ColumnParallelBatchedLinear
        Description: Test Base Parallel Case: dp=2, tp=2, ep=2, bias=True, skip_bias_add=True,
                                              skip_weight_param_allocation=False, num_moe_experts=2, weight=None
        Exception: AssertionError
        """
        commands = [
            (f"export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 && msrun --worker_num=4 --local_worker_num=4 "
             f"--master_port=8118 --log_dir=log_4cards --join=True "
             f"{cur_dir}/run_column_parallel_batched_linear.py --dp 2 --tp 2 --ep 2 --bias --skip_bias_add"),
        ]
        result = subprocess.run(commands, shell=True, capture_output=True, text=True)
        assert result.returncode == 0
