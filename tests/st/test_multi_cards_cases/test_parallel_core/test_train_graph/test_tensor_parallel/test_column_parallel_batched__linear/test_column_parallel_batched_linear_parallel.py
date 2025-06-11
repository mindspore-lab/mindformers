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
import random
import subprocess
import pytest
from tests.st.test_multi_cards_cases.utils import TaskType

cur_dir = os.path.dirname(os.path.abspath(__file__))

_LEVEL_0_TASK_TIME = 111
_LEVEL_1_TASK_TIME = 0
_TASK_TYPE = TaskType.FOUR_CARDS_TASK

class TestColumnParallelBatchedLinear:
    """A test class for testing ColumnBatchedParallelLinear"""
    @pytest.mark.level0
    def test_parallel_case(self):
        """
        Feature: ColumnParallelBatchedLinear
        Description: Test Base Parallel Case: dp=2, tp=2, ep=2, skip_weight_param_allocation=False, num_moe_experts=2,
                                              weight=None
        Exception: AssertionError
        """
        port_id = int(os.environ.get("ASCEND_PORT_ID", random.randint(50000, 65535)))
        commands = [
            (f"msrun --worker_num=4 --local_worker_num=4 "
             f"--master_port={port_id} --log_dir={cur_dir}/log_4cards --join=True "
             f"{cur_dir}/run_column_parallel_batched_linear.py --dp 2 --tp 2 --ep 2 "),
        ]
        result = subprocess.run(commands, shell=True, capture_output=True, text=True)
        assert result.returncode == 0, (
            f"Test script failed with non-zero exit code: "
            f"{result.returncode}.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"
        )
