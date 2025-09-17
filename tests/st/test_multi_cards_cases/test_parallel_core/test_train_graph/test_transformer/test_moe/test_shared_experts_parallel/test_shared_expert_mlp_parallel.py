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

_LEVEL_0_TASK_TIME = 26
_LEVEL_1_TASK_TIME = 0
_TASK_TYPE = TaskType.TWO_CARDS_TASK
cur_dir = os.path.dirname(os.path.abspath(__file__))
port_id = int(os.environ.get("ASCEND_PORT_ID", random.randint(50000, 65535)))

class TestSharedExpertMLPTwoCards:
    """A test class for testing MLP"""

    @pytest.mark.level0
    def test_parallel_case(self):
        """
        Feature: SharedExpertMLP
        Description: Test Parallel Case: gate: True
        Exception: AssertionError
        """
        commands = [
            (f"msrun --worker_num=2 --local_worker_num=2 --master_port={port_id} --log_dir=log_2cards --join=True "
             f"{cur_dir}/run_shared_expert_mlp.py --tp 2 --gate"),
        ]
        result = subprocess.run(commands, shell=True, capture_output=True, text=True)
        assert result.returncode == 0, (
            f"Test script failed with non-zero exit code: "
            f"{result.returncode}.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"
        )
