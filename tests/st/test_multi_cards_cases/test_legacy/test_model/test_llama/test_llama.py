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
"""test legacy parallel llama model."""

import os
import random
import subprocess
import pytest

from tests.st.test_multi_cards_cases.utils import TaskType

_LEVEL_0_TASK_TIME = 100
_LEVEL_1_TASK_TIME = 0
_TASK_TYPE = TaskType.FOUR_CARDS_TASK

WORK_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.level0
def test_llama_train_mp2_pp2():
    """
    Feature: Legacy llama model test
    Description: Test legacy llama model on 4 cards
    Expectation: No error
    """
    # Define the environment variables
    os.environ['MS_MEMORY_POOL_RECYCLE'] = '1'

    # Define the command as a list
    command = [
        "msrun",
        "--worker_num=4",
        "--local_worker_num=4",
        f"--master_port={int(os.environ.get('ASCEND_PORT_ID', random.randint(50000, 65535)))}",
        f"--log_dir={WORK_DIR}/log_train_mp2_pp2",
        "--join=True",
        f"{WORK_DIR}/run_parallel.py",
        "--mode=parallel_train_mp2_pp2"
    ]

    # Run the command using subprocess
    result = subprocess.run(command, shell=False, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Test script failed with non-zero exit code: {result.returncode}.\n"
            f"Stdout:\n{result.stdout}\nStderr:\n{result.stderr}"
        )
