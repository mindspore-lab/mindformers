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
"""Test DeepseekV3 training"""
import os
from multiprocessing.pool import Pool
from pathlib import Path
import random
import pytest

from mindformers.tools.logger import logger
from tests.st.test_multi_cards_cases.utils import TaskType


_LEVEL_0_TASK_TIME = 120
_LEVEL_1_TASK_TIME = 0
_TASK_TYPE = TaskType.FOUR_CARDS_TASK


def run_command(command_info):
    cmd, log_path = command_info
    logger.info(f"Running command: {cmd}")
    ret = os.system(cmd)
    return ret, log_path


def check_results(commands, results):
    error_idx = [_ for _ in range(len(results)) if results[_][0] != 0]
    for idx in error_idx:
        print(f"testcase {commands[idx]} failed. please check log {results[idx][1]}.")
        os.system(f"grep -E 'ERROR|error|Error' {results[idx][1]} -C 5")
    assert error_idx == []


class TestDeepseekV3:
    """Test class for DeepseekV3"""

    def setup_method(self):
        """Setup method to prepare test environment"""
        os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
        self.sh_path = Path(__file__).parent.resolve()
        self.run_script_path = self.sh_path / "run_deepseek3.py"
        assert self.run_script_path.exists(), f"Run script not found: {self.run_script_path}"

    @pytest.mark.level0
    def test_four_card_configurations(self):
        """Test eight cards for DeepseekV3."""
        port_id = int(os.environ.get("ASCEND_PORT_ID", random.randint(50000, 65535)))
        cmd_list = [
            (f"msrun --worker_num=4 --local_worker_num=4 --master_port={port_id} --log_dir=./msrun_log_deepseekv3 "
             f"--join=True {self.run_script_path} --mode=parallel_train_dp2_mp2_ep2",
             f"./msrun_log_deepseekv3/worker_3.log"),
        ]
        with Pool(len(cmd_list)) as pool:
            results = list(pool.imap(run_command, cmd_list))
        check_results(cmd_list, results)
