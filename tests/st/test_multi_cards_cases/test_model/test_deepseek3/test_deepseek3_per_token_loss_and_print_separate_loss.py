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
"""Test DeepseekV3 training with calculate_per_token_loss=True and print_separate_loss=True"""
import os
from multiprocessing.pool import Pool
from pathlib import Path
import random
import pytest

from mindformers.tools.logger import logger
from tests.st.test_multi_cards_cases.utils import TaskType


_LEVEL_0_TASK_TIME = 105
_LEVEL_1_TASK_TIME = 0
_TASK_TYPE = TaskType.FOUR_CARDS_TASK


def run_command(command_info):
    cmd, log_path = command_info
    logger.info(f"Running command: {cmd}")
    ret = os.system(cmd)
    return ret, log_path


class TestDeepseekV3WithCalculatePerTokenLossAndPrintSeparateLoss:
    """Test class for DeepseekV3 in TND layout"""

    def setup_method(self):
        """Setup method to prepare test environment"""
        os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
        self.sh_path = Path(__file__).parent.resolve()
        self.run_script_path = self.sh_path / "run_deepseek3.py"
        assert self.run_script_path.exists(), f"Run script not found: {self.run_script_path}"

    @pytest.mark.level0
    def test_four_card_configurations_calculate_per_token_loss_and_print_seperate_loss(self):
        """Test four cards for DeepseekV3."""
        port_id = int(os.environ.get("ASCEND_PORT_ID", random.randint(50000, 65535)))
        cmd_list = [
            (f"msrun --worker_num=4 --local_worker_num=4 --master_port={port_id} --log_dir=./msrun_log_deepseekv3 "
             f"--join=True {self.run_script_path} "
             f"--mode=parallel_train_dp2_mp2_ep2_calculate_per_token_loss_and_print_seperate_loss",
             f"./msrun_log_deepseekv3/worker_3.log"),
        ]
        with Pool(len(cmd_list)) as pool:
            results = list(pool.imap(run_command, cmd_list))

        ret, log_path = results[0]
        assert ret == 0

        required_losses = {'lm_loss', 'aux_loss', 'mtp_loss'}
        found_losses = set()

        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if len(found_losses) < len(required_losses):
                        for loss in required_losses - found_losses:
                            if loss in line:
                                found_losses.add(loss)
                    else:
                        break
        except FileNotFoundError:
            raise AssertionError(f"log path {log_path} is not found.")
        except Exception as e:
            raise AssertionError(f"Error when reading log file: {e}")

        assert found_losses == required_losses
