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
Test module for testing the paralleled deepseek3 interface used for mindformers.
How to run this:
    pytest tests/st/test_model/test_deepseek3_model/test_parallel.py
"""
import os
from multiprocessing.pool import Pool

import pytest


cur_dir = os.path.dirname(os.path.abspath(__file__))


def run_command(command_info):
    cmd, log_path = command_info
    ret = os.system(cmd)
    return ret, log_path


def check_results(commands, results):
    error_idx = [_ for _ in range(len(results)) if results[_][0] != 0]
    for idx in error_idx:
        print(f"testcase {commands[idx]} failed. please check log {results[idx][1]}.")
        os.system(f"grep -E 'ERROR|error|Error' {results[idx][1]} -C 5")
    assert error_idx == []


class TestDeepseek3Parallel:
    """A test class for testing pipeline."""

    @staticmethod
    def setup_method():
        # runtime performance
        os.environ["MS_DEV_RUNTIME_CONF"] = "multi_stream:true"

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    def test_base_cases(self):
        """
        Feature: Trainer.train() and Trainer.predict()
        Description: Test parallel trainer for training and prediction.
        Expectation: AssertionError
        """
        master_port = 8000
        hccl_if_base_port = 60100
        commands = [
            (f"export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && export HCCL_IF_BASE_PORT={hccl_if_base_port} &&"
             f"msrun --worker_num=8 --local_worker_num=8 --master_port={master_port} "
             f"--log_dir=log_train_gmm_dp2mp2ep4pp2 "
             f"--join=True {cur_dir}/run_parallel.py --mode parallel_train_gmm_dp2mp2ep4pp2",
             'log_train_gmm_dp2mp2ep4pp2/worker_7.log')
        ]

        with Pool(len(commands)) as pool:
            results = list(pool.imap(run_command, commands))
        check_results(commands, results)


    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    def test_deredundency_cases(self):
        """
        Feature: Trainer.train() and Trainer.predict()
        Description: Test parallel trainer for training and prediction.
        Expectation: AssertionError
        """
        master_port = 8010
        hccl_if_base_port = 60200
        commands = [
            (f"export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && export HCCL_IF_BASE_PORT={hccl_if_base_port} &&"
             f"msrun --worker_num=8 --local_worker_num=8 --master_port={master_port} "
             f"--log_dir=log_train_gmm_dp2mp2ep2pp2_deredundency "
             f"--join=True {cur_dir}/run_parallel.py --mode parallel_train_gmm_dp2mp2ep2pp2_deredundency",
             'log_train_gmm_dp2mp2ep2pp2_deredundency/worker_7.log')
        ]

        with Pool(len(commands)) as pool:
            results = list(pool.imap(run_command, commands))
        check_results(commands, results)


    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    def test_bmm_cases(self):
        """
        Feature: Trainer.train() and Trainer.predict()
        Description: Test parallel trainer for training and prediction.
        Expectation: AssertionError
        """
        master_port = 8020
        hccl_if_base_port = 60500
        commands = [
            (f"export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && export HCCL_IF_BASE_PORT={hccl_if_base_port} &&"
             f"msrun --worker_num=8 --local_worker_num=8 --master_port={master_port} "
             f"--log_dir=log_train_bmm_dp2mp2ep2pp2 "
             f"--join=True {cur_dir}/run_parallel.py --mode parallel_train_bmm_dp2mp2ep2pp2",
             'log_train_bmm_dp2mp2ep2pp2/worker_7.log')
        ]

        with Pool(len(commands)) as pool:
            results = list(pool.imap(run_command, commands))
        check_results(commands, results)
