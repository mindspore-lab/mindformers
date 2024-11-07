# Copyright 2024 Huawei Technologies Co., Ltd
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
Test module for testing the paralleled qwen2 interface used for mindformers.
How to run this:
    pytest tests/st/test_model/test_qwen2_model/test_parallel.py
"""
import os
from multiprocessing.pool import Pool

import pytest

from tests.st.test_model.test_llama2_model.test_parallel import run_command, check_results

cur_dir = os.path.dirname(os.path.abspath(__file__))


class TestQwen2Parallel:
    """A test class for testing pipeline."""

    @staticmethod
    def setup_method():
        os.environ['MS_MEMORY_POOL_RECYCLE'] = '1'

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    def test_base_cases(self):
        """
        Feature: Trainer.train() and Trainer.predict()
        Description: Test parallel trainer for training and prediction.
        Expectation: AssertionError
        """
        commands = [
            (f"export ASCEND_RT_VISIBLE_DEVICES=4,5 && "
             f"msrun --worker_num=2 --local_worker_num=2 --master_port=8518 --log_dir=log_predict_mp2 --join=True "
             f"{cur_dir}/run_parallel.py --mode parallel_predict_mp2", 'log_predict_mp2/worker_0.log'),
        ]

        with Pool(len(commands)) as pool:
            results = list(pool.imap(run_command, commands))
        check_results(commands, results)
