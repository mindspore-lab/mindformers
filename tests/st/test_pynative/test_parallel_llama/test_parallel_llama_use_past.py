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
"""Test Pallel LLaMa"""
import os

import pytest


class TestParallelLLaMa:
    """A test class for testing Linear."""
    @pytest.mark.level0
    @pytest.mark.run(order=1)
    def test_llama_use_past_pynative(self):
        """
        Feature: test llama pynative with use_past
        Description: run pynative mode llama to predict
        Expectation: test success
        """
        scripts_name = "run_parallel_llama_use_past.py"
        device_num = 2

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)
        ckpt_path = ""

        scripts_cmd = f"{scripts_path} --config_path {sh_path}/../../../../configs/llama2/predict_llama2_13b.yaml " \
                      f"--checkpoint_path {ckpt_path}"
        cmd = f"msrun --worker_num={device_num} " + \
              f"--local_worker_num={device_num} " + \
              f"--master_port=8248 " + \
              f"--log_dir=msrun_log_pynative_predict " + \
              f"--join=True " + \
              f"--cluster_time_out=300 " + \
              f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative_predict/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_pynative_predict/worker_*.log"
