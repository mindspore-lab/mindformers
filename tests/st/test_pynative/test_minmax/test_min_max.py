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

from mindformers import MindFormerConfig

test_config_list = []
for transpose_b in [True, False]:
    for keepdims in [True, False]:
        config_column = MindFormerConfig(
            mode="column",
            reduce_out=True,
            transpose_b=transpose_b,
            keepdims=keepdims
        )
        config_row = MindFormerConfig(
            mode="row",
            reduce_out=False,
            transpose_b=transpose_b,
            keepdims=keepdims
        )
        test_config_list += [config_column, config_row]


class TestParallelLLaMaxMin:
    """A test class for testing Linear."""

    @pytest.mark.level0
    @pytest.mark.parametrize("test_config", test_config_list)
    def test_min_max(self, test_config):
        """
        Feature: generate llama golden
        Description: run graph mode llama to generate golden ckpt and loss
        Expectation: test success
        """
        os.environ['GRAPH_OP_RUN'] = "1"
        scripts_name = "run_min_max.py"
        device_num = 2

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        os.system("kill -9 $(lsof -i:8124 | awk '{print $2}')")

        scripts_cmd = f"{scripts_path} --mode {test_config.mode}" + \
            (" --keepdims" if test_config.keepdims else "") + \
            (" --reduce_out" if test_config.reduce_out else "") + \
            (" --transpose_b" if test_config.transpose_b else "")

        cmd = f"msrun --worker_num={device_num} " + \
              f"--local_worker_num={device_num} " + \
              f"--master_port=8124 " + \
              f"--log_dir=msrun_log " + \
              f"--join=True " + \
              f"--cluster_time_out=300 " + \
              f"{scripts_cmd}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log/worker_*.log"
