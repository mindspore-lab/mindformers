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
"""Test LoRA"""

import os
import pytest

class TestLlama2:
    """A test class for testing lora."""
    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.run(order=1)
    def test_llama2_train(self):
        """
        Feature: test transformer block pretrain
        Description: run pynative mode to generate transformer block pretrain model.
        Expectation: test success
        """
        scripts_name = "run_llama2.py"
        yaml_name = "pretrain_llama2.yaml"
        device_num = 1

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)
        config_path = os.path.join(sh_path, yaml_name)

        scripts_cmd = f"{scripts_path}"
        cmd = f"msrun --worker_num={device_num} " + \
              f"--local_worker_num={device_num} " + \
              "--master_port=8118 " + \
              "--log_dir=msrun_log_llama2_pretrain " + \
              "--join=True " + \
              "--cluster_time_out=300 " + \
              f"{scripts_cmd} " + \
              f"--config_path={config_path}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_llama2_pretrain/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_llama2_pretrain/worker_*.log"
