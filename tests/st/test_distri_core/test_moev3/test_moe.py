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
"""Test MoE"""
import os
import pytest


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
class TestMoE:
    """A test class for testing MoE."""
    @pytest.mark.skip(reason="golden")
    @pytest.mark.run(order=5)
    def test_moev3_golden_bs8(self):
        """
        Feature: test_moe_golden
        Description: run graph mode moe to generate golden ckpt and loss
        Exception: AssertionError
        """
        os.environ['GRAPH_OP_RUN'] = "1"
        scripts_name = "run_moe.py"
        device_num = 1
        bs = 8

        rm_list = ["npy_golden*",
                   "msrun_log_graph*",
                   "kernel_meta*",
                   "golden_moe*"]
        print("")
        for rm_path in rm_list:
            rm_path = os.path.join(os.getcwd(), rm_path)
            print(f"removing {rm_path}")
            os.system(f"rm -rf {rm_path}")

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --generate_golden --dp=1 --ep=1 --mp=1 --batch_size={bs} --use_gmm=True"
        cmd = f"msrun --worker_num={device_num} " + \
                    f"--local_worker_num={device_num} " + \
                    f"--master_port=3722 " + \
                    f"--log_dir=msrun_log_graph " + \
                    f"--join=True " + \
                    f"--cluster_time_out=300 " + \
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_graph/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_graph/worker_*.log"

    # @pytest.mark.skip(reason="test_moev3_dp4ep4mp1_use_gmm")
    @pytest.mark.run(order=6)
    def test_moev3_dp8ep8mp1_use_gmm(self):
        """
        Feature: moe feature GroupedMatmul
        Description: test moe feature GroupedMatmul
        Exception: AssertionError
        """
        os.environ['HCCL_BUFFSIZE'] = "200"
        scripts_name = "run_moe.py"
        device_num = 8
        dp = 8
        mp = 1
        ep = 8
        assert dp * mp == device_num, "device_num should be equal to dp * mp"

        rm_list = ["npy_pynative_dp2*", "msrun_log_golden_use_gmm*", "kernel_meta*"]
        for rm_path in rm_list:
            rm_path = os.path.join(os.getcwd(), rm_path)
            print(f"removing {rm_path}")
            os.system(f"rm -rf {rm_path}")

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --dp={dp} --ep={ep} --mp={mp} --hidden_act='silu' --batch_size=1 " + \
                                     f"--use_gmm=True"
        cmd = f"msrun --worker_num={device_num} " + \
                    f"--local_worker_num={device_num} " + \
                    f"--master_port=3721 " + \
                    f"--log_dir=msrun_log_golden_use_gmm " + \
                    f"--join=True " + \
                    f"--cluster_time_out=300 " + \
                    f"{scripts_cmd}"
        print(f"\nrun cmd:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_golden_use_gmm/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_golden_use_gmm/worker_*.log"
