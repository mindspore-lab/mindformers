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
"""Test ParallelMLP"""
import os

import pytest


class TestParallelTransformer:
    """A test class for testing Linear."""
    # os.environ['ASCEND_RT_VISIBLE_DEVICES']="0,1"
    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    # @pytest.mark.skip(reason='no need to run graph mode')
    @pytest.mark.run(order=1)
    def test_transformer_golden(self):
        """
        Feature: generate transformer golden
        Description: run graph mode linaer to generate golden ckpt and loss
        Expectation: test success
        """
        os.environ['GRAPH_OP_RUN'] = "1"
        scripts_name = "run_transformer_with_ring_attention.py"
        device_num = 1

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --generate_golden_with_fa"
        cmd = f"msrun --worker_num={device_num} " +\
            f"--local_worker_num={device_num} " +\
            f"--master_port=2001 " +\
            f"--log_dir=msrun_log_generate_golden_with_fa " +\
            f"--join=True " +\
            f"--cluster_time_out=300 " +\
            f"{scripts_cmd}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(
            f"grep -E 'ERROR|error' {sh_path}/msrun_log_generate_golden_with_fa/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_generate_golden_with_fa/worker_*.log"


    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.skip(reason='passed')
    @pytest.mark.run(order=2)
    def test_transformer_pynative_cp2_dp2(self):
        """
        Feature: test ParallelTransformer pynative
        Description: run pynative mode transformer to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_transformer_with_ring_attention.py"
        device_num = 4

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --use_cp_and_dp"
        cmd = f"msrun --worker_num={device_num} " +\
            f"--local_worker_num={device_num} " +\
            f"--master_port=8123 " +\
            f"--log_dir=msrun_log_cp2dp2 " +\
            f"--join=True " +\
            f"--cluster_time_out=300 " +\
            f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(
            f"grep -E 'ERROR|error' {sh_path}/msrun_log_cp2dp2/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_cp2dp2/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.skip(reason='passed')
    @pytest.mark.run(order=3)
    def test_transformer_pynative_cp2_dp2_zero1(self):
        """
        Feature: test ParallelTransformer pynative
        Description: run pynative mode transformer to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_transformer_with_ring_attention.py"
        device_num = 4

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --use_cp_and_zero1"
        cmd = f"msrun --worker_num={device_num} " +\
            f"--local_worker_num={device_num} " +\
            f"--master_port=2006 " +\
            f"--log_dir=msrun_log_cp2dp2zero1 " +\
            f"--join=True " +\
            f"--cluster_time_out=300 " +\
            f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(
            f"grep -E 'ERROR|error' {sh_path}/msrun_log_cp2dp2zero1/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_cp2dp2zero1/worker_*.log"


    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.skip(reason='passed')
    @pytest.mark.run(order=4)
    def test_transformer_pynative_cp2_tp2(self):
        """
        Feature: test ParallelTransformer pynative
        Description: run pynative mode transformer to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_transformer_with_ring_attention.py"
        device_num = 4

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --use_cp_and_tp"
        cmd = f"msrun --worker_num={device_num} " +\
            f"--local_worker_num={device_num} " +\
            f"--master_port=8122 " +\
            f"--log_dir=msrun_log_cp2tp2 " +\
            f"--join=True " +\
            f"--cluster_time_out=300 " +\
            f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(
            f"grep -E 'ERROR|error' {sh_path}/msrun_log_cp2tp2/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_cp2tp2/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.skip(reason='passed')
    @pytest.mark.run(order=5)
    def test_transformer_pynative_cp2_dp2_tp2(self):
        """
        Feature: test ParallelTransformer pynative
        Description: run pynative mode transformer to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_transformer_with_ring_attention.py"
        device_num = 8

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --use_cp_and_tp_and_dp"
        cmd = f"msrun --worker_num={device_num} " +\
            f"--local_worker_num={device_num} " +\
            f"--master_port=2040 " +\
            f"--log_dir=msrun_log_cp2dp2tp2 " +\
            f"--join=True " +\
            f"--cluster_time_out=300 " +\
            f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(
            f"grep -E 'ERROR|error' {sh_path}/msrun_log_cp2dp2tp2/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_cp2dp2tp2/worker_*.log"



    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.skip(reason='passed')
    @pytest.mark.run(order=6)
    def test_transformer_pynative_cp2_dp2_tp2_zero1(self):
        """
        Feature: test ParallelTransformer pynative
        Description: run pynative mode transformer to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_transformer_with_ring_attention.py"
        device_num = 8

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --use_cp_and_tp_and_zero1"
        cmd = f"msrun --worker_num={device_num} " +\
            f"--local_worker_num={device_num} " +\
            f"--master_port=8122 " +\
            f"--log_dir=msrun_log_tp2cp2dp2_zero1 " +\
            f"--join=True " +\
            f"--cluster_time_out=300 " +\
            f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(
            f"grep -E 'ERROR|error' {sh_path}/msrun_log_tp2cp2dp2_zero1/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_tp2cp2dp2_zero1/worker_*.log"


    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.skip(reason='passed')
    @pytest.mark.run(order=7)
    def test_transformer_pynative_cp2_dp2_tp2_zero2(self):
        """
        Feature: test ParallelTransformer pynative
        Description: run pynative mode transformer to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_transformer_with_ring_attention.py"
        device_num = 8

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --use_cp_and_tp_and_zero2"
        cmd = f"msrun --worker_num={device_num} " +\
            f"--local_worker_num={device_num} " +\
            f"--master_port=8122 " +\
            f"--log_dir=msrun_log_tp2cp2dp2_zero2 " +\
            f"--join=True " +\
            f"--cluster_time_out=300 " +\
            f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(
            f"grep -E 'ERROR|error' {sh_path}/use_cp_and_tp_and_zero2/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check use_cp_and_tp_and_zero2/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.skip(reason='passed')
    @pytest.mark.run(order=8)
    def test_transformer_pynative_cp2_dp2_tp2_zero3(self):
        """
        Feature: test ParallelTransformer pynative
        Description: run pynative mode transformer to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_transformer_with_ring_attention.py"
        device_num = 8

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --use_cp_and_tp_and_zero3"
        cmd = f"msrun --worker_num={device_num} " +\
            f"--local_worker_num={device_num} " +\
            f"--master_port=8122 " +\
            f"--log_dir=msrun_log_tp2cp2dp2_zero3 " +\
            f"--join=True " +\
            f"--cluster_time_out=300 " +\
            f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(
            f"grep -E 'ERROR|error' {sh_path}/use_cp_and_tp_and_zero3/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check use_cp_and_tp_and_zero3/worker_*.log"
