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
"""Test train apis"""

import os
import pytest


def run_get_optimizer(config_path, master_port, device_num, group_params=False, case_name="get_optimizer"):
    """Run get optimizer"""
    os.environ["HCCL_BUFFSIZE"] = "50"
    scripts_name = "run_get_optimizer.py"

    sh_path = os.path.split(os.path.realpath(__file__))[0]
    scripts_path = os.path.join(sh_path, scripts_name)
    log_dir = f"msrun_{case_name}"

    cmd = (
        f"msrun --worker_num={device_num} "
        + f"--local_worker_num={device_num} "
        + f"--master_port={master_port} "
        + f"--log_dir={log_dir} "
        + "--join=True "
        + "--cluster_time_out=300 "
        + f"python {scripts_path} --config_path {config_path} --group_params {group_params}"
    )
    ret = os.system(cmd)
    os.system(f"grep -E 'ERROR|error' {sh_path}/{log_dir}/worker_0.log -C 3")
    assert ret == 0, f"msrun failed, please check {log_dir}/worker_*.log"


def run_train_test(config_path, master_port, device_num, case_name="train"):
    """Run train test"""
    os.environ["HCCL_BUFFSIZE"] = "50"
    scripts_name = "run_train.py"

    sh_path = os.path.split(os.path.realpath(__file__))[0]
    scripts_path = os.path.join(sh_path, scripts_name)
    log_dir = f"msrun_{case_name}"

    scripts_cmd = f"{scripts_path} --config_path {config_path}"
    cmd = (
        f"msrun --worker_num={device_num} "
        + f"--local_worker_num={device_num} "
        + f"--master_port={master_port} "
        + f"--log_dir={log_dir} "
        + "--join=True "
        + "--cluster_time_out=300 "
        + f"{scripts_cmd}"
    )
    ret = os.system(cmd)
    os.system(f"grep -E 'ERROR|error' {sh_path}/{log_dir}/worker_0.log -C 3")
    assert ret == 0, f"msrun failed, please check {log_dir}/worker_*.log"


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
class TestTrain:
    """A test class for testing train apis"""

    @pytest.mark.run(order=1)
    def test_adamw(self):
        """
        Feature: test get adamw optimizer
        Description: run get optimizer with adam
        Expectation: test success
        """
        config_path = "./adamw.yaml"
        master_port = "8118"
        device_num = 1
        run_get_optimizer(config_path, master_port, device_num, case_name="adamw")

    @pytest.mark.run(order=2)
    def test_adam(self):
        """
        Feature: test get adam optimizer
        Description: run get optimizer with adam and group params
        Expectation: test success
        """
        config_path = "./adam.yaml"
        master_port = "8128"
        device_num = 1
        run_get_optimizer(config_path, master_port, device_num, case_name="adam")

    @pytest.mark.run(order=3)
    def test_adamw_zero_group_params(self):
        """
        Feature: test get zero adamw optimizer with group params
        Description: run get optimizer with adam and group params
        Expectation: test success
        """
        config_path = "./adamw_zero.yaml"
        master_port = "8138"
        device_num = 2
        run_get_optimizer(config_path, master_port, device_num, group_params=True, case_name="adamw_zero_group_params")

    @pytest.mark.run(order=4)
    def test_sgd(self):
        """
        Feature: test get sgd optimizer
        Description: run get optimizer with adam and group params
        Expectation: test success
        """
        config_path = "./sgd.yaml"
        master_port = "8148"
        device_num = 1
        run_get_optimizer(config_path, master_port, device_num, case_name="sgd")

    @pytest.mark.run(order=5)
    def test_came(self):
        """
        Feature: test get came optimizer
        Description: run get optimizer with adam and group params
        Expectation: test success
        """
        config_path = "./came.yaml"
        master_port = "8158"
        device_num = 2
        run_get_optimizer(config_path, master_port, device_num, case_name="came")

    @pytest.mark.run(order=6)
    def test_train_zero(self):
        """
        Feature: test train with zero
        Description: run training in tp 2 dp 2
        Expectation: test success
        """
        config_path = "./train.yaml"
        master_port = "8168"
        device_num = 4
        run_train_test(config_path, master_port, device_num, case_name="train_zero")

    @pytest.mark.run(order=7)
    def test_train_eval_save(self):
        """
        Feature: test train with eval and save
        Description: run training in tp 2 dp 2
        Expectation: test success
        """
        config_path = "./train_with_save_and_eval.yaml"
        master_port = "8178"
        device_num = 4
        run_train_test(config_path, master_port, device_num, case_name="train_eval_save")
