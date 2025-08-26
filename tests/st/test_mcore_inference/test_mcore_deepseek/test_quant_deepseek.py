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
"""Test ColumnParallelLinear with various configurations"""
import os
import socket
import time
import pytest

def get_available_port(start=10000, end=11000):
    """get_available_port"""
    def is_port_available(port_):
        """is_port_available"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect(('localhost', port_))
                return True
            except ConnectionRefusedError:
                return False

    for port in range(start, end):
        if not is_port_available(port):
            return port
    return start

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
@pytest.mark.parametrize("quant_algo", ['A8W8'])
def test_quant_deepseek_level0(quant_algo):
    """
    Feature: test deepseek inference of different quantization algorithms.
    Description: 4-layer quantized DeepSeek network runs on 2 cards.
    Expectation: The first 10 output tokens are as expected.
    """
    run_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quant_deepseek_predict.py")
    port = get_available_port()
    os.system(f"kill -9 $(lsof -i:{port} | " + "awk '{print $2}')")
    time.sleep(1.0)
    return_code = os.system(
        f"msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 "
        f"--master_port={port} --join=True --log_dir=./test_quant_deepseek_{quant_algo}_2p_logs "
        f"python {run_file} -a {quant_algo}"
    )
    if return_code != 0:
        log_file = open(f"./test_quant_deepseek_{quant_algo}_2p_logs/worker_0.log", "r", encoding="utf-8")
        for line in log_file:
            print(line, flush=True)
        log_file.close()
    os.system("ps -u | grep 'quant_deepseek_predict' | grep -v grep | awk -F ' ' '{print$2}' | xargs kill -9")
    os.system(f"kill -9 $(lsof -i:{port} | " + "awk '{print $2}')")
    time.sleep(1.0)
    assert return_code == 0
