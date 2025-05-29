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
"""Test Multi-Token Prediction (MTP) with various configurations."""
import os
import socket
import subprocess
import time
from pathlib import Path
import pytest
import numpy as np
from mindformers.tools.logger import logger
from tests.utils.double_benchmark import DoubleBenchmarkStandard, DoubleBenchmarkComparator
from tests.st.test_ut.test_parallel_core.test_training_graph.test_transformer.test_multi_token_prediction.data_gen_utils import GOLDEN_DATA, GPU_DATA

BATCH_SIZE = 2
SEQ_LENGTH = 4
HIDDEN_SIZE = 16
SINGLE_CARD_TEST_PARAM = 'golden_data_key'
SINGLE_CARD_TEST_CASES = [
    (
        # case 1
        # The single-card mtp baseline.
        # expected result: The results from NPU, GPU, and CPU comply with the dual-benchmark verification standard.
        'single_card_baseline'
    ),
]

MULTI_CARD_TEST_PARAM = 'golden_data_key, tensor_parallel'
MULTI_CARD_TEST_CASES = [
    (
        # case 1
        # The multi-card mtp baseline (dp=2, tp=2).
        # expected result: The results of multi-card test should comply with the results of single-card test.
        'multi_card_baseline',
        2
    ),
]

def get_free_port():
    """Getting a random free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('0.0.0.0', 0))
        _, port = s.getsockname()
    time.sleep(0.05)
    return port

def build_msrun_command_list(
        worker_num, local_worker_num, log_dir, run_script_path,
        batch_size, seq_length, hidden_size,  # Input shape args
        output_path_param, tensor_parallel,
        test_name
):
    """ Build the msrun command for Multi-Token Prediction (MTP). """
    if worker_num == 1:
        cmd_list = [
            "python",
        ]
    else:
        cmd_list = [
            "msrun",
            f"--worker_num={worker_num}",
            f"--local_worker_num={local_worker_num}",
            f"--master_port={get_free_port()}",
            f"--log_dir={log_dir}",
            "--join=True"
        ]

    cmd_list.extend([
        str(run_script_path),
        f"--batch_size={batch_size}",
        f"--seq_length={seq_length}",
        f"--hidden_size={hidden_size}",
        f"--output_path={output_path_param}",
        f"--tp={tensor_parallel}",
        f"--test_name={test_name}"
    ])

    logger.info(f"Test case shell command: {' '.join(cmd_list)}")
    return cmd_list


class TestMTP:
    """Test class for Multi-Token Prediction (MTP)"""
    LOG_DIR_NAME = "msrun_mtp_log"
    WORKER_LOG_FILENAME = "worker_0.log"

    def setup_method(self):
        """Setup method to prepare test environment"""
        self.current_dir = Path(__file__).parent.resolve()
        self.run_script_path = self.current_dir / "run_multi_token_prediction.py"

    def check_acc(self, output_ms_dict, golden_data_key):
        """
        Compare output_ms with GOLDEN_DATA and GPU_DATA.
        """
        if not golden_data_key:  # Should not happen if expect_error is False
            logger.warning("No golden_data_key provided for accuracy check.")
            return

        dtype_str = "bfloat16"
        if "fp16" in golden_data_key:
            dtype_str = "float16"

        standard = DoubleBenchmarkStandard(dtype=dtype_str)

        npu_data = output_ms_dict.get('mtp_loss')
        golden_data = GOLDEN_DATA.get(golden_data_key)
        gpu_data = GPU_DATA.get(golden_data_key)

        DoubleBenchmarkComparator.check_pass_or_not(
            npu_data=npu_data,
            gpu_data=gpu_data,
            golden_data=golden_data,
            standard=standard
        )

    def run_mtp_test(
            self,
            worker_num,
            local_worker_num,
            golden_data_key,
            tmp_path,
            tensor_parallel=1,
    ):
        """Helper function to run MTP test and check results"""
        output_file_path = tmp_path / f"output_mtp_ms_{'single' if tensor_parallel==1 else 'multi'}.npz"
        log_dir_path = tmp_path / self.LOG_DIR_NAME
        log_dir_path.mkdir(parents=True, exist_ok=True)
        worker_log_file = log_dir_path / self.WORKER_LOG_FILENAME

        cmd_list = build_msrun_command_list(
            worker_num=worker_num,
            local_worker_num=local_worker_num,
            log_dir=log_dir_path,
            run_script_path=self.run_script_path,
            batch_size=BATCH_SIZE,
            seq_length=SEQ_LENGTH,
            hidden_size=HIDDEN_SIZE,
            output_path_param=output_file_path,
            tensor_parallel=tensor_parallel,
            test_name=golden_data_key
        )
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = "0"
        result = subprocess.run(cmd_list, shell=False, capture_output=True, text=True, check=False, env=env)

        if worker_num != 1:
            assert worker_log_file.exists()
        assert result.returncode == 0, (
            "Multi-Token Prediction (MTP) script failed with non-zero exit code: "
            f"{result.returncode}.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"
        )
        assert output_file_path.exists(), (
            f"Output file {output_file_path} was not created."
        )

        if worker_num == 1:
            single_card_output_dict = np.load(output_file_path)
            self.check_acc(single_card_output_dict, golden_data_key)
        else:
            multi_card_mtp_loss = np.load(output_file_path)['mtp_loss']
            prefix = tmp_path.parent
            single_card_output_file = prefix / 'test_single_card_mtp_cases_sin0' / 'output_mtp_ms_single.npz'
            assert single_card_output_file.exists(), ('Single card test must be run first.')
            single_card_mtp_loss = np.load(single_card_output_file, allow_pickle=True)['mtp_loss']
            assert np.allclose(multi_card_mtp_loss, single_card_mtp_loss)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.parametrize(
        SINGLE_CARD_TEST_PARAM,
        SINGLE_CARD_TEST_CASES
    )
    def test_single_card_mtp_cases(
            self,
            golden_data_key,
            tmp_path
    ):
        """Test Multi-Token Prediction (MTP) on single card with various configurations."""
        self.run_mtp_test(
            worker_num=1,
            local_worker_num=1,
            golden_data_key=golden_data_key,
            tmp_path=tmp_path
        )

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.parametrize(
        MULTI_CARD_TEST_PARAM,
        MULTI_CARD_TEST_CASES
    )
    def test_multi_card_mtp_cases(
            self,
            golden_data_key,
            tensor_parallel,
            tmp_path
    ):
        """Test Multi-Token Prediction (MTP) on multiple cards with various configurations."""
        self.run_mtp_test(
            worker_num=4,
            local_worker_num=4,
            golden_data_key=golden_data_key,
            tensor_parallel=tensor_parallel,
            tmp_path=tmp_path
        )
