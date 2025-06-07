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
"""mcore Dopt_Attn UT of inference"""
import subprocess
from pathlib import Path
import random
import numpy as np
import pytest

from mindformers.tools.logger import logger
from tests.utils.double_benchmark import DoubleBenchmarkStandard, DoubleBenchmarkComparator
from tests.st.test_ut.test_parallel_core.test_inference.test_transformer.test_dotproduct_attention.data_gen_utils import (
    get_init_params,
    BATCH_SIZE,
    SEQ_LENGTH,
    NUM_HEADS,
    HIDDEN_SIZE,
    GOLDEN_DATA,
    GPU_DATA
)

DPA_SINGLE_CARD_TEST_PARAM = "model_args, data_keys, expect_error"
DPA_SINGLE_CARD_TEST_CASES = [
    (
        # 并行策略: 单卡, batch_size: 2, seq_length: 2, num_heads: 2,
        # num_query_groups: 1, hidden_size: 32
        # expected result: 功能跑通。
        {"batch_size": BATCH_SIZE, "seq_length": SEQ_LENGTH, "num_heads": NUM_HEADS,
         "num_query_groups": 1, "hidden_size": HIDDEN_SIZE},
        {"output": "output_1"},
        False
    ),
    (
        # 并行策略: 单卡, batch_size: 2, seq_length: 1, num_heads: 2,
        # num_query_groups: 2, hidden_size: 32
        # expected result: 功能跑通。
        {"batch_size": BATCH_SIZE, "seq_length": 1, "num_heads": NUM_HEADS,
         "num_query_groups": 2, "hidden_size": HIDDEN_SIZE},
        {"output": "output_2"},
        False
    )
]


def generate_random_port(start, end):
    """ Get random port."""
    port = random.randint(start, end)
    return port


def build_msrun_command_list(
        worker_num, local_worker_num, log_dir, run_script_path, batch_size,
        seq_length, num_heads, num_query_groups, hidden_size, output_path_param: str = None
):
    """ Build the msrun command with the specified parameters. """
    if worker_num == 1:
        cmd_list = ["python"]
    else:
        cmd_list = [
            "msrun",
            f"--worker_num={worker_num}",
            f"--local_worker_num={local_worker_num}",  # Should match NPU cards available
            f"--master_port={generate_random_port(10100, 10200)}", # Ensure port is unique per test run if parallelized at pytest level
            f"--log_dir={log_dir}",
            "--join=True"]
    cmd_list += [
        str(run_script_path),
        f"--batch_size={batch_size}",
        f"--seq_length={seq_length}",
        f"--num_heads={num_heads}",
        f"--num_query_groups={num_query_groups}",
        f"--hidden_size={hidden_size}",
        f"--output_path={output_path_param}",
    ]
    logger.info(f"Equivalent shell command for debugging (approximate): {' '.join(cmd_list)}")
    return cmd_list


class TestDotProductAttention:
    """Test class for DotProduct Attention with different configurations"""
    OUTPUT_MS_FILENAME = "output_ms_dpa.npz"
    LOG_DIR_NAME = "msrun_log"
    WORKER_LOG_FILENAME = "worker_0.log"

    def setup_method(self):
        """Setup method to prepare test environment"""
        self.sh_path = Path(__file__).parent.resolve()
        self.run_script_path = self.sh_path / "run_infer_dotproduct_attention.py"

    @staticmethod
    def check_function(output_ms_dict, model_args, data_keys):
        """
        Compare the shapes of output_ms and input_ms whether they are the same.
        """
        for key, _ in data_keys.items():
            output_data = output_ms_dict.get(key)
            query = get_init_params(num_kv_heads=model_args["num_query_groups"])["query"]

            assert np.array_equal(output_data.shape, query.shape), \
                (f"The shapes of output data and input data are different, "
                 f"got output shape: {output_data.shape} and input shape: {query.shape}")

    @staticmethod
    def check_acc(output_ms_dict, data_keys):
        """Compare output_ms with GOLDEN_DATA and GPU_DATA using DoubleBenchmarkComparator."""
        standard = DoubleBenchmarkStandard(dtype="bfloat16")

        for key, data_key in data_keys.items():
            npu_data = output_ms_dict.get(key)
            golden_data = GOLDEN_DATA.get(data_key)
            gpu_data = GPU_DATA.get(data_key)

            DoubleBenchmarkComparator.check_pass_or_not(
                npu_data=npu_data,
                gpu_data=gpu_data,
                golden_data=golden_data,
                standard=standard
                )

    def run_test(
            self,
            worker_num,
            local_worker_num,
            model_args,
            data_keys,
            tmp_path,
            expect_error=False,
    ):
        """Helper function to run test"""
        output_file_path = tmp_path / self.OUTPUT_MS_FILENAME
        log_dir_path = tmp_path / self.LOG_DIR_NAME
        log_dir_path.mkdir(parents=True, exist_ok=True)
        worker_log_file = log_dir_path / self.WORKER_LOG_FILENAME

        cmd_list = build_msrun_command_list(
            worker_num=worker_num,
            local_worker_num=local_worker_num,
            log_dir=log_dir_path,
            run_script_path=self.run_script_path,
            batch_size=model_args["batch_size"],
            seq_length=model_args["seq_length"],
            num_heads=model_args["num_heads"],
            num_query_groups=model_args["num_query_groups"],
            hidden_size=model_args["hidden_size"],
            output_path_param=output_file_path,
        )

        cmd_result = subprocess.run(
            cmd_list, shell=False, capture_output=True, text=True, check=False)

        if worker_num > 1:
            assert worker_log_file.exists()

        if expect_error:
            assert cmd_result.returncode != 0, (
                f"Expected an error but test script passed. "
                f"Stdout:\n{cmd_result.stdout}\n"
                f"Stderr:\n{cmd_result.stderr}\n"
            )
        else:
            assert cmd_result.returncode == 0, (
                f"Test script failed with non-zero exit code: "
                f"{cmd_result.returncode}.\nStdout:\n{cmd_result.stdout}\nStderr:\n{cmd_result.stderr}\n"
            )
            assert output_file_path.exists(), (
                f"Output file {output_file_path} was not created."
            )

        ouput_ms_dict = np.load(output_file_path)

        self.check_function(ouput_ms_dict, model_args, data_keys)

        self.check_acc(ouput_ms_dict, data_keys)

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.parametrize(
        DPA_SINGLE_CARD_TEST_PARAM,
        DPA_SINGLE_CARD_TEST_CASES
    )
    def test_single_card_configurations(self, model_args, data_keys, expect_error, tmp_path):
        """Test single card with various configurations."""
        self.run_test(
            worker_num=1, local_worker_num=1,
            model_args=model_args, expect_error=expect_error,
            data_keys=data_keys,
            tmp_path=tmp_path
        )
