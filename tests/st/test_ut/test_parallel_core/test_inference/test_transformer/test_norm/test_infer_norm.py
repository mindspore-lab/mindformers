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
"""mcore norm UT of inference"""
from pathlib import Path
import subprocess
import pytest
import numpy as np

from mindformers.tools.logger import logger

from tests.utils.double_benchmark import DoubleBenchmarkStandard, DoubleBenchmarkComparator
from tests.st.test_ut.test_parallel_core.test_inference.test_transformer.test_norm.data_gen_utils import (
    get_init_params,
    GPU_DATA,
    GOLDEN_DATA,
)


LAYERNORM_SINGLE_CARD_TEST_PARAM = "model_args, data_keys, expect_error"
LAYERNORM_SINGLE_CARD_TEST_CASES = [
    (
        # 并行策略: 单卡, batch_size: 2, seq_length: 2, hidden_size: 32, eps: 1e-5
        # expected result: 功能跑通, 精度对齐。
        {"batch_size": 2, "seq_length": 2, "hidden_size": 32, "eps": 1e-5},
        {"output": "layernorm_output1"},
        False
    ),
    (
        # 并行策略: 单卡, batch_size: 2, seq_length: 2, hidden_size: 32, eps: 0.1
        # expected result: 功能跑通, 精度对齐。
        {"batch_size": 2, "seq_length": 2, "hidden_size": 32, "eps": 0.1},
        {"output": "layernorm_output2"},
        False
    )
]


def build_msrun_command_list(
        worker_num, local_worker_num, port, log_dir, run_script_path, module,
        batch_size, seq_length, hidden_size, eps, output_path_param: str = None
):
    """ Build the msrun command with the specified parameters. """
    cmd_list = [
        "msrun",
        f"--worker_num={worker_num}",
        f"--local_worker_num={local_worker_num}",
        f"--master_port={port}",
        f"--log_dir={log_dir}",
        "--join=True",
        str(run_script_path),
        f"--module={module}",
        f"--batch_size={batch_size}",
        f"--seq_length={seq_length}",
        f"--hidden_size={hidden_size}",
        f"--eps={eps}",
        f"--output_path={output_path_param}",
    ]
    logger.info(f"Equivalent shell command for debugging (approximate): {' '.join(cmd_list)}")
    return cmd_list


class TestInferLayerNorm:
    """Test class for LayerNorm with different configurations"""
    OUTPUT_MS_FILENAME = "output_ms_layernorm.npz"
    LOG_DIR_NAME = "msrun_log"
    WORKER_LOG_FILENAME = "worker_0.log"

    def setup_method(self):
        """Setup method to prepare test environment"""
        self.sh_path = Path(__file__).parent.resolve()
        self.run_script_path = self.sh_path / "run_infer_norm.py"

    @staticmethod
    def check_function(output_ms_dict, model_args, data_keys):
        """
        Compare the shapes of output_ms and input_ms whether they are the same.
        """
        for key, _ in data_keys.items():
            output_data = output_ms_dict.get(key)
            params = get_init_params(model_args["batch_size"], model_args["seq_length"], model_args["hidden_size"])
            input_data = params["input"]

            assert output_data.shape == input_data.shape, \
                (f"The shapes of output data and input data are different, "
                 f"got output shape: {output_data.shape} and input shape: {input_data.shape}")

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

    def check_result(
            self,
            output_file_path,
            worker_log_file,
            model_args,
            data_keys,
            result,
            expect_error
    ):
        """Helper function to check results"""
        assert worker_log_file.exists()

        if expect_error:
            assert result.returncode != 0, (
                f"Expected an error but test script passed. "
                f"Stdout:\n{result.stdout}\n"
                f"Stderr:\n{result.stderr}"
            )
        else:
            assert result.returncode == 0, (
                f"Test script failed with non-zero exit code: "
                f"{result.returncode}.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"
            )
            assert output_file_path.exists(), (
                f"Output file {output_file_path} was not created."
            )

            output_ms_dict = np.load(output_file_path)

            # check whether the function of rotary embedding module works properly.
            self.check_function(output_ms_dict, model_args, data_keys)
            # Check whether the accuracy is consistent with vLLM
            self.check_acc(output_ms_dict, data_keys)

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
            port=10000,
            log_dir=log_dir_path,
            run_script_path=self.run_script_path,
            module="LayerNorm",
            batch_size=model_args["batch_size"],
            seq_length=model_args["seq_length"],
            hidden_size=model_args["hidden_size"],
            eps=model_args["eps"],
            output_path_param=output_file_path,
        )

        cmd_result = subprocess.run(
            cmd_list, shell=False, capture_output=True, text=True, check=False)

        self.check_result(output_file_path, worker_log_file, model_args, data_keys, cmd_result, expect_error)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.parametrize(
        LAYERNORM_SINGLE_CARD_TEST_PARAM,
        LAYERNORM_SINGLE_CARD_TEST_CASES
    )
    def test_single_card_configurations(self, model_args, data_keys, expect_error, tmp_path):
        """Test single card with various configurations."""
        self.run_test(
            worker_num=1, local_worker_num=1,
            model_args=model_args, expect_error=expect_error,
            data_keys=data_keys,
            tmp_path=tmp_path
        )
