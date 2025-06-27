"""Test ColumnParallelLinear with various configurations"""
from pathlib import Path
import subprocess
import pytest
import numpy as np

from mindformers.tools.logger import logger
from tests.st.test_ut.test_parallel_core.test_inference.test_tensor_parallel.test_column_parallel_linear.data_gen_utils import LEGACY_DATA
from tests.utils.precision_utils import PrecisionChecker


INPUT_SIZE = 32
OUTPUT_SIZE = 32

SINGLE_CARD_TEST_PARAM = "model_args, data_keys, expect_error"
SINGLE_CARD_TEST_CASES = [
    (
        {"bias": False, "skip_bias_add": False, "skip_weight_param_allocation": False, "use_weight_tensor": False},
        {"output": "output_only"},
        False
    ),
    (
        {"bias": True, "skip_bias_add": False, "skip_weight_param_allocation": False, "use_weight_tensor": False},
        {"output": "output_with_bias"},
        False
    ),
    (
        {"bias": True, "skip_bias_add": False, "skip_weight_param_allocation": False, "use_weight_tensor": True},
        {"output": "output_use_weight"},
        False
    ),
    (
        {"bias": True, "skip_bias_add": False, "skip_weight_param_allocation": True, "use_weight_tensor": True},
        {"output": "output_use_weight"},
        False
    ),
]


def build_msrun_command_list(
        worker_num, local_worker_num, log_dir, run_script_path,
        input_size, output_size,
        bias, skip_bias_add, skip_weight_param_allocation, use_weight_tensor,
        output_path_param, tensor_parallel, port
    ):
    """ Build the msrun command with the specified parameters. """
    if worker_num == 1 and local_worker_num == 1:
        cmd_list = ["python"]
    else:
        cmd_list = [
            "msrun",
            f"--worker_num={worker_num}",
            f"--local_worker_num={local_worker_num}",
            f"--master_port={port}", # Ensure port is unique per test run if parallelized at pytest level
            f"--log_dir={log_dir}",
            "--join=True",
        ]
    cmd_list += [
        str(run_script_path),
        f"--input_size={input_size}",
        f"--output_size={output_size}",
        f"--bias={str(bias).lower()}",
        f"--skip_bias_add={str(skip_bias_add).lower()}",
        f"--skip_weight_param_allocation={str(skip_weight_param_allocation).lower()}",
        f"--use_weight_tensor={str(use_weight_tensor).lower()}",
        f"--output_path={output_path_param}",
        f"--tensor_parallel={tensor_parallel}"
    ]
    logger.info(f"Equivalent shell command for debugging (approximate): {' '.join(cmd_list)}")
    return cmd_list


class TestColumnParallelLinear:
    """Test class for ColumnParallelLinear with different configurations"""
    OUTPUT_MS_FILENAME = "output_ms.npz"
    LOG_DIR_NAME = "msrun_log"
    def setup_method(self):
        """Setup method to prepare test environment"""
        self.sh_path = Path(__file__).parent.resolve()
        self.run_script_path = self.sh_path / "run_column_parallel_linear.py"

    def check_acc(self, output_ms_dict, data_keys):
        """
        Compare output_ms with GOLDEN_DATA and GPU_DATA using DoubleBenchmarkComparator.
        """
        checker = PrecisionChecker()

        for key, data_key in data_keys.items():
            npu_data = output_ms_dict.get(key).astype(np.float32)
            golden_data = LEGACY_DATA.get(data_key).astype(np.float32)
            checker.check_precision(golden_data, npu_data)

    def check_output_keys(self, output_ms_dict, expected_bias_key_present):
        """ Check if the 'bias' key is present or absent as expected in the output. """
        output_keys = output_ms_dict.keys()
        if expected_bias_key_present:
            assert "bias" in output_keys, (
                f"The 'bias' key is expected in the output "
                f"dictionary but was not found. Keys: {output_keys}"
            )
        else:
            assert "bias" not in output_keys, (
                f"The 'bias' key is not expected in the output "
                f"dictionary but was found. Keys: {output_keys}"
            )

    def run_test(
            self,
            worker_num,
            local_worker_num,
            model_args,
            data_keys,
            tmp_path,
            tensor_parallel=1,
            expect_error=False,
            port=8118
        ):
        """Helper function to run test and check results"""
        output_file_path = tmp_path / self.OUTPUT_MS_FILENAME
        log_dir_path = tmp_path / self.LOG_DIR_NAME
        log_dir_path.mkdir(parents=True, exist_ok=True)

        cmd_list = build_msrun_command_list(
            worker_num=worker_num,
            local_worker_num=local_worker_num,
            log_dir=log_dir_path,
            run_script_path=self.run_script_path,
            input_size=INPUT_SIZE,
            output_size=OUTPUT_SIZE,
            bias=model_args["bias"],
            skip_bias_add=model_args["skip_bias_add"],
            skip_weight_param_allocation=model_args["skip_weight_param_allocation"],
            use_weight_tensor=model_args["use_weight_tensor"],
            output_path_param=output_file_path,
            tensor_parallel=tensor_parallel,
            port=port
        )

        result = subprocess.run(
            cmd_list, shell=False, capture_output=True, text=True, check=False)

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
            should_bias_key_be_present = model_args["bias"] and model_args["skip_bias_add"]

            self.check_output_keys(output_ms_dict, should_bias_key_be_present)
            self.check_acc(output_ms_dict, data_keys)


class TestColumnParallelLinearSingleCard(TestColumnParallelLinear):
    """Test class for ColumnParallelLinear with single card configurations"""
    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.parametrize(
        SINGLE_CARD_TEST_PARAM,
        SINGLE_CARD_TEST_CASES
    )
    def test_single_card_configurations(self, model_args, expect_error, data_keys, tmp_path):
        """Test single card with various configurations."""
        self.run_test(
            worker_num=1, local_worker_num=1,
            model_args=model_args, expect_error=expect_error,
            data_keys=data_keys,
            tmp_path=tmp_path
        )
