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
"""Test VocabParallelCrossEntropy with various configurations"""

from pathlib import Path
import subprocess
import pytest
import numpy as np
from data_gen_utils import GOLDEN_DATA, GPU_DATA
from tests.utils.double_benchmark import DoubleBenchmarkStandard, DoubleBenchmarkComparator

TOTAL_VOCAB_SIZE = 1024
BATCH_SIZE = 4
SEQ_LENGTH = 4

SINGLE_CARD_TEST_PARAM = "model_args, data_keys, expect_error"
SINGLE_CARD_TEST_CASES = [
    # Test Case 1: Single Card, check_for_nan=False, calculate_per_token=True
    (
        {"check_for_nan_in_loss_and_grad": False, "calculate_per_token_loss": True},
        {"numerator": "numerator", "denominator": "denominator"},
        False,
    ),
    # Test Case 2: Single Card, check_for_nan=True, calculate_per_token=True
    (
        {"check_for_nan_in_loss_and_grad": True, "calculate_per_token_loss": True},
        {"numerator": "numerator", "denominator": "denominator"},
        False,
    ),
]

FOUR_CARD_TEST_PARAM = "model_args, data_keys, expect_error, tensor_parallel"
FOUR_CARD_TEST_CASES = [
    # Test Case 3: Four Cards (DP=2, TP=2), check_for_nan=False, calculate_per_token=True
    (
        {"check_for_nan_in_loss_and_grad": False, "calculate_per_token_loss": True},
        {"numerator": "numerator", "denominator": "denominator"},
        False,
        2,
    ),
]


def build_msrun_command_list(
        worker_num,
        local_worker_num,
        log_dir,
        run_script_path,
        vocab_size,
        batch_size,
        seq_length,
        check_for_nan_in_loss_and_grad,
        calculate_per_token_loss,
        output_path_param,
        tensor_parallel,
    ):
    """Build the msrun command with the specified parameters for VocabParallelCrossEntropy."""
    if tensor_parallel == 1:
        cmd_list = ["python"]
    else:
        cmd_list = [
            "msrun",
            f"--worker_num={worker_num}",
            f"--local_worker_num={local_worker_num}",
            "--master_port=8167",
            f"--log_dir={log_dir}",
            "--join=True",]
    cmd_list += [
        str(run_script_path),
        f"--vocab_size={vocab_size}",
        f"--batch_size={batch_size}",
        f"--seq_length={seq_length}",
        f"--check_for_nan_in_loss_and_grad={str(check_for_nan_in_loss_and_grad).lower()}",
        f"--calculate_per_token_loss={str(calculate_per_token_loss).lower()}",
        f"--output_path={output_path_param}",
        f"--tensor_parallel={tensor_parallel}",
    ]
    print(f"Equivalent shell command for debugging (approximate): {' '.join(cmd_list)}")
    return cmd_list


class TestVocabParallelCrossEntropy:
    """Test class for VocabParallelCrossEntropy with different configurations"""

    OUTPUT_MS_FILENAME = "output_ms_loss.npz"
    LOG_DIR_NAME = "msrun_log_loss"
    WORKER_LOG_FILENAME = "worker_0.log"

    def setup_method(self):
        """Setup method to prepare test environment"""
        self.sh_path = Path(__file__).parent.resolve()
        self.run_script_path = self.sh_path / "run_vocab_parallel_cross_entropy.py"

    def check_acc(self, output_ms_dict, data_keys):
        """
        Compare output_ms with GOLDEN_DATA and GPU_DATA using DoubleBenchmarkComparator.
        """
        standard = DoubleBenchmarkStandard(dtype="float32")

        for key, data_key in data_keys.items():
            assert key in output_ms_dict, f"Key '{key}' not found in MindSpore output."
            npu_data = output_ms_dict.get(key)

            assert data_key in GOLDEN_DATA, f"Golden data key '{data_key}' not found."
            golden_data = GOLDEN_DATA.get(data_key)

            gpu_data = GPU_DATA.get(data_key)

            DoubleBenchmarkComparator.check_pass_or_not(
                npu_data=npu_data, gpu_data=gpu_data, golden_data=golden_data, standard=standard
            )

    def check_output_keys(self, output_ms_dict, model_args):
        """Check if the expected keys are present in the output based on model_args."""
        output_keys = list(output_ms_dict.keys())
        calculate_per_token_loss = model_args["calculate_per_token_loss"]

        if calculate_per_token_loss:
            assert "numerator" in output_keys, (
                f"The 'numerator' key is expected when calculate_per_token_loss is True. Found keys: {output_keys}"
            )
            assert "denominator" in output_keys, (
                f"The 'denominator' key is expected when calculate_per_token_loss is True. Found keys: {output_keys}"
            )
            assert "loss" not in output_keys, (
                f"The 'loss' key is NOT expected when calculate_per_token_loss is True. Found keys: {output_keys}"
            )
        else:
            assert "loss" in output_keys, (
                f"The 'loss' key is expected when calculate_per_token_loss is False. Found keys: {output_keys}"
            )
            assert "numerator" not in output_keys, (
                f"The 'numerator' key is NOT expected when calculate_per_token_loss is False. Found keys: {output_keys}"
            )
            assert "denominator" not in output_keys, (
                f"The 'denominator' key is NOT expected when calculate_per_token_loss is False. "
                f"Found keys: {output_keys}"
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
            vocab_size=TOTAL_VOCAB_SIZE,
            batch_size=BATCH_SIZE,
            seq_length=SEQ_LENGTH,
            check_for_nan_in_loss_and_grad=model_args["check_for_nan_in_loss_and_grad"],
            calculate_per_token_loss=model_args["calculate_per_token_loss"],
            output_path_param=output_file_path,
            tensor_parallel=tensor_parallel,
        )

        result = subprocess.run(cmd_list, shell=False, capture_output=True, text=True, check=False)

        if expect_error:
            assert result.returncode != 0, (
                f"Expected an error but test script passed. Stdout:\n{result.stdout}\nStderr:\n{result.stderr}"
            )
        else:
            assert result.returncode == 0, (
                f"Test script failed with non-zero exit code: "
                f"{result.returncode}.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"
            )
            assert output_file_path.exists(), f"Output file {output_file_path} was not created."

            output_ms_dict = np.load(output_file_path)
            self.check_output_keys(output_ms_dict, model_args)
            self.check_acc(output_ms_dict, data_keys)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.parametrize(
        SINGLE_CARD_TEST_PARAM,
        SINGLE_CARD_TEST_CASES
    )
    def test_single_card_cases(self, model_args, data_keys, expect_error, tmp_path):
        """Test single card with various configurations."""
        self.run_test(
            worker_num=1,
            local_worker_num=1,
            model_args=model_args,
            data_keys=data_keys,
            expect_error=expect_error,
            tmp_path=tmp_path,
            tensor_parallel=1,
        )

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.parametrize(
        FOUR_CARD_TEST_PARAM,
        FOUR_CARD_TEST_CASES
    )
    def test_four_cards_case(self, model_args, data_keys, expect_error, tensor_parallel, tmp_path):
        """Test four cards with various configurations."""
        self.run_test(
            worker_num=4,
            local_worker_num=4,
            model_args=model_args,
            data_keys=data_keys,
            expect_error=expect_error,
            tmp_path=tmp_path,
            tensor_parallel=tensor_parallel,
        )
