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
"""Test CrossEntropyLoss with various configurations"""

from pathlib import Path
import subprocess
import pytest
import numpy as np
from tests.utils.double_benchmark import DoubleBenchmarkStandard, DoubleBenchmarkComparator


TOTAL_VOCAB_SIZE = 1024
BATCH_SIZE = 4
SEQ_LENGTH = 8

SINGLE_CARD_TEST_PARAM = "model_args, data_keys"
SINGLE_CARD_TEST_CASES = [
    # Test Case 1: Single Card, calculate_per_token=False
    (
        {"calculate_per_token_loss": False},
        {"loss": "loss", "grad": "grad"},
    ),
    # Test Case 2: Single Card, calculate_per_token=True
    (
        {"calculate_per_token_loss": True},
        {"numerator": "numerator", "denominator": "denominator"},
    ),
]

def build_msrun_command_list(
        run_script_path,
        vocab_size,
        batch_size,
        seq_length,
        calculate_per_token_loss,
        output_path_param,
    ):
    """Build the msrun command with the specified parameters for VocabParallelCrossEntropy."""
    cmd_list = [
        "python",
        str(run_script_path),
        f"--vocab_size={vocab_size}",
        f"--batch_size={batch_size}",
        f"--seq_length={seq_length}",
        f"--calculate_per_token_loss={str(calculate_per_token_loss).lower()}",
        f"--output_path={output_path_param}",
    ]
    print(f"Equivalent shell command for debugging (approximate): {' '.join(cmd_list)}")
    return cmd_list


class TestCrossEntropyLoss:
    """Test class for CrossEntropyLoss with different configurations"""

    OUTPUT_PYNATIVE_FILENAME = "output_pynative_loss.npz"
    OUTPUT_STATIC_FILENAME = "output_static_loss.npz"
    OUTPUT_CPU_FILENAME = "output_cpu_loss.npz"

    def setup_method(self):
        """Setup method to prepare test environment"""
        self.sh_path = Path(__file__).parent.resolve()
        self.run_script_path = self.sh_path / "run_cross_entropy_loss.py"

    def check_acc(self, output_pynative_dict, output_static_dict, output_cpu_dict, data_keys):
        """
        Compare output using DoubleBenchmarkComparator.
        """
        standard = DoubleBenchmarkStandard(dtype="float32")

        for key, data_key in data_keys.items():
            assert key in output_pynative_dict, f"Key '{key}' not found in MindSpore output."
            assert data_key in output_cpu_dict, f"Golden data key '{data_key}' not found."
            npu_data = output_pynative_dict.get(key)
            golden_data = output_static_dict.get(data_key)
            gpu_data = output_cpu_dict.get(data_key)

            DoubleBenchmarkComparator.check_pass_or_not(
                npu_data=npu_data, gpu_data=gpu_data, golden_data=golden_data, standard=standard
            )

    def run_test(
            self,
            model_args,
            data_keys,
            tmp_path,
        ):
        """Helper function to run test and check results"""
        output_file_path = tmp_path

        cmd_list = build_msrun_command_list(
            run_script_path=self.run_script_path,
            vocab_size=TOTAL_VOCAB_SIZE,
            batch_size=BATCH_SIZE,
            seq_length=SEQ_LENGTH,
            calculate_per_token_loss=model_args["calculate_per_token_loss"],
            output_path_param=output_file_path,
        )

        result = subprocess.run(cmd_list, shell=False, capture_output=True, text=True, check=False)

        assert result.returncode == 0, (
            f"Test script failed with non-zero exit code: "
            f"{result.returncode}.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"
        )
        assert output_file_path.exists(), f"Output file {output_file_path} was not created."
        output_pynative_dict = np.load(tmp_path / self.OUTPUT_PYNATIVE_FILENAME)
        output_static_dict = np.load(tmp_path / self.OUTPUT_STATIC_FILENAME)
        output_cpu_dict = np.load(tmp_path / self.OUTPUT_CPU_FILENAME)
        self.check_acc(output_pynative_dict, output_static_dict, output_cpu_dict, data_keys)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.parametrize(
        SINGLE_CARD_TEST_PARAM,
        SINGLE_CARD_TEST_CASES
    )
    def test_single_card_cases(self, model_args, data_keys, tmp_path):
        """Test single card with various configurations."""
        self.run_test(
            model_args=model_args,
            data_keys=data_keys,
            tmp_path=tmp_path,
        )
