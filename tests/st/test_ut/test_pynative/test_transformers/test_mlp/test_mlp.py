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
"""Test MLP with various configurations"""
from pathlib import Path
import subprocess
import pytest

from mindformers.tools.logger import logger


SINGLE_CARD_TEST_PARAM = "model_args, expect_error"
SINGLE_CARD_TEST_CASES = [
    (
        {
            "input_size": None,
            "hidden_act": "silu",
            "add_bias_linear": False,
            "gated_linear_unit": False,
            "compute_dtype": "bfloat16",
            "params_dtype": "float32",
            "enable_backward": True
        },
        False
    ),
    (
        {
            "input_size": None,
            "hidden_act": "silu",
            "add_bias_linear": False,
            "gated_linear_unit": True,
            "compute_dtype": "bfloat16",
            "params_dtype": "float32",
            "enable_backward": False
        },
        False
    ),
    (
        {
            "input_size": None,
            "hidden_act": "fusedswiglu",
            "add_bias_linear": True,
            "gated_linear_unit": True,
            "compute_dtype": "bfloat16",
            "params_dtype": "float32",
            "enable_backward": False
        },
        False
    ),
    (
        {
            "input_size": 32,
            "hidden_act": "fusedswiglu",
            "add_bias_linear": True,
            "gated_linear_unit": True,
            "compute_dtype": "bfloat16",
            "params_dtype": "float32",
            "enable_backward": False
        },
        False
    ),
    (
        {
            "input_size": None,
            "hidden_act": "silu",
            "add_bias_linear": False,
            "gated_linear_unit": True,
            "compute_dtype": "bfloat16",
            "params_dtype": "bfloat16",
            "enable_backward": False
        },
        False
    ),
    (
        {
            "input_size": None,
            "hidden_act": "silu",
            "add_bias_linear": False,
            "gated_linear_unit": True,
            "compute_dtype": "float32",
            "params_dtype": "float32",
            "enable_backward": False
        },
        False
    ),
    (
        {
            "input_size": None,
            "hidden_act": None,
            "add_bias_linear": False,
            "gated_linear_unit": False,
            "compute_dtype": "bfloat16",
            "params_dtype": "float32",
            "enable_backward": False
        },
        True
    ),
    (
        {
            "input_size": None,
            "hidden_act": "swiglu",
            "add_bias_linear": False,
            "gated_linear_unit": False,
            "compute_dtype": "bfloat16",
            "params_dtype": "float32",
            "enable_backward": False
        },
        True
    ),
]


def build_command_list(run_script_path, model_args):
    """Build the command with the specified parameters."""
    cmd_list = ["python", str(run_script_path)]

    if model_args.get("input_size") is not None:
        cmd_list.append(f"--input_size={model_args['input_size']}")

    if model_args.get("hidden_act") is not None:
        cmd_list.append(f"--hidden_act={model_args['hidden_act']}")

    if model_args.get("add_bias_linear", False):
        cmd_list.append("--add_bias_linear")

    if model_args.get("gated_linear_unit", False):
        cmd_list.append("--gated_linear_unit")

    if model_args.get("compute_dtype") is not None:
        cmd_list.append(f"--compute_dtype={model_args['compute_dtype']}")

    if model_args.get("params_dtype") is not None:
        cmd_list.append(f"--params_dtype={model_args['params_dtype']}")

    if model_args.get("enable_backward", False):
        cmd_list.append("--enable_backward")

    logger.info(f"Equivalent shell command for debugging (approximate): {' '.join(cmd_list)}")
    return cmd_list


class TestMLP:
    """Test class for MLP with different configurations"""

    def setup_method(self):
        """Setup method to prepare test environment"""
        self.sh_path = Path(__file__).parent.resolve()
        self.run_script_path = self.sh_path / "run_mlp.py"

    def run_test(self, model_args, expect_error=False):
        """Helper function to run test and check results"""
        cmd_list = build_command_list(self.run_script_path, model_args)

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


class TestMLPSingleCard(TestMLP):
    """Test class for MLP with single card configurations"""

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.parametrize(
        SINGLE_CARD_TEST_PARAM,
        SINGLE_CARD_TEST_CASES
    )
    def test_single_card_configurations(self, model_args, expect_error):
        """Test single card with various configurations."""
        self.run_test(model_args=model_args, expect_error=expect_error)
