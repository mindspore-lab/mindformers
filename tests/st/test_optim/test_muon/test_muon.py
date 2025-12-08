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
"""
Test module for testing the Muon optimizer interface used for MindFormers.
How to run this:
pytest tests/st/test_optim/test_muon/test_muon.py
"""
from pathlib import Path
import subprocess
import pytest
import numpy as np

from tests.st.test_optim.test_muon.data_utils import (
    BASELINE_LOSSES_NESTEROV_TRUE,
    BASELINE_LOSSES_NESTEROV_FALSE,
    BASELINE_LOSSES_DIFF_LR,
    compare_losses,
    DEFAULT_RTOL,
    DEFAULT_ATOL,
)

from mindformers.tools.logger import logger

# Test parameters definition
SINGLE_CARD_TEST_CASES = [
    # Default config with nesterov=True
    {
        "learning_rate": 0.02,
        "weight_decay": 0.1,
        "momentum": 0.95,
        "nesterov": True,
        "num_steps": 20,
        "baseline_losses": BASELINE_LOSSES_NESTEROV_TRUE,
    },
    # Config without Nesterov momentum
    {
        "learning_rate": 0.02,
        "weight_decay": 0.1,
        "momentum": 0.95,
        "nesterov": False,
        "num_steps": 20,
        "baseline_losses": BASELINE_LOSSES_NESTEROV_FALSE,
    },
    # Config with different learning rate
    {
        "learning_rate": 0.01,
        "weight_decay": 0.05,
        "momentum": 0.9,
        "nesterov": True,
        "num_steps": 20,
        "baseline_losses": BASELINE_LOSSES_DIFF_LR,
    },
]


def build_msrun_command_list(
        worker_num,
        local_worker_num,
        log_dir,
        run_script_path,
        learning_rate,
        weight_decay,
        momentum,
        nesterov,
        num_steps,
        output_path,
        port=29500
    ):
    """Build the msrun command with the specified parameters."""
    cmd_list = [
        "msrun",
        f"--worker_num={worker_num}",
        f"--local_worker_num={local_worker_num}",
        f"--master_port={port}",
        f"--log_dir={log_dir}",
        "--join=True",
        str(run_script_path),
        f"--learning_rate={learning_rate}",
        f"--weight_decay={weight_decay}",
        f"--momentum={momentum}",
        f"--nesterov={str(nesterov).lower()}",
        f"--num_steps={num_steps}",
        f"--output_path={output_path}",
    ]
    logger.info(f"Equivalent shell command for Muon test: {' '.join(cmd_list)}")
    return cmd_list


class TestMuon:
    """Test class for Muon optimizer with different configurations."""
    OUTPUT_FILENAME = "output_muon.npz"
    LOG_DIR_NAME = "msrun_log"

    def setup_method(self):
        """Setup method to prepare test environment."""
        self.sh_path = Path(__file__).parent.resolve()
        self.run_script_path = self.sh_path / "run_muon.py"

    def check_results(self, output_dict, baseline_losses=None):
        """
        Check the output results from the Muon optimizer run.

        Args:
            output_dict: Dictionary containing the output results
            num_params: Expected number of parameters
            baseline_losses: Expected baseline losses for comparison
        """
        # Check losses
        losses = output_dict.get("losses")
        assert losses is not None, "Losses not found in output"
        assert len(losses) > 0, "Losses array is empty"
        assert not np.any(np.isnan(losses)), "Losses contain NaN values"
        assert not np.any(np.isinf(losses)), "Losses contain Inf values"

        # Compare with baseline if provided
        if baseline_losses is not None:
            assert compare_losses(losses, baseline_losses, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL), (
                f"Losses do not match baseline.\n"
                f"Actual: {losses}\n"
                f"Expected: {baseline_losses}\n"
                f"Max diff: {np.max(np.abs(losses - baseline_losses))}"
            )

    def run_test(
            self,
            worker_num,
            local_worker_num,
            optimizer_args,
            tmp_path,
            port=29500,
            baseline_losses=None
        ):
        """Helper function to run test and check results."""
        output_file_path = tmp_path / self.OUTPUT_FILENAME
        log_dir_path = tmp_path / self.LOG_DIR_NAME
        log_dir_path.mkdir(parents=True, exist_ok=True)

        cmd_list = build_msrun_command_list(
            worker_num=worker_num,
            local_worker_num=local_worker_num,
            log_dir=log_dir_path,
            run_script_path=self.run_script_path,
            learning_rate=optimizer_args["learning_rate"],
            weight_decay=optimizer_args["weight_decay"],
            momentum=optimizer_args["momentum"],
            nesterov=optimizer_args["nesterov"],
            num_steps=optimizer_args["num_steps"],
            output_path=output_file_path,
            port=port
        )

        result = subprocess.run(
            cmd_list, shell=False, capture_output=True, text=True, check=False
        )

        assert result.returncode == 0, (
            f"Test script failed with non-zero exit code: "
            f"{result.returncode}.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"
        )
        assert output_file_path.exists(), (
            f"Output file {output_file_path} was not created."
        )

        output_dict = np.load(output_file_path)
        self.check_results(output_dict, baseline_losses=baseline_losses)

        return output_dict


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
class TestMuonSingleCard(TestMuon):
    """Test class for Muon optimizer with single card configurations."""

    @pytest.mark.parametrize("optimizer_args", SINGLE_CARD_TEST_CASES)
    def test_muon_single_card(self, optimizer_args, tmp_path):
        """
        Feature: Muon optimizer training
        Description: Test computation of Muon optimizer with various configurations.
        Expectation: Training completes successfully with valid losses matching baseline
        """
        baseline_losses = optimizer_args.get("baseline_losses")
        self.run_test(
            worker_num=1,
            local_worker_num=1,
            optimizer_args=optimizer_args,
            tmp_path=tmp_path,
            baseline_losses=baseline_losses
        )
