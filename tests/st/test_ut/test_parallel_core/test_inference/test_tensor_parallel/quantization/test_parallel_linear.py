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


from typing import Optional
from pathlib import Path
import subprocess
import pytest
import numpy as np
from tests.utils.precision_utils import PrecisionChecker
from mindformers.tools.logger import logger


def build_msrun_command_list(linear_types, log_dir, run_script_path, output_path_param, tensor_parallel,
                             port, quantization, quant_policies:Optional[list]=None):
    """ Build the msrun command with the specified parameters. """
    if tensor_parallel == 1:
        cmd_list = ["python"]
    else:
        cmd_list = [
            "msrun",
            f"--worker_num={tensor_parallel}",
            f"--local_worker_num={tensor_parallel}",
            f"--master_port={port}", # Ensure port is unique per test run if parallelized at pytest level
            f"--log_dir={log_dir}",
            "--join=True",
        ]

    cmd_list += [
        str(run_script_path),
        f"--output_path={output_path_param}",
        f"--tensor_parallel={tensor_parallel}",
    ]
    for linear_type in linear_types:
        cmd_list.append(f"--linear_types={linear_type}")
    for quant_policy in quant_policies:
        cmd_list.append(f"--quant_policies={quant_policy}")
    if quantization is not None:
        cmd_list.append(f"--quantization={quantization}")
        if quant_policies is None:
            raise RuntimeError("quant_policies must be provided when quantization is enabled.")

    logger.info(f"Equivalent shell command for debugging (approximate): {' '.join(cmd_list)}")
    return cmd_list


class TestParallelLinear:
    """Test class for ParallelLinear with different configurations"""
    def setup_method(self):
        """Setup method to prepare test environment"""
        self.sh_path = Path(__file__).parent.resolve()
        self.run_script_path = self.sh_path / "run_parallel_linear.py"
        self.log_file_path = self.sh_path / 'test_output' / 'logs'
        self.log_file_path.mkdir(parents=True, exist_ok=True)

    def infer(self, linear_types, log_dir_path, output_file_path, tensor_parallel, port, quantization,
              quant_policies=None):
        """Run inference with the specified parameters and check for output file."""
        cmd_list = build_msrun_command_list(
            linear_types=linear_types,
            log_dir=log_dir_path,
            run_script_path=self.run_script_path,
            output_path_param=output_file_path,
            tensor_parallel=tensor_parallel,
            port=port,
            quantization=quantization,
            quant_policies=quant_policies,
        )

        result = subprocess.run(
            cmd_list, shell=False, capture_output=True, text=True, check=False)

        assert result.returncode == 0, (
            f"Test script failed with non-zero exit code: "
            f"{result.returncode}.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"
        )
        assert output_file_path.exists(), (
            f"Output file {output_file_path} was not created."
        )

    def run_test(self, linear_types, quant_policies, tmp_path, tensor_parallel=1, port=8118):
        """Helper function to run test and check results"""
        output_file_path = tmp_path / 'quant-output.npz'
        self.infer(
            linear_types=linear_types,
            log_dir_path=self.log_file_path,
            output_file_path=output_file_path,
            tensor_parallel=tensor_parallel,
            port=port,
            quantization='golden-stick',
            quant_policies=quant_policies,
        )
        quant_output = np.load(output_file_path)

        output_file_path = tmp_path / 'float-output.npz'
        self.infer(
            linear_types=linear_types,
            log_dir_path=self.log_file_path,
            output_file_path=output_file_path,
            tensor_parallel=tensor_parallel,
            port=port+1,
            quantization=None,
            quant_policies=quant_policies,
        )
        float_output = np.load(output_file_path)
        checker = PrecisionChecker()
        succeed = True
        for key in quant_output:
            fkey = key[:key.rfind('-')] + '-quant_type_float'
            if fkey not in float_output:
                raise ValueError(f"Diff key in quant_output but not in float_output: {key}")
            try:
                checker.check_precision(float_output[fkey], quant_output[key])
                print(f"Check precision for {key} succeed", flush=True)
            except AssertionError as e:
                print(f"Check precision for {key} failed: {e}", flush=True)
                succeed = False
        assert succeed, "Some precision check failed"

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_single_card_configurations(self, tmp_path):
        """Test single card with various configurations."""
        linear_types = ["ColumnParallelLinear", "RowParallelLinear"]
        quant_policies = ["a8w8", "a8dynw8"]
        self.run_test(linear_types=linear_types, quant_policies=quant_policies,
                      tmp_path=tmp_path, port=8888)
