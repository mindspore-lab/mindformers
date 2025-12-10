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


from pathlib import Path
import subprocess

import pytest
import numpy as np
import yaml
from tests.utils.precision_utils import PrecisionChecker
from mindformers.tools.logger import logger


def build_msrun_command_list(log_dir, run_script_path, output_path_param, tensor_parallel,
                             port, quantization, config_yaml_path):
    """ Build the msrun command with the specified parameters. """
    if tensor_parallel == 1:
        cmd_list = ["python"]
    else:
        cmd_list = [
            "msrun",
            f"--worker_num={tensor_parallel}",
            f"--local_worker_num={tensor_parallel}",
            f"--master_port={port}",
            f"--log_dir={log_dir}",
            "--join=True",
        ]

    cmd_list += [
        str(run_script_path),
        f"--config_file={config_yaml_path}",
        f"--output_path={output_path_param}",
        f"--tensor_parallel={tensor_parallel}",
    ]
    if quantization is not None:
        cmd_list.append(f"--quantization={quantization}")

    logger.info(f"Equivalent shell command for debugging (approximate): {' '.join(cmd_list)}")
    return cmd_list


class TestParallelLinear:
    """Test class for ParallelLinear with different configurations"""
    def setup_method(self):
        """Setup method to prepare test environment"""
        self.sh_path = Path(__file__).parent.resolve()
        self.run_script_path = self.sh_path / "simple_mcore.py"
        self.log_file_path = self.sh_path / 'test_output' / 'logs'
        self.log_file_path.mkdir(parents=True, exist_ok=True)

        # Load test configurations from yaml
        self.config_file = self.sh_path / "test_configs.yaml"
        with open(self.config_file, 'r', encoding='utf-8') as f:
            self.configs = yaml.safe_load(f)

    def infer(self, log_dir_path, output_file_path, tensor_parallel, port, quantization, config_yaml_path):
        """Run inference with the specified parameters and check for output file."""
        cmd_list = build_msrun_command_list(
            log_dir=log_dir_path,
            run_script_path=self.run_script_path,
            output_path_param=output_file_path,
            tensor_parallel=tensor_parallel,
            port=port,
            quantization=quantization,
            config_yaml_path=config_yaml_path,
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

    def run_test_from_yaml(self, test_cases_key, tmp_path, tensor_parallel=1, port=8118):
        """Run test based on yaml configurations."""
        test_cases = self.configs[test_cases_key]
        default_precision = self.configs['default_precision']

        # Build precision map: key -> (linear_type, has_bias, quant_policy)
        precision_map = {}
        for case in test_cases:
            linear_type = case['linear_type']
            has_bias = case['has_bias']
            quant_policy = case['quant_policy']
            precision = case.get('precision', default_precision)
            key = (linear_type, has_bias, quant_policy)
            precision_map[key] = precision

        # Run quantized inference
        output_file_path = tmp_path / 'quant-output.npz'
        self.infer(
            log_dir_path=self.log_file_path,
            output_file_path=output_file_path,
            tensor_parallel=tensor_parallel,
            port=port,
            quantization='golden-stick',
            config_yaml_path=self.config_file,
        )
        quant_output = np.load(output_file_path)

        # Run float inference
        output_file_path = tmp_path / 'float-output.npz'
        self.infer(
            log_dir_path=self.log_file_path,
            output_file_path=output_file_path,
            tensor_parallel=tensor_parallel,
            port=port+1,
            quantization=None,
            config_yaml_path=self.config_file,
        )
        float_output = np.load(output_file_path)

        # Check precision for each output
        succeed = True
        for key in quant_output:
            fkey = key[:key.rfind('-')] + '-quant_type_float'
            if fkey not in float_output:
                raise ValueError(f"Diff key in quant_output but not in float_output: {key}")

            # Parse key to get linear_type, has_bias, and quant_policy
            # key format: index_{index}-{linear_type}-has_bias_{has_bias}-compute_dtype_{dtype}-quant_type_{policy}
            parts = key.split('-')
            linear_type = parts[1]
            has_bias_str = parts[2].split('_')[-1]
            if has_bias_str == 'None':
                has_bias = None
            else:
                has_bias = has_bias_str == 'True'
            # Extract quant_policy from "quant_type_POLICY" format
            quant_policy = parts[-1].split('_', 2)[-1]  # Split by '_' and get the last part after 'quant_type_'

            # Get precision config for this specific case
            config_key = (linear_type, has_bias, quant_policy)
            precision = precision_map.get(config_key, default_precision)

            # Create checker with appropriate thresholds
            checker = PrecisionChecker(
                cos_sim_thd=precision['cos_sim_thd'],
                l1_norm_thd=precision['l1_norm_thd'],
                kl_dvg_thd=precision['kl_dvg_thd']
            )

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
    def test_parallel_linear_quantization(self, tmp_path):
        """Test parallel linear layers with various configurations from yaml."""
        self.run_test_from_yaml('test_cases', tmp_path, tensor_parallel=1, port=8888)
