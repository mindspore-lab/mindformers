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
"""Test TransformerLayer with various configurations"""
from pathlib import Path
import subprocess
import pytest
import numpy as np

from mindformers.tools.logger import logger
from tests.utils.double_benchmark import DoubleBenchmarkStandard, DoubleBenchmarkComparator

from .data_gen_utils import GOLDEN_DATA, GPU_DATA, \
    DEFAULT_SEQ_LENGTH, DEFAULT_BATCH_SIZE, DEFAULT_HIDDEN_SIZE, \
    DEFAULT_FFN_HIDDEN_SIZE, DEFAULT_NUM_HEADS


SINGLE_CARD_TEST_PARAM = "model_args, data_keys, expect_error"
SINGLE_CARD_TEST_CASES = [
    # Case 1: Standard Norm, SelfAttention, Norm, MLP
    (
        {
            "input_layernorm": "Norm", "self_attention": "SelfAttention",
            "pre_cross_attn_layernorm": "IdentityOp", "cross_attention": "IdentityOp",
            "pre_mlp_layernorm": "Norm", "mlp": "MLP"
        },
        {"output": "output_default", "extra_loss": "extra_loss_default"},  # Expect output and extra_loss
        False
    ),
]




def build_msrun_command_list(
        worker_num, local_worker_num, log_dir, run_script_path,
        model_args,  # This will be a dictionary
        output_path_param, tensor_parallel,
        # Default dimensions, can be overridden by model_args if specific tests need them
        seq_length=DEFAULT_SEQ_LENGTH, batch_size=DEFAULT_BATCH_SIZE,
        hidden_size=DEFAULT_HIDDEN_SIZE, ffn_hidden_size=DEFAULT_FFN_HIDDEN_SIZE,
        num_attention_heads=DEFAULT_NUM_HEADS
):
    """ Build the msrun command with the specified parameters. """
    if worker_num == 1:
        cmd_list = ["python"]
    else:
        cmd_list = [
            "msrun",
            f"--worker_num={worker_num}",
            f"--local_worker_num={local_worker_num}",  # Should match NPU cards available
            "--master_port=8167",  # Ensure port is unique per test run if parallelized at pytest level
            f"--log_dir={log_dir}",
            "--join=True"]
    cmd_list += [str(run_script_path),
                 f"--output_path={output_path_param}",
                 f"--tensor_parallel={tensor_parallel}",
                 # Add model dimensions
                 f"--seq_length={model_args.get('seq_length', seq_length)}",
                 f"--batch_size={model_args.get('batch_size', batch_size)}",
                 f"--hidden_size={model_args.get('hidden_size', hidden_size)}",
                 f"--ffn_hidden_size={model_args.get('ffn_hidden_size', ffn_hidden_size)}",
                 f"--num_attention_heads={model_args.get('num_attention_heads', num_attention_heads)}",
                 # Add submodule choices
                 f"--input_layernorm={model_args['input_layernorm']}",
                 f"--self_attention={model_args['self_attention']}",
                 f"--pre_cross_attn_layernorm={model_args['pre_cross_attn_layernorm']}",
                 f"--cross_attention={model_args['cross_attention']}",
                 f"--pre_mlp_layernorm={model_args['pre_mlp_layernorm']}",
                 f"--mlp={model_args['mlp']}",
                 ]
    # Log the approximate command for debugging
    # Note: For environment variables like RANK_ID, MS_WORKER_NUM, they are set by msrun itself.
    logger.info(f"Equivalent shell command for debugging (approximate): {' '.join(cmd_list)}")
    return cmd_list


class TestTransformerLayer:
    """Test class for TransformerLayer with different configurations"""
    OUTPUT_MS_FILENAME = "output_transformer_layer.npz"  # Must match run_transformer_layer.py
    LOG_DIR_NAME = "msrun_log_transformer_layer"
    WORKER_LOG_FILENAME_PATTERN = "worker_*.log"  # msrun creates worker_{rank_id}.log

    def setup_method(self):
        """Setup method to prepare test environment"""
        self.sh_path = Path(__file__).parent.resolve()
        self.run_script_path = self.sh_path / "run_transformer_layer.py"
        assert self.run_script_path.exists(), f"Run script not found: {self.run_script_path}"

    def check_acc(self, output_ms_dict, data_keys):
        """
        Compare output_ms with GOLDEN_DATA and GPU_DATA using DoubleBenchmarkComparator.
        """
        # Using bfloat16 as the dtype for comparison standard, similar to DataParallelLinear example
        # The actual data from run_script is saved as float32 after conversion from bfloat16.
        standard = DoubleBenchmarkStandard(dtype="bfloat16")  # Relaxed threshold

        for key_in_output, key_in_golden in data_keys.items():
            if key_in_output not in output_ms_dict:
                logger.warning(f"Key '{key_in_output}' not found in MindSpore output. Skipping accuracy check for it.")
                continue

            npu_data = output_ms_dict.get(key_in_output)
            golden_data = GOLDEN_DATA.get(key_in_golden)
            gpu_data = GPU_DATA.get(key_in_golden)  # Assuming GPU_DATA uses same keys

            if npu_data is None:  # Should not happen if key is in output_ms_dict
                pytest.fail(f"NPU data for key '{key_in_output}' is None.")

            if golden_data is None:
                pytest.fail(f"Golden data for key '{key_in_golden}' is None. Check data_gen_utils.py.")

            logger.info(f"Checking accuracy for key: '{key_in_output}' (maps to golden key '{key_in_golden}')")
            logger.info(f"NPU data shape: {npu_data.shape}, dtype: {npu_data.dtype}")
            logger.info(f"Golden data shape: {golden_data.shape}, dtype: {golden_data.dtype}")
            if gpu_data is not None:
                logger.info(f"GPU data shape: {gpu_data.shape}, dtype: {gpu_data.dtype}")

            # Basic shape check before detailed comparison
            assert npu_data.shape == golden_data.shape, \
                f"Shape mismatch for '{key_in_output}': NPU {npu_data.shape}, Golden {golden_data.shape}"

            # The DoubleBenchmarkComparator might have specific requirements for dtypes.
            # Since run_script saves as float32, ensure comparison is appropriate.
            DoubleBenchmarkComparator.check_pass_or_not(
                npu_data=npu_data.astype(np.float32),  # Ensure float32 for comparison
                gpu_data=gpu_data.astype(np.float32) if gpu_data is not None else None,
                golden_data=golden_data.astype(np.float32),
                standard=standard,
                # name=key_in_output # Optional: for more descriptive error messages from comparator
            )
            logger.info(f"Accuracy check passed for key: '{key_in_output}'")

    def check_output_keys(self, output_ms_dict, data_keys):
        """ Check if all expected keys (from data_keys) are present in the output. """
        for expected_key in data_keys.keys():
            assert expected_key in output_ms_dict, \
                f"Expected key '{expected_key}' was not found in the output file. " \
                f"Available keys: {list(output_ms_dict.keys())}"
        logger.info(f"All expected output keys found: {list(data_keys.keys())}")

    def run_test(
            self,
            worker_num,
            local_worker_num,
            model_args,  # Dictionary of model arguments
            data_keys,  # Dictionary mapping output keys to golden data keys
            tmp_path,  # Pytest fixture for temporary directory
            tensor_parallel=1,
            expect_error=False,
    ):
        """Helper function to run test and check results"""
        output_file_path = tmp_path / self.OUTPUT_MS_FILENAME
        log_dir_path = tmp_path / self.LOG_DIR_NAME
        # log_dir_path.mkdir(parents=True, exist_ok=True) # msrun creates log_dir

        cmd_list = build_msrun_command_list(
            worker_num=worker_num,
            local_worker_num=local_worker_num,
            log_dir=log_dir_path,
            run_script_path=self.run_script_path,
            model_args=model_args,
            output_path_param=output_file_path,
            tensor_parallel=tensor_parallel,
        )

        logger.info(f"Running command: {' '.join(cmd_list)}")
        result = subprocess.run(
            cmd_list, shell=False, capture_output=True, text=True, check=False,
        )

        if expect_error:
            assert result.returncode != 0, (
                f"Expected an error but test script passed with exit code {result.returncode}.\n"
                f"Stdout:\n{result.stdout}\n"
                f"Stderr:\n{result.stderr}"
            )
            logger.info("Test failed as expected.")
        else:
            assert result.returncode == 0, (
                f"Test script failed with non-zero exit code: {result.returncode}.\n"
                f"Stdout:\n{result.stdout}\n"
                f"Stderr:\n{result.stderr}"
            )
            assert output_file_path.exists(), (
                f"Output file {output_file_path} was not created by the test script."
            )

            output_ms_dict = dict(np.load(output_file_path))  # Ensure it's a mutable dict
            logger.info(f"Loaded output from {output_file_path}. Keys: {list(output_ms_dict.keys())}")

            self.check_output_keys(output_ms_dict, data_keys)

            logger.info("Test passed successfully.")

class TestTransformerLayerSingleCard(TestTransformerLayer):
    """Test TransformerLayer with single card configurations"""
    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training  # Or your specific platform
    @pytest.mark.env_onecard  # Single card environment
    @pytest.mark.parametrize(SINGLE_CARD_TEST_PARAM, SINGLE_CARD_TEST_CASES)
    def test_single_card_configurations(self, model_args, data_keys, expect_error, tmp_path):
        """Test single card with various configurations for TransformerLayer."""
        logger.info(f"--- Running Single Card Test: model_args={model_args} ---")
        self.run_test(
            worker_num=1, local_worker_num=1,
            model_args=model_args,
            data_keys=data_keys,
            expect_error=expect_error,
            tmp_path=tmp_path,
            tensor_parallel=1
        )
