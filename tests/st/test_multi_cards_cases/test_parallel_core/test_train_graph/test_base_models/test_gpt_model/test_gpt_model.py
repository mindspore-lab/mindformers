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
"""Test GPTModel with various configurations"""
import os
from pathlib import Path
import random
import subprocess
import pytest
import numpy as np
from mindformers.tools.logger import logger
from tests.st.test_multi_cards_cases.utils import TaskType
from tests.st.test_multi_cards_cases.test_parallel_core.test_train_graph.test_base_models.test_gpt_model.data_gen_utils import DEFAULT_SEQ_LENGTH, \
    DEFAULT_BATCH_SIZE, DEFAULT_HIDDEN_SIZE, DEFAULT_FFN_HIDDEN_SIZE, DEFAULT_NUM_HEADS


_LEVEL_0_TASK_TIME = 50
_LEVEL_1_TASK_TIME = 0
_TASK_TYPE = TaskType.TWO_CARDS_TASK


TWO_CARD_TEST_PARAM = "model_args, data_keys, expect_error, tensor_parallel"
TWO_CARD_TEST_CASES = [
    # Case 1: DP=1, TP=2 for Norm, SelfAttention, Norm, MLP, num_layers=2
    (
        {
            "input_layernorm": "Norm", "self_attention": "SelfAttention",
            "pre_cross_attn_layernorm": "IdentityOp", "cross_attention": "IdentityOp",
            "pre_mlp_layernorm": "Norm", "mlp": "MLP", "num_layers": 2,
        },
        {"output": "output_default"}, False, 2
    ),
]


def build_msrun_command_list(
        worker_num, local_worker_num, log_dir, run_script_path, model_args, output_path_param,
        tensor_parallel, seq_length=DEFAULT_SEQ_LENGTH, batch_size=DEFAULT_BATCH_SIZE,
        hidden_size=DEFAULT_HIDDEN_SIZE, ffn_hidden_size=DEFAULT_FFN_HIDDEN_SIZE,
        num_attention_heads=DEFAULT_NUM_HEADS):
    """ Build the msrun command with the specified parameters. """
    port_id = int(os.environ.get("ASCEND_PORT_ID", random.randint(50000, 65535)))
    if worker_num == 1:
        cmd_list = ["python"]
    else:
        cmd_list = ["msrun", f"--worker_num={worker_num}", f"--local_worker_num={local_worker_num}",
                    f"--master_port={port_id}", f"--log_dir={log_dir}", "--join=True"]
    cmd_list += [str(run_script_path),
                 f"--output_path={output_path_param}",
                 f"--tensor_parallel={tensor_parallel}",
                 f"--seq_length={model_args.get('seq_length', seq_length)}",
                 f"--batch_size={model_args.get('batch_size', batch_size)}",
                 f"--hidden_size={model_args.get('hidden_size', hidden_size)}",
                 f"--ffn_hidden_size={model_args.get('ffn_hidden_size', ffn_hidden_size)}",
                 f"--num_attention_heads={model_args.get('num_attention_heads', num_attention_heads)}",
                 f"--input_layernorm={model_args['input_layernorm']}",
                 f"--self_attention={model_args['self_attention']}",
                 f"--pre_cross_attn_layernorm={model_args['pre_cross_attn_layernorm']}",
                 f"--cross_attention={model_args['cross_attention']}",
                 f"--pre_mlp_layernorm={model_args['pre_mlp_layernorm']}",
                 f"--mlp={model_args['mlp']}",
                 f"--num_layers={model_args['num_layers']}"
                 ]
    logger.info(f"Equivalent shell command for debugging (approximate): {' '.join(cmd_list)}")
    return cmd_list


class TestTransformerLayer:
    """Test class for GPTModel with different configurations"""
    OUTPUT_MS_FILENAME = "output_gpt_model.npz"  # Must match run_gpt_model.py
    LOG_DIR_NAME = "msrun_log_gpt_model"
    WORKER_LOG_FILENAME_PATTERN = "worker_*.log"  # msrun creates worker_{rank_id}.log

    def setup_method(self):
        """Setup method to prepare test environment"""
        self.sh_path = Path(__file__).parent.resolve()
        self.run_script_path = self.sh_path / "run_gpt_model.py"
        assert self.run_script_path.exists(), f"Run script not found: {self.run_script_path}"

    def check_output_keys(self, output_ms_dict, data_keys):
        """ Check if all expected keys (from data_keys) are present in the output. """
        for expected_key in data_keys.keys():
            assert expected_key in output_ms_dict, \
                f"Expected key '{expected_key}' was not found in the output file. " \
                f"Available keys: {list(output_ms_dict.keys())}"
        logger.info(f"All expected output keys found: {list(data_keys.keys())}")

    def run_test(self, worker_num, local_worker_num, model_args, data_keys, tmp_path,
                 tensor_parallel=1, expect_error=False):
        """Helper function to run test and check results"""
        output_file_path = tmp_path / self.OUTPUT_MS_FILENAME
        log_dir_path = tmp_path / self.LOG_DIR_NAME
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
        result = subprocess.run(cmd_list, shell=False, capture_output=True, text=True, check=False)

        if expect_error:
            assert result.returncode != 0, (
                f"Expected an error but test script passed with exit code {result.returncode}.\n"
                f"Stdout:\n{result.stdout}\nStderr:\n{result.stderr}")
        else:
            assert result.returncode == 0, (
                f"Test script failed with non-zero exit code: {result.returncode}.\n"
                f"Stdout:\n{result.stdout}\nStderr:\n{result.stderr}")
            assert output_file_path.exists(), (
                f"Output file {output_file_path} was not created by the test script.")
            output_ms_dict = dict(np.load(output_file_path))  # Ensure it's a mutable dict
            logger.info(f"Loaded output from {output_file_path}. Keys: {list(output_ms_dict.keys())}")
            self.check_output_keys(output_ms_dict, data_keys)

    @pytest.mark.level0
    @pytest.mark.parametrize(TWO_CARD_TEST_PARAM, TWO_CARD_TEST_CASES)
    def test_multi_card_configurations(self, model_args, data_keys, expect_error, tensor_parallel, tmp_path):
        """Test four cards with various configurations for GPTModel."""
        num_devices = 2
        logger.info(
            f"--- Running Multi-Card ({num_devices} devices) Test: model_args={model_args}, TP={tensor_parallel} ---")
        self.run_test(worker_num=num_devices, local_worker_num=num_devices, model_args=model_args,
                      data_keys=data_keys, expect_error=expect_error, tmp_path=tmp_path,
                      tensor_parallel=tensor_parallel)
