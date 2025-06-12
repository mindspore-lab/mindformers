#  Copyright 2025 Huawei Technologies Co., Ltd
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ============================================================================
"""Test Multi-head Latent Attention (MLA) with various configurations."""
import os
import time
import socket
import subprocess
from pathlib import Path
import pytest
import numpy as np
from mindformers.tools.logger import logger
from tests.utils.double_benchmark import DoubleBenchmarkStandard, DoubleBenchmarkComparator
from tests.st.test_ut.test_parallel_core.test_training_graph.test_transformer.test_multi_latent_attention.data_gen_utils import GOLDEN_DATA, GPU_DATA

BATCH_SIZE = 2
SEQ_LENGTH = 4
HIDDEN_SIZE = 16
SINGLE_CARD_TEST_PARAM = 'model_args, golden_data_key'
SINGLE_CARD_TEST_CASES = [
    (
        # case 1
        # Input:
        #   - model_args: {
        #       'q_lora_rank': 8,        # Enable q_lora_rank with rank 8 for query projection
        #       'use_flash_attn': True,  # Use flash attention optimization
        #       'q_layernorm': 'RMSNorm', # Use RMSNorm for query layer normalization
        #       'k_layernorm': 'RMSNorm'  # Use RMSNorm for key layer normalization
        #     }
        #   - golden_data_key: 'q8_flash_ql_kl' (reference output key)
        # Expected Output: Model output should match reference data stored under 'q8_flash_ql_kl'
        {
            'q_lora_rank': 8,
            'use_flash_attn': True,
            'q_layernorm': 'RMSNorm',
            'k_layernorm': 'RMSNorm'
        },
        'q8_flash_ql_kl'
    ),
    (
        # case 2
        # Input:
        #   - model_args: {
        #       'q_lora_rank': 0,        # Disable q_lora_rank for query projection
        #       'use_flash_attn': True,  # Use flash attention optimization
        #       'q_layernorm': 'RMSNorm', # Use RMSNorm for query layer normalization
        #       'k_layernorm': 'RMSNorm'  # Use RMSNorm for key layer normalization
        #     }
        #   - golden_data_key: 'q0_flash_ql_kl' (reference output key)
        # Expected Output: Model output should match reference data stored under 'q0_flash_ql_kl'
        {
            'q_lora_rank': 0,
            'use_flash_attn': True,
            'q_layernorm': 'RMSNorm',
            'k_layernorm': 'RMSNorm'
        },
        'q0_flash_ql_kl'
    ),
    (
        # case 3
        # Input:
        #   - model_args: {
        #       'q_lora_rank': 8,        # Enable q_lora_rank with rank 8 for query projection
        #       'use_flash_attn': True,  # Use flash attention optimization
        #       'q_layernorm': None,     # Disable query layer normalization
        #       'k_layernorm': 'RMSNorm'  # Use RMSNorm for key layer normalization
        #     }
        #   - golden_data_key: 'q8_flash_kl' (reference output key)
        # Expected Output: Model output should match reference data stored under 'q8_flash_kl'
        {
            'q_lora_rank': 8,
            'use_flash_attn': True,
            'q_layernorm': None,
            'k_layernorm': 'RMSNorm'
        },
        'q8_flash_kl'
    )
]

def get_free_port():
    """Getting a random free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('0.0.0.0', 0))
        _, port = s.getsockname()
    time.sleep(0.05)
    return port


def build_msrun_command_list(
        worker_num, local_worker_num, log_dir, run_script_path, struct,
        batch_size, seq_length, hidden_size,  # Input shape args
        model_args,  # Dictionary of model config args
        output_path_param, tensor_parallel,
        test_name, port
):
    """ Build the msrun command for Multi-head Latent Attention. """
    if worker_num == 1:
        cmd_list = [
            "python",
        ]
    else:
        cmd_list = [
            "msrun",
            f"--worker_num={worker_num}",
            f"--local_worker_num={local_worker_num}",
            f"--master_port={port}",
            f"--log_dir={log_dir}",
            "--join=True"
        ]

    cmd_list.extend([
        str(run_script_path),
        f"--struct={struct}",
        f"--batch_size={batch_size}",
        f"--seq_length={seq_length}",
        f"--hidden_size={hidden_size}",
        f"--q_lora_rank={model_args['q_lora_rank']}",
        f"--use_flash_attn={model_args['use_flash_attn']}",
        f"--q_layernorm={model_args['q_layernorm']}",
        f"--k_layernorm={model_args['k_layernorm']}",
        f"--output_path={output_path_param}",
        f"--tensor_parallel={tensor_parallel}",
        f"--test_name={test_name}"
    ])

    logger.info(f"Test case shell command: {' '.join(cmd_list)}")
    return cmd_list


class TestMultiLatentAttention:
    """Test class for Multi-head Latent Attention"""
    LOG_DIR_NAME = "msrun_mla_log"
    WORKER_LOG_FILENAME = "worker_0.log"

    def setup_method(self):
        """Setup method to prepare test environment"""
        self.current_dir = Path(__file__).parent.resolve()
        self.run_script_path = self.current_dir / "run_multi_latent_attention.py"

    def check_acc(self, output_ms_dict, golden_data_key):
        """
        Compare output_ms with GOLDEN_DATA and GPU_DATA.
        """
        if not golden_data_key:  # Should not happen if expect_error is False
            logger.warning("No golden_data_key provided for accuracy check.")
            return

        dtype_str = "bfloat16"
        if "fp16" in golden_data_key:
            dtype_str = "float16"

        standard = DoubleBenchmarkStandard(dtype=dtype_str)

        npu_data = output_ms_dict.get("output")
        golden_data = GOLDEN_DATA.get(golden_data_key)
        gpu_data = GPU_DATA.get(golden_data_key)

        DoubleBenchmarkComparator.check_pass_or_not(
            npu_data=npu_data,
            gpu_data=gpu_data,
            golden_data=golden_data,
            standard=standard
        )

    def run_mla_test(
            self,
            worker_num,
            local_worker_num,
            struct,
            model_args,
            golden_data_key,
            tmp_path,
            tensor_parallel=1,
            port=8118
    ):
        """Helper function to run MLA test and check results"""
        output_file_path = tmp_path / f"output_mla_ms_{struct}_{'single' if tensor_parallel == 1 else 'multi'}.npz"
        log_dir_path = tmp_path / self.LOG_DIR_NAME
        log_dir_path.mkdir(parents=True, exist_ok=True)
        worker_log_file = log_dir_path / self.WORKER_LOG_FILENAME

        cmd_list = build_msrun_command_list(
            worker_num=worker_num,
            local_worker_num=local_worker_num,
            log_dir=log_dir_path,
            run_script_path=self.run_script_path,
            struct=struct,
            batch_size=BATCH_SIZE,
            seq_length=SEQ_LENGTH,
            hidden_size=HIDDEN_SIZE,
            model_args=model_args,
            output_path_param=output_file_path,
            tensor_parallel=tensor_parallel,
            test_name=golden_data_key,
            port=port
        )
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = "0"
        result = subprocess.run(cmd_list, shell=False, capture_output=True, text=True, check=False, env=env)

        if worker_num != 1:
            assert worker_log_file.exists()
        assert result.returncode == 0, (
            "Multi-head Latent Attention script failed with non-zero exit code: "
            f"{result.returncode}.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"
        )
        assert output_file_path.exists(), (
            f"Output file {output_file_path} was not created."
        )

        if tensor_parallel == 1:
            output_ms_dict = np.load(output_file_path)
            self.check_acc(output_ms_dict, golden_data_key)
        else:
            if struct == 'a2':
                prefix, mode_num = tmp_path.name.split('mode')
                mega_output_dir = prefix + 'mode' + str(int(mode_num) - 1)
                mega_output_file = tmp_path.parent / mega_output_dir / "output_mla_ms_megatron_multi.npz"
                output_ms_dict_mega = np.load(mega_output_file, allow_pickle=True)['output']
                output_ms_dict_mind = np.load(output_file_path, allow_pickle=True)['output']
                assert np.allclose(output_ms_dict_mind, output_ms_dict_mega)

class TestMultiLatentAttentionSingleCard(TestMultiLatentAttention):
    """Test class for Multi-head Latent Attention on single card"""
    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.parametrize('struct', ['megatron', 'a2'])
    @pytest.mark.parametrize(
        SINGLE_CARD_TEST_PARAM,
        SINGLE_CARD_TEST_CASES
    )
    def test_single_card_mla_cases(
            self,
            struct,
            model_args,
            golden_data_key,
            tmp_path
    ):
        """Test Multi-head Latent Attention on single card with various configurations."""
        self.run_mla_test(
            worker_num=1,
            local_worker_num=1,
            struct=struct,
            model_args=model_args,
            golden_data_key=golden_data_key,
            tmp_path=tmp_path
        )
