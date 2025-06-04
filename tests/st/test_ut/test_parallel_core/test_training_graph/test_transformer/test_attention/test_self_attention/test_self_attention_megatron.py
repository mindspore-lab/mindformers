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
"""Test SelfAttentionMegatron with various configurations"""
from pathlib import Path
import subprocess
import pytest
import numpy as np
from mindformers.tools.logger import logger
from tests.utils.double_benchmark import DoubleBenchmarkStandard, DoubleBenchmarkComparator

from .data_gen_utils import GOLDEN_DATA, GPU_DATA

SEQ_LEN = 2
BATCH_SIZE = 2
HIDDEN_SIZE = 32
NUM_HEAD = 8
KV_HIDDEN_SIZE = HIDDEN_SIZE // NUM_HEAD  # 32

SINGLE_CARD_TEST_PARAM = "model_args, data_keys, expect_error"
SINGLE_CARD_TEST_CASES = [
    (
        {"use_flash_attn": True, "num_query_groups": 4, "q_layernorm": None, "k_layernorm": None},
        {"output": "output_query_group_4", "bias": "bias_query_group_4"},
        False
    ),
    (
        {"use_flash_attn": True, "num_query_groups": 8, "q_layernorm": None, "k_layernorm": None},
        {"output": "output_query_group_8", "bias": "bias_query_group_8"},
        False
    ),
    (
        {"use_flash_attn": True, "num_query_groups": 4, "q_layernorm": "Norm", "k_layernorm": "Norm"},
        {"output": "output_query_group_4", "bias": "bias_query_group_4"},
        False
    ),
]


def build_msrun_command_list(
        worker_num, local_worker_num, log_dir, run_script_path,
        seq_len, batch_size, hidden_size, kv_hidden_size, num_attention_heads,
        has_bias, use_flash_attn, num_query_groups,
        q_layernorm, k_layernorm, output_path_param, tensor_parallel
):
    """Build the msrun command with the specified parameters."""
    if worker_num == 1:
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
        f"--seq_len={seq_len}",
        f"--batch_size={batch_size}",
        f"--hidden_size={hidden_size}",
        f"--kv_hidden_size={kv_hidden_size}",
        f"--num_attention_heads={num_attention_heads}",
        f"--has_bias={str(has_bias).lower()}",
        f"--use_flash_attn={str(use_flash_attn).lower()}",
        f"--num_query_groups={num_query_groups}",
        f"--q_layernorm={q_layernorm if q_layernorm else 'None'}",
        f"--k_layernorm={k_layernorm if k_layernorm else 'None'}",
        f"--output_path={output_path_param}",
        f"--tensor_parallel={tensor_parallel}"
    ]
    logger.info(f"Equivalent shell command for debugging (approximate): {' '.join(cmd_list)}")
    return cmd_list


class TestSelfAttentionMegatron:
    """Test class for SelfAttentionMegatron with different configurations"""
    OUTPUT_MS_FILENAME = "output_ms.npz"
    LOG_DIR_NAME = "msrun_log"

    def setup_method(self):
        """Setup method to prepare test environment"""
        self.sh_path = Path(__file__).parent.resolve()
        self.run_script_path = self.sh_path / "run_self_attention_megatron.py"

    def check_acc(self, output_ms_dict, data_keys):
        """Compare output_ms with GOLDEN_DATA and GPU_DATA using DoubleBenchmarkComparator."""
        standard = DoubleBenchmarkStandard(dtype="bfloat16")
        for key, data_key in data_keys.items():
            npu_data = output_ms_dict.get(key)
            golden_data = GOLDEN_DATA.get(data_key)
            gpu_data = GPU_DATA.get(data_key)
            DoubleBenchmarkComparator.check_pass_or_not(
                npu_data=npu_data,
                gpu_data=gpu_data,
                golden_data=golden_data,
                standard=standard
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
            seq_len=SEQ_LEN,
            batch_size=BATCH_SIZE,
            hidden_size=HIDDEN_SIZE,
            kv_hidden_size=KV_HIDDEN_SIZE,
            num_attention_heads=NUM_HEAD,
            has_bias=True,
            use_flash_attn=model_args["use_flash_attn"],
            num_query_groups=model_args["num_query_groups"],
            q_layernorm=model_args["q_layernorm"],
            k_layernorm=model_args["k_layernorm"],
            output_path_param=output_file_path,
            tensor_parallel=tensor_parallel
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

            self.check_acc(output_ms_dict, data_keys)
class TestSelfAttentionMegatronSingleCard(TestSelfAttentionMegatron):
    """Test SelfAttentionMegatron with single card configurations"""
    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.parametrize(
        SINGLE_CARD_TEST_PARAM,
        SINGLE_CARD_TEST_CASES
    )
    def test_single_card_configurations(self, model_args, data_keys, expect_error, tmp_path):
        """Test single card with various configurations."""
        self.run_test(
            worker_num=1, local_worker_num=1,
            model_args=model_args, data_keys=data_keys,
            expect_error=expect_error, tmp_path=tmp_path
        )
