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
"""Test CausalMaskGenerate with various configurations"""
from pathlib import Path
import subprocess
import pytest
import numpy as np
from tests.utils.double_benchmark import DoubleBenchmarkStandard, DoubleBenchmarkComparator
from mindformers.tools.logger import logger

from .data_gen_utils import GOLDEN_DATA

DEFAULT_SEQ_LENGTH = 8

SINGLE_CARD_TEST_PARAM = "model_args, data_keys, expect_error"
SINGLE_CARD_TEST_CASES = [
    # Test case 1: tokens branch
    (
        {
            "is_dynamic": False,
            "use_attn_mask_compression": False,
            "use_tokens": True,
        },
        {"output": "output_tokens"},
        False
    ),
    # Test case 2: tokens branch with dynamic sequence length
    (
        {
            "is_dynamic": True,
            "use_attn_mask_compression": False,
            "use_tokens": True,
        },
        {"output": "output_dynamic"},
        False
    ),
    # Test case 3: masks branch (tokens is None, masks is not None)
    (
        {
            "is_dynamic": False,
            "use_attn_mask_compression": False,
            "use_tokens": False,
        },
        {"output": "output_with_masks"},
        False
    ),
    # Test case 4: use_attn_mask_compression=True (returns early, ignores tokens/masks)
    # seq_length must be >= 2048 when use_attn_mask_compression is True
    (
        {
            "seq_length": 2048,
            "is_dynamic": False,
            "use_attn_mask_compression": True,
            "use_tokens": True,
        },
        {"output": "output_mask_compression"},
        False
    ),
]

def build_msrun_command_list(
        worker_num, local_worker_num, log_dir, run_script_path,
        seq_length, is_dynamic, use_attn_mask_compression,
        use_tokens,
        output_path_param, tensor_parallel, port_id
    ):
    """ Build the msrun command with the specified parameters. """
    if worker_num == 1 and local_worker_num == 1:
        cmd_list = ["python"]
    else:
        cmd_list = [
            "msrun",
            f"--worker_num={worker_num}",
            f"--local_worker_num={local_worker_num}",
            f"--master_port={port_id}",
            f"--log_dir={log_dir}",
            "--join=True",
        ]
    cmd_list += [
        str(run_script_path),
        f"--seq_length={seq_length}",
        f"--is_dynamic={str(is_dynamic).lower()}",
        f"--use_attn_mask_compression={str(use_attn_mask_compression).lower()}",
        f"--use_tokens={str(use_tokens).lower()}",
        f"--output_path={output_path_param}",
        f"--tensor_parallel={tensor_parallel}"
    ]
    logger.info(f"Equivalent shell command for debugging (approximate): {' '.join(cmd_list)}")
    return cmd_list


class TestCausalMaskGenerate:
    """Test class for CausalMaskGenerate with different configurations"""
    OUTPUT_MS_FILENAME = "output_ms.npz"
    LOG_DIR_NAME = "msrun_log"
    def setup_method(self):
        """Setup method to prepare test environment"""
        self.sh_path = Path(__file__).parent.resolve()
        self.run_script_path = self.sh_path / "run_causal_mask_generate.py"

    def check_acc(self, output_ms_dict, data_keys):
        """
        Compare output_ms with GOLDEN_DATA using DoubleBenchmarkComparator.
        """
        standard = DoubleBenchmarkStandard(dtype="float16")

        for key, data_key in data_keys.items():
            npu_data = output_ms_dict.get(key)
            golden_data = GOLDEN_DATA.get(data_key)

            DoubleBenchmarkComparator.check_pass_or_not(
                npu_data=npu_data,
                gpu_data=golden_data,
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
            port_id=8118
        ):
        """Helper function to run test and check results"""
        output_file_path = tmp_path / self.OUTPUT_MS_FILENAME
        log_dir_path = tmp_path / self.LOG_DIR_NAME
        log_dir_path.mkdir(parents=True, exist_ok=True)

        seq_length = model_args.get("seq_length", DEFAULT_SEQ_LENGTH)
        cmd_list = build_msrun_command_list(
            worker_num=worker_num,
            local_worker_num=local_worker_num,
            log_dir=log_dir_path,
            run_script_path=self.run_script_path,
            seq_length=seq_length,
            is_dynamic=model_args["is_dynamic"],
            use_attn_mask_compression=model_args["use_attn_mask_compression"],
            use_tokens=model_args["use_tokens"],
            output_path_param=output_file_path,
            tensor_parallel=tensor_parallel,
            port_id=port_id
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

class TestCausalMaskGenerateSingleCard(TestCausalMaskGenerate):
    """Test class for CausalMaskGenerate with single card configurations"""
    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.parametrize(
        SINGLE_CARD_TEST_PARAM,
        SINGLE_CARD_TEST_CASES
    )
    def test_single_card_configurations(self, model_args, expect_error, data_keys, tmp_path):
        """Test single card with various configurations."""
        self.run_test(
            worker_num=1, local_worker_num=1,
            model_args=model_args, expect_error=expect_error,
            data_keys=data_keys,
            tmp_path=tmp_path
        )
