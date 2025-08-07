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
"""Test YarnRotaryEmbedding with various configurations"""
from pathlib import Path
import subprocess
import pytest
import numpy as np
from mindformers.tools.logger import logger
from tests.utils.double_benchmark import DoubleBenchmarkStandard, DoubleBenchmarkComparator

from .data_gen_utils import GOLDEN_DATA, GPU_DATA

KV_CHANNELS = 32

SINGLE_CARD_TEST_PARAM = "model_args, data_keys"
SINGLE_CARD_TEST_CASES = [
    (
        {"rotary_percent": 1, "rotary_interleaved": False, "seq_len_interpolation_factor": None,
         "original_max_position_embeddings": 4096, "beta_fast": 32, "beta_slow": 1, "mscale": 1, "mscale_all_dim": 0},
        {"output": "output_1", "mscale": "mscale_1"},
    ),
]


def build_msrun_command_list(
        worker_num, local_worker_num, log_dir, run_script_path,
        kv_channels, rotary_percent, rotary_interleaved, seq_len_interpolation_factor, original_max_position_embeddings,
        beta_fast, beta_slow, mscale, mscale_all_dim,
        output_path_param, tensor_parallel
    ):
    """ Build the msrun command with the specified parameters. """
    if worker_num == 1 and local_worker_num == 1:
        cmd_list = ["python"]
    else:
        cmd_list = [
            "msrun",
            f"--worker_num={worker_num}",
            f"--local_worker_num={local_worker_num}",
            "--master_port=8168",
            f"--log_dir={log_dir}",
            "--join=True",
        ]
    cmd_list += [
        str(run_script_path),
        f"--kv_channels={kv_channels}",
        f"--rotary_percent={rotary_percent}",
        f"--rotary_interleaved={str(rotary_interleaved).lower()}",
        f"--original_max_position_embeddings={original_max_position_embeddings}",
        f"--beta_fast={beta_fast}",
        f"--beta_slow={beta_slow}",
        f"--mscale={mscale}",
        f"--mscale_all_dim={mscale_all_dim}",
        f"--output_path={output_path_param}",
        f"--tensor_parallel={tensor_parallel}"
    ]
    if seq_len_interpolation_factor is not None:
        cmd_list.append(f"--seq_len_interpolation_factor={seq_len_interpolation_factor}")
    logger.info(f"Equivalent shell command for YarnRotaryEmbedding (approximate): {' '.join(cmd_list)}")
    return cmd_list


class TestYarnRotaryEmbedding:
    """Test class for YarnRotaryEmbedding with different configurations"""
    OUTPUT_MS_FILENAME = "output_ms.npz"
    LOG_DIR_NAME = "msrun_log"

    def setup_method(self):
        """Setup method to prepare test environment"""
        self.sh_path = Path(__file__).parent.resolve()
        self.run_script_path = self.sh_path / "run_yarn_rotary_embedding.py"

    def check_acc(self, output_ms_dict, data_keys):
        """
        Compare output_ms with GOLDEN_DATA and GPU_DATA using DoubleBenchmarkComparator.
        """
        standard = DoubleBenchmarkStandard(dtype="float32")

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
            kv_channels=KV_CHANNELS,
            rotary_percent=model_args["rotary_percent"],
            rotary_interleaved=model_args["rotary_interleaved"],
            seq_len_interpolation_factor=model_args["seq_len_interpolation_factor"],
            original_max_position_embeddings=model_args["original_max_position_embeddings"],
            beta_fast=model_args["beta_fast"],
            beta_slow=model_args["beta_slow"],
            mscale=model_args["mscale"],
            mscale_all_dim=model_args["mscale_all_dim"],
            output_path_param=output_file_path,
            tensor_parallel=tensor_parallel,
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

        output_ms_dict = np.load(output_file_path)

        self.check_acc(output_ms_dict, data_keys)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.parametrize(
        SINGLE_CARD_TEST_PARAM,
        SINGLE_CARD_TEST_CASES
    )
    def test_yarn_no_parallel_case(self, model_args, data_keys, tmp_path):
        """Test single card with various configurations."""
        self.run_test(
            worker_num=1, local_worker_num=1,
            model_args=model_args,
            data_keys=data_keys,
            tmp_path=tmp_path
        )
