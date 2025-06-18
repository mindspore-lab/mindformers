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
"""mcore RoPE UT of inference"""
from pathlib import Path
import subprocess
import random
import pytest
import numpy as np

from mindformers.tools.logger import logger

from tests.utils.double_benchmark import DoubleBenchmarkStandard, DoubleBenchmarkComparator
from tests.st.test_ut.test_parallel_core.test_inference.test_common.test_embeddings.test_rotary_pos_embedding.data_gen_utils import (
    get_init_params,
    KV_CHANNELS,
    ROTARY_PERCENT,
    MAX_POSITION_EMBEDDING,
    GPU_DATA,
    GOLDEN_DATA,
)

ROPE_SINGLE_CARD_TEST_PARAM = "model_args, data_keys, expect_error"
ROPE_SINGLE_CARD_TEST_CASES = [
    (
        # 并行策略: 单卡, batch_size: 1, seq_length: 2, kv_channels: 32, rotary_percent: 1.0, rotary_interleaved: FALSE,
        # seq_len_interpolation_factor: NONE, rotary_base: 10000, rotary_cos_format:2, max_position_embedding: 1024, is_prefill: TRUE
        # expected result: 功能跑通。
        {"batch_size": 1, "seq_length": 2, "kv_channels": KV_CHANNELS, "rotary_percent": ROTARY_PERCENT,
         "rotary_interleaved": False, "seq_len_interpolation_factor": None, "rotary_base": 10000,
         "rotary_cos_format": 2, "max_position_embedding": MAX_POSITION_EMBEDDING, "is_prefill": True},
        {"query": "rope_q_emb1_for_prefill", "key": "rope_k_emb1_for_prefill"},
        False
    ),
    (
        # 并行策略: 单卡, batch_size: 1, seq_length: 1, kv_channels: 32, rotary_percent: 1.0, rotary_interleaved: FALSE,
        # seq_len_interpolation_factor: NONE, rotary_base: 10000, rotary_cos_format:2, max_position_embedding: 1024, is_prefill: FALSE
        # expected result: 功能跑通。
        {"batch_size": 1, "seq_length": 1, "kv_channels": KV_CHANNELS, "rotary_percent": ROTARY_PERCENT,
         "rotary_interleaved": False, "seq_len_interpolation_factor": None, "rotary_base": 10000,
         "rotary_cos_format": 2, "max_position_embedding": MAX_POSITION_EMBEDDING, "is_prefill": False},
        {"query": "rope_q_emb1_for_decode", "key": "rope_k_emb1_for_decode"},
        False,
    ),
    (
        # 并行策略: 单卡, batch_size: 2, seq_length: 2, kv_channels: 32, rotary_percent: 1.0, rotary_interleaved: FALSE,
        # seq_len_interpolation_factor: NONE, rotary_base: 10000, rotary_cos_format:2, max_position_embedding: 1024, is_prefill: TRUE
        # expected result: 功能跑通。
        {"batch_size": 2, "seq_length": 2, "kv_channels": KV_CHANNELS, "rotary_percent": ROTARY_PERCENT,
         "rotary_interleaved": False, "seq_len_interpolation_factor": None, "rotary_base": 10000,
         "rotary_cos_format": 2, "max_position_embedding": MAX_POSITION_EMBEDDING, "is_prefill": True},
        {"query": "rope_q_emb2_for_prefill", "key": "rope_k_emb2_for_prefill"},
        False,
    ),
    (
        # 并行策略: 单卡, batch_size: 2, seq_length: 1, kv_channels: 32, rotary_percent: 1.0, rotary_interleaved: FALSE,
        # seq_len_interpolation_factor: NONE, rotary_base: 10000, rotary_cos_format:2, max_position_embedding: 1024, is_prefill: FALSE
        # expected result: 功能跑通。
        {"batch_size": 2, "seq_length": 1, "kv_channels": KV_CHANNELS, "rotary_percent": ROTARY_PERCENT,
         "rotary_interleaved": False, "seq_len_interpolation_factor": None, "rotary_base": 10000,
         "rotary_cos_format": 2, "max_position_embedding": MAX_POSITION_EMBEDDING, "is_prefill": False},
        {"query": "rope_q_emb2_for_decode", "key": "rope_k_emb2_for_decode"},
        False,
    ),
    (
        # 并行策略: 单卡, batch_size: 1, seq_length: 2, kv_channels: 32, rotary_percent: 1.0, rotary_interleaved: TRUE,
        # seq_len_interpolation_factor: NONE, rotary_base: 10000, rotary_cos_format:2, max_position_embedding: 1024, is_prefill: TRUE
        # expected result: 抛异常报错。
        {"batch_size": 1, "seq_length": 2, "kv_channels": KV_CHANNELS, "rotary_percent": ROTARY_PERCENT,
         "rotary_interleaved": True, "seq_len_interpolation_factor": None, "rotary_base": 10000,
         "rotary_cos_format": 2, "max_position_embedding": MAX_POSITION_EMBEDDING, "is_prefill": True},
        {},
        True,
    ),
    (
        # 并行策略: 单卡, batch_size: 1, seq_length: 2, kv_channels: 32, rotary_percent: 1.0, rotary_interleaved: FALSE,
        # seq_len_interpolation_factor: 1.1, rotary_base: 10000, rotary_cos_format:2, max_position_embedding: 1024, is_prefill: TRUE
        # expected result: 抛异常报错。
        {"batch_size": 1, "seq_length": 2, "kv_channels": KV_CHANNELS, "rotary_percent": ROTARY_PERCENT,
         "rotary_interleaved": False, "seq_len_interpolation_factor": 1.1, "rotary_base": 10000,
         "rotary_cos_format": 2, "max_position_embedding": MAX_POSITION_EMBEDDING, "is_prefill": True},
        {},
        True,
    ),
]

LLAMA3_ROPE_SINGLE_CARD_TEST_PARAM = "model_args, data_keys, expect_error"
LLAMA3_ROPE_SINGLE_CARD_TEST_CASES = [
    (
        # 并行策略: 单卡, batch_size: 2, seq_length: 2, kv_channels: 32, rotary_percent: 1.0, rotary_interleaved: FALSE,
        # seq_len_interpolation_factor: NONE, rotary_base: 10000, rotary_cos_format:2, max_position_embedding: 1024, is_prefill: TRUE
        # scaling_factor: 8.0, low_freq_factor: 1.0, high_freq_factor: 4.0, orig_max_position: 512
        # expected result: 功能跑通。
        {"batch_size": 2, "seq_length": 2, "kv_channels": KV_CHANNELS, "rotary_percent": ROTARY_PERCENT,
         "rotary_interleaved": False, "seq_len_interpolation_factor": None, "rotary_base": 10000,
         "rotary_cos_format": 2, "max_position_embedding": MAX_POSITION_EMBEDDING, "is_prefill": True,
         "scaling_factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "orig_max_position": 512},
        {"query": "llama3_rope_q_emb1_for_prefill", "key": "llama3_rope_k_emb1_for_prefill"},
        False
    ),
    (
        # 并行策略: 单卡, batch_size: 2, seq_length: 1, kv_channels: 32, rotary_percent: 1.0, rotary_interleaved: FALSE,
        # seq_len_interpolation_factor: NONE, rotary_base: 10000, rotary_cos_format:2, max_position_embedding: 1024, is_prefill: FALSE
        # scaling_factor: 8.0, low_freq_factor: 1.0, high_freq_factor: 4.0, orig_max_position: 512
        # expected result: 功能跑通。
        {"batch_size": 2, "seq_length": 1, "kv_channels": KV_CHANNELS, "rotary_percent": ROTARY_PERCENT,
         "rotary_interleaved": False, "seq_len_interpolation_factor": None, "rotary_base": 10000,
         "rotary_cos_format": 2, "max_position_embedding": MAX_POSITION_EMBEDDING, "is_prefill": False,
         "scaling_factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "orig_max_position": 512},
        {"query": "llama3_rope_q_emb1_for_decode", "key": "llama3_rope_k_emb1_for_decode"},
        False,
    ),
    (
        # 并行策略: 单卡, batch_size: 2, seq_length: 2, kv_channels: 32, rotary_percent: 1.0, rotary_interleaved: FALSE,
        # seq_len_interpolation_factor: NONE, rotary_base: 10000, rotary_cos_format:2, max_position_embedding: 1024, is_prefill: TRUE
        # scaling_factor: 40.0, low_freq_factor: 1.0, high_freq_factor: 4.0, orig_max_position: 512
        # expected result: 功能跑通。
        {"batch_size": 2, "seq_length": 2, "kv_channels": KV_CHANNELS, "rotary_percent": ROTARY_PERCENT,
         "rotary_interleaved": False, "seq_len_interpolation_factor": None, "rotary_base": 10000,
         "rotary_cos_format": 2, "max_position_embedding": MAX_POSITION_EMBEDDING, "is_prefill": True,
         "scaling_factor": 40.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "orig_max_position": 512},
        {"query": "llama3_rope_q_emb2_for_prefill", "key": "llama3_rope_k_emb2_for_prefill"},
        False
    ),
    (
        # 并行策略: 单卡, batch_size: 2, seq_length: 2, kv_channels: 32, rotary_percent: 1.0, rotary_interleaved: FALSE,
        # seq_len_interpolation_factor: NONE, rotary_base: 10000, rotary_cos_format:2, max_position_embedding: 1024, is_prefill: TRUE
        # scaling_factor: 8.0, low_freq_factor: 2.0, high_freq_factor: 80.0, orig_max_position: 512
        # expected result: 功能跑通。
        {"batch_size": 2, "seq_length": 2, "kv_channels": KV_CHANNELS, "rotary_percent": ROTARY_PERCENT,
         "rotary_interleaved": False, "seq_len_interpolation_factor": None, "rotary_base": 10000,
         "rotary_cos_format": 2, "max_position_embedding": MAX_POSITION_EMBEDDING, "is_prefill": True,
         "scaling_factor": 8.0, "low_freq_factor": 2.0, "high_freq_factor": 80.0, "orig_max_position": 512},
        {"query": "llama3_rope_q_emb3_for_prefill", "key": "llama3_rope_k_emb3_for_prefill"},
        False
    ),
]

YARN_ROPE_SINGLE_CARD_TEST_PARAM = "model_args, data_keys, expect_error"
YARN_ROPE_SINGLE_CARD_TEST_CASES = [
    (
        # 并行策略: 单卡, batch_size: 2, seq_length: 2, kv_channels: 32, rotary_percent: 1.0, rotary_interleaved: FALSE,
        # seq_len_interpolation_factor: NONE, rotary_base: 10000, rotary_cos_format:2, max_position_embedding: 1024, is_prefill: TRUE
        # scaling_factor: 8.0, orig_max_position: 512, beta_fast: 32, beta_slow: 1, mscale: 1.0, mscale_all_dim: 0.0
        # expected result: 功能跑通。
        {"batch_size": 2, "seq_length": 2, "kv_channels": KV_CHANNELS, "rotary_percent": ROTARY_PERCENT,
         "rotary_interleaved": False, "seq_len_interpolation_factor": None, "rotary_base": 10000,
         "rotary_cos_format": 2, "max_position_embedding": MAX_POSITION_EMBEDDING, "is_prefill": True,
         "scaling_factor": 8.0, "orig_max_position": 512, "beta_fast": 32, "beta_slow": 1,
         "mscale": 1.0, "mscale_all_dim": 0.0},
        {"query": "yarn_rope_q_emb1_for_prefill", "key": "yarn_rope_k_emb1_for_prefill"},
        False
    ),
    (
        # 并行策略: 单卡, batch_size: 2, seq_length: 1, kv_channels: 32, rotary_percent: 1.0, rotary_interleaved: FALSE,
        # seq_len_interpolation_factor: NONE, rotary_base: 10000, rotary_cos_format:2, max_position_embedding: 1024, is_prefill: FALSE
        # scaling_factor: 8.0, orig_max_position: 512, beta_fast: 32, beta_slow: 1, mscale: 1.0, mscale_all_dim: 0.0
        # expected result: 功能跑通。
        {"batch_size": 2, "seq_length": 1, "kv_channels": KV_CHANNELS, "rotary_percent": ROTARY_PERCENT,
         "rotary_interleaved": False, "seq_len_interpolation_factor": None, "rotary_base": 10000,
         "rotary_cos_format": 2, "max_position_embedding": MAX_POSITION_EMBEDDING, "is_prefill": False,
         "scaling_factor": 8.0, "orig_max_position": 512, "beta_fast": 32, "beta_slow": 1,
         "mscale": 1.0, "mscale_all_dim": 0.0},
        {"query": "yarn_rope_q_emb1_for_decode", "key": "yarn_rope_k_emb1_for_decode"},
        False,
    ),
    (
        # 并行策略: 单卡, batch_size: 2, seq_length: 2, kv_channels: 32, rotary_percent: 1.0, rotary_interleaved: FALSE,
        # seq_len_interpolation_factor: NONE, rotary_base: 10000, rotary_cos_format:2, max_position_embedding: 1024, is_prefill: TRUE
        # scaling_factor: 40.0, orig_max_position: 512, beta_fast: 32, beta_slow: 1, mscale: 1.0, mscale_all_dim: 0.0
        # expected result: 功能跑通。
        {"batch_size": 2, "seq_length": 2, "kv_channels": KV_CHANNELS, "rotary_percent": ROTARY_PERCENT,
         "rotary_interleaved": False, "seq_len_interpolation_factor": None, "rotary_base": 10000,
         "rotary_cos_format": 2, "max_position_embedding": MAX_POSITION_EMBEDDING, "is_prefill": True,
         "scaling_factor": 40.0, "orig_max_position": 512, "beta_fast": 32, "beta_slow": 1,
         "mscale": 1.0, "mscale_all_dim": 0.0},
        {"query": "yarn_rope_q_emb2_for_prefill", "key": "yarn_rope_k_emb2_for_prefill"},
        False
    ),
    (
        # 并行策略: 单卡, batch_size: 2, seq_length: 2, kv_channels: 32, rotary_percent: 1.0, rotary_interleaved: FALSE,
        # seq_len_interpolation_factor: NONE, rotary_base: 10000, rotary_cos_format:2, max_position_embedding: 1024, is_prefill: TRUE
        # scaling_factor: 8.0, orig_max_position: 512, beta_fast: 60, beta_slow: 10, mscale: 1.0, mscale_all_dim: 0.0
        # expected result: 功能跑通。
        {"batch_size": 2, "seq_length": 2, "kv_channels": KV_CHANNELS, "rotary_percent": ROTARY_PERCENT,
         "rotary_interleaved": False, "seq_len_interpolation_factor": None, "rotary_base": 10000,
         "rotary_cos_format": 2, "max_position_embedding": MAX_POSITION_EMBEDDING, "is_prefill": True,
         "scaling_factor": 8.0, "orig_max_position": 512, "beta_fast": 60, "beta_slow": 10,
         "mscale": 1.0, "mscale_all_dim": 0.0},
        {"query": "yarn_rope_q_emb3_for_prefill", "key": "yarn_rope_k_emb3_for_prefill"},
        False
    ),
]


def generate_random_port(start, end):
    """ Get random port."""
    port = random.randint(start, end)
    return port


def build_msrun_command_list(
        worker_num, local_worker_num, log_dir, run_script_path, module,
        batch_size, seq_length, rotary_percent, rotary_interleaved, seq_len_interpolation_factor,
        rotary_base, rotary_cos_format, max_position_embedding, is_prefill,
        scaling_factor: float = None, low_freq_factor: float = None, high_freq_factor: float = None,
        orig_max_position: int = None, beta_fast: int = None, beta_slow: int = None,
        mscale: float = None, mscale_all_dim: float = None, output_path_param: str = None
):
    """ Build the msrun command with the specified parameters. """
    if worker_num == 1:
        cmd_list = ["python"]
    else:
        cmd_list = [
            "msrun",
            f"--worker_num={worker_num}",
            f"--local_worker_num={local_worker_num}",  # Should match NPU cards available
            f"--master_port={generate_random_port(10000, 10100)}", # Ensure port is unique per test run if parallelized at pytest level
            f"--log_dir={log_dir}",
            "--join=True"]
    cmd_list += [
        str(run_script_path),
        f"--module={module}",
        f"--batch_size={batch_size}",
        f"--seq_length={seq_length}",
        f"--rotary_percent={rotary_percent}",
        f"--rotary_interleaved={str(rotary_interleaved).lower()}",
        f"--rotary_base={rotary_base}",
        f"--rotary_cos_format={rotary_cos_format}",
        f"--max_position_embedding={max_position_embedding}",
        f"--is_prefill={str(is_prefill).lower()}",
        f"--output_path={output_path_param}",
    ]
    if seq_len_interpolation_factor is not None:
        cmd_list.append(f"--seq_len_interpolation_factor={seq_len_interpolation_factor}")
    if scaling_factor is not None:
        cmd_list.append(f"--scaling_factor={scaling_factor}")
    if low_freq_factor is not None:
        cmd_list.append(f"--low_freq_factor={low_freq_factor}")
    if high_freq_factor is not None:
        cmd_list.append(f"--high_freq_factor={high_freq_factor}")
    if orig_max_position is not None:
        cmd_list.append(f"--orig_max_position={orig_max_position}")
    if beta_fast is not None:
        cmd_list.append(f"--beta_fast={beta_fast}")
    if beta_slow is not None:
        cmd_list.append(f"--beta_slow={beta_slow}")
    if mscale is not None:
        cmd_list.append(f"--mscale={mscale}")
    if mscale_all_dim is not None:
        cmd_list.append(f"--mscale_all_dim={mscale_all_dim}")
    logger.info(f"Equivalent shell command for debugging (approximate): {' '.join(cmd_list)}")
    return cmd_list


class TestInferRotaryEmbedding:
    """Test class for Rotary Embedding with different configurations"""
    OUTPUT_MS_FILENAME = "output_ms_rope.npz"
    LOG_DIR_NAME = "msrun_log"
    WORKER_LOG_FILENAME = "worker_0.log"

    def setup_method(self):
        """Setup method to prepare test environment"""
        self.sh_path = Path(__file__).parent.resolve()
        self.run_script_path = self.sh_path / "run_infer_rotary_embedding.py"

    @staticmethod
    def check_function(output_ms_dict, model_args, data_keys):
        """
        Compare the shapes of output_ms and input_ms whether they are the same.
        """
        for key, _ in data_keys.items():
            output_data = output_ms_dict.get(key)
            params = get_init_params(model_args["batch_size"], model_args["seq_length"], KV_CHANNELS)
            if model_args["is_prefill"]:
                input_data = params[f"{key}_for_prefill"]
            else:
                input_data = params[f"{key}_for_decode"]

            assert output_data.shape == input_data.shape, \
                (f"The shapes of output data and input data are different, "
                 f"got output shape: {output_data.shape} and input shape: {input_data.shape}")

    @staticmethod
    def check_acc(output_ms_dict, data_keys):
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

    def check_result(
            self,
            output_file_path,
            model_args,
            data_keys,
            result,
            expect_error
    ):
        """Helper function to check results"""

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

            # check whether the function of rotary embedding module works properly.
            self.check_function(output_ms_dict, model_args, data_keys)
            # Check whether the accuracy is consistent with vLLM
            self.check_acc(output_ms_dict, data_keys)

    def run_test(
            self,
            worker_num,
            local_worker_num,
            model_args,
            data_keys,
            tmp_path,
            expect_error=False,
    ):
        """Helper function to run test"""
        output_file_path = tmp_path / self.OUTPUT_MS_FILENAME
        log_dir_path = tmp_path / self.LOG_DIR_NAME
        log_dir_path.mkdir(parents=True, exist_ok=True)
        worker_log_file = log_dir_path / self.WORKER_LOG_FILENAME

        cmd_list = build_msrun_command_list(
            worker_num=worker_num,
            local_worker_num=local_worker_num,
            log_dir=log_dir_path,
            run_script_path=self.run_script_path,
            module="default",
            batch_size=model_args["batch_size"],
            seq_length=model_args["seq_length"],
            rotary_percent=model_args["rotary_percent"],
            rotary_interleaved=model_args["rotary_interleaved"],
            seq_len_interpolation_factor=model_args["seq_len_interpolation_factor"],
            rotary_base=model_args["rotary_base"],
            rotary_cos_format=model_args["rotary_cos_format"],
            max_position_embedding=model_args["max_position_embedding"],
            is_prefill=model_args["is_prefill"],
            output_path_param=output_file_path,
        )

        cmd_result = subprocess.run(
            cmd_list, shell=False, capture_output=True, text=True, check=False)

        if worker_num > 1:
            assert worker_log_file.exists()

        self.check_result(output_file_path, model_args, data_keys, cmd_result, expect_error)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.parametrize(
        ROPE_SINGLE_CARD_TEST_PARAM,
        ROPE_SINGLE_CARD_TEST_CASES
    )
    def test_single_card_configurations(self, model_args, data_keys, expect_error, tmp_path):
        """Test single card with various configurations."""
        self.run_test(
            worker_num=1, local_worker_num=1,
            model_args=model_args, expect_error=expect_error,
            data_keys=data_keys,
            tmp_path=tmp_path
        )


class TestInferLlama3RotaryEmbedding(TestInferRotaryEmbedding):
    """Test class for Llama3 Rotary Embedding with different configurations"""
    OUTPUT_MS_FILENAME = "output_ms_llama3_rope.npz"

    def run_test(
            self,
            worker_num,
            local_worker_num,
            model_args,
            data_keys,
            tmp_path,
            expect_error=False,
    ):
        """Helper function to run test"""
        output_file_path = tmp_path / self.OUTPUT_MS_FILENAME
        log_dir_path = tmp_path / self.LOG_DIR_NAME
        log_dir_path.mkdir(parents=True, exist_ok=True)
        worker_log_file = log_dir_path / self.WORKER_LOG_FILENAME

        cmd_list = build_msrun_command_list(
            worker_num=worker_num,
            local_worker_num=local_worker_num,
            log_dir=log_dir_path,
            run_script_path=self.run_script_path,
            module="llama3",
            batch_size=model_args["batch_size"],
            seq_length=model_args["seq_length"],
            rotary_percent=model_args["rotary_percent"],
            rotary_interleaved=model_args["rotary_interleaved"],
            seq_len_interpolation_factor=model_args["seq_len_interpolation_factor"],
            rotary_base=model_args["rotary_base"],
            rotary_cos_format=model_args["rotary_cos_format"],
            max_position_embedding=model_args["max_position_embedding"],
            is_prefill=model_args["is_prefill"],
            scaling_factor=model_args["scaling_factor"],
            low_freq_factor=model_args["low_freq_factor"],
            high_freq_factor=model_args["high_freq_factor"],
            orig_max_position=model_args["orig_max_position"],
            output_path_param=output_file_path,
        )

        cmd_result = subprocess.run(
            cmd_list, shell=False, capture_output=True, text=True, check=False)

        if worker_num > 1:
            assert worker_log_file.exists()

        self.check_result(output_file_path, model_args, data_keys, cmd_result, expect_error)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.parametrize(
        LLAMA3_ROPE_SINGLE_CARD_TEST_PARAM,
        LLAMA3_ROPE_SINGLE_CARD_TEST_CASES
    )
    def test_single_card_configurations(self, model_args, data_keys, expect_error, tmp_path):
        """Test single card with various configurations."""
        super().test_single_card_configurations(model_args, data_keys, expect_error, tmp_path)


class TestInferYaRNScalingRotaryEmbedding(TestInferRotaryEmbedding):
    """Test class for YaRN Scaling Rotary Embedding with different configurations"""
    OUTPUT_MS_FILENAME = "output_ms_yarn_rope.npz"

    def run_test(
            self,
            worker_num,
            local_worker_num,
            model_args,
            data_keys,
            tmp_path,
            expect_error=False,
    ):
        """Helper function to run test"""
        output_file_path = tmp_path / self.OUTPUT_MS_FILENAME
        log_dir_path = tmp_path / self.LOG_DIR_NAME
        log_dir_path.mkdir(parents=True, exist_ok=True)
        worker_log_file = log_dir_path / self.WORKER_LOG_FILENAME

        cmd_list = build_msrun_command_list(
            worker_num=worker_num,
            local_worker_num=local_worker_num,
            log_dir=log_dir_path,
            run_script_path=self.run_script_path,
            module="yarn",
            batch_size=model_args["batch_size"],
            seq_length=model_args["seq_length"],
            rotary_percent=model_args["rotary_percent"],
            rotary_interleaved=model_args["rotary_interleaved"],
            seq_len_interpolation_factor=model_args["seq_len_interpolation_factor"],
            rotary_base=model_args["rotary_base"],
            rotary_cos_format=model_args["rotary_cos_format"],
            max_position_embedding=model_args["max_position_embedding"],
            is_prefill=model_args["is_prefill"],
            scaling_factor=model_args["scaling_factor"],
            orig_max_position=model_args["orig_max_position"],
            beta_fast=model_args["beta_fast"],
            beta_slow=model_args["beta_slow"],
            mscale=model_args["mscale"],
            mscale_all_dim=model_args["mscale_all_dim"],
            output_path_param=output_file_path,
        )

        cmd_result = subprocess.run(
            cmd_list, shell=False, capture_output=True, text=True, check=False)

        if worker_num > 1:
            assert worker_log_file.exists()

        self.check_result(output_file_path, model_args, data_keys, cmd_result, expect_error)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.parametrize(
        YARN_ROPE_SINGLE_CARD_TEST_PARAM,
        YARN_ROPE_SINGLE_CARD_TEST_CASES
    )
    def test_single_card_configurations(self, model_args, data_keys, expect_error, tmp_path):
        """Test single card with various configurations."""
        super().test_single_card_configurations(model_args, data_keys, expect_error, tmp_path)
