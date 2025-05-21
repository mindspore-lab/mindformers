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
"""Test FusedScaleMaskSoftmax with various configurations"""
from pathlib import Path
import subprocess
import os
import pytest
import numpy as np
from data_gen_utils import GOLDEN_DATA, GPU_DATA
from mindformers.tools.logger import logger
from tests.utils.double_benchmark import DoubleBenchmarkStandard, DoubleBenchmarkComparator

BATCH_SIZE = 4
NUM_HEADS = 2
SEQ_LENGTH = 4

# Test case parameters: model_args, golden_data_key, expect_error
SINGLE_CARD_TEST_PARAM = "model_args, golden_data_key, expect_error"
SINGLE_CARD_TEST_CASES = [
    (
        # input: 并行策略: 单卡, input_in_fp16: FALSE, input_in_bf16: TRUE, attn_mask_type: AttnMaskType.causal, mask_func: AttnMaskFill, softmax_in_fp32: TRUE, scale: None, mask: None
        # expected result: 功能跑通，精度对齐。
        {"input_in_fp16": False, "input_in_bf16": True, "attn_mask_type": "causal", "mask_func_name": "attn_mask_fill",
         "softmax_in_fp32": True, "scale": None, "use_construct_mask": False},
        "output_base",
        False
    ),
    (
        # input: 并行策略: 单卡, input_in_fp16: TRUE, input_in_bf16: TRUE, attn_mask_type: AttnMaskType.causal, mask_func: AttnMaskFill, softmax_in_fp32: TRUE, scale: None, mask: None
        # expected result: 报错
        {"input_in_fp16": True, "input_in_bf16": True, "attn_mask_type": "causal", "mask_func_name": "attn_mask_fill",
         "softmax_in_fp32": True, "scale": None, "use_construct_mask": False},
        None,
        True
    ),
    (
        # input: 并行策略: 单卡, input_in_fp16: TRUE, input_in_bf16: FALSE, attn_mask_type: AttnMaskType.causal, mask_func: AttnMaskFill, softmax_in_fp32: TRUE, scale: None, mask: None
        # expected result: 功能跑通，精度对齐。
        {"input_in_fp16": True, "input_in_bf16": False, "attn_mask_type": "causal", "mask_func_name": "attn_mask_fill",
         "softmax_in_fp32": True, "scale": None, "use_construct_mask": False},
        "output_fp16",
        False
    ),
    (
        # input: 并行策略: 单卡, input_in_fp16: FALSE, input_in_bf16: TRUE, attn_mask_type: others, mask_func: AttnMaskFill, softmax_in_fp32: TRUE, scale: None, mask: None
        # expected result: 功能跑通，精度对齐。
        {"input_in_fp16": False, "input_in_bf16": True, "attn_mask_type": "padding", "mask_func_name": "attn_mask_fill",
         "softmax_in_fp32": True, "scale": None, "use_construct_mask": False},
        "output_padding",
        False
    ),
    (
        # input: 并行策略: 单卡, input_in_fp16: FALSE, input_in_bf16: TRUE, attn_mask_type: AttnMaskType.causal, mask_func: AttnMaskFill, softmax_in_fp32: TRUE, scale: 0.9, mask: None
        # expected result: 功能跑通，精度对齐。
        {"input_in_fp16": False, "input_in_bf16": True, "attn_mask_type": "causal", "mask_func_name": "attn_mask_fill",
         "softmax_in_fp32": True, "scale": "0.9", "use_construct_mask": False},
        "output_scale_0_9",
        False
    ),
    (
        # input: 并行策略: 单卡, input_in_fp16: FALSE, input_in_bf16: TRUE, attn_mask_type: AttnMaskType.causal, mask_func: AttnMaskFill, softmax_in_fp32: TRUE, scale: None, mask: tensor
        # expected result: 功能跑通，精度对齐。（传入的mask优先）
        {"input_in_fp16": False, "input_in_bf16": True, "attn_mask_type": "causal", "mask_func_name": "attn_mask_fill",
         "softmax_in_fp32": True, "scale": None, "use_construct_mask": True},
        "output_use_construct_mask",
        False
    )
]

MULTI_CARD_TEST_PARAM = "model_args, golden_data_key, expect_error, tensor_parallel"
MULTI_CARD_TEST_CASES = [
    (
        # input: 并行策略: 四卡dp2tp2并行, input_in_fp16: FALSE, input_in_bf16: TRUE, attn_mask_type: AttnMaskType.causal, mask_func: AttnMaskFill, softmax_in_fp32: TRUE, scale: None, mask: None
        # expected result: 功能跑通，精度对齐。
        {"input_in_fp16": False, "input_in_bf16": True, "attn_mask_type": "causal", "mask_func_name": "attn_mask_fill",
         "softmax_in_fp32": True, "scale": None, "use_construct_mask": False},
        "output_base",
        False,
        2
    ),
]


def build_msrun_command_list(
        worker_num, local_worker_num, log_dir, run_script_path,
        batch_size, num_heads, seq_length, # Input shape args
        model_args, # Dictionary of model config args
        output_path_param, tensor_parallel
    ):
    """ Build the msrun command for FusedScaleMaskSoftmax. """
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
        f"--batch_size={batch_size}",
        f"--num_heads={num_heads}",
        f"--seq_length={seq_length}",
        f"--input_in_fp16={str(model_args['input_in_fp16']).lower()}",
        f"--input_in_bf16={str(model_args['input_in_bf16']).lower()}",
        f"--attn_mask_type={model_args['attn_mask_type']}",
        f"--mask_func_name={model_args['mask_func_name']}",
        f"--softmax_in_fp32={str(model_args['softmax_in_fp32']).lower()}",
        f"--use_construct_mask={str(model_args['use_construct_mask']).lower()}",
        f"--output_path={output_path_param}",
        f"--tensor_parallel={tensor_parallel}"
    ]
    if model_args['scale'] is not None:
        cmd_list.append(f"--scale={model_args['scale']}")
    logger.info(f"Equivalent shell command for FusedScaleMaskSoftmax (approximate): {' '.join(cmd_list)}")
    return cmd_list

class TestFusedScaleMaskSoftmax:
    """Test class for FusedScaleMaskSoftmax"""
    OUTPUT_MS_FILENAME = "output_softmax_ms.npz"
    LOG_DIR_NAME = "msrun_softmax_log"

    def setup_method(self):
        """Setup method to prepare test environment"""
        self.current_dir = Path(__file__).parent.resolve()
        self.run_script_path = self.current_dir / "run_fused_scale_mask_softmax.py"

    def check_acc(self, output_ms_dict, golden_data_key):
        """
        Compare output_ms with GOLDEN_DATA and GPU_DATA.
        """
        if not golden_data_key: # Should not happen if expect_error is False
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

    def run_softmax_test(
            self,
            worker_num,
            local_worker_num,
            model_args,
            golden_data_key,
            tmp_path,
            tensor_parallel=1,
            expect_error=False,
        ):
        """Helper function to run softmax test and check results"""
        output_file_path = tmp_path / self.OUTPUT_MS_FILENAME
        log_dir_path = tmp_path / self.LOG_DIR_NAME
        log_dir_path.mkdir(parents=True, exist_ok=True)

        cmd_list = build_msrun_command_list(
            worker_num=worker_num,
            local_worker_num=local_worker_num,
            log_dir=log_dir_path,
            run_script_path=self.run_script_path,
            batch_size=BATCH_SIZE,
            num_heads=NUM_HEADS,
            seq_length=SEQ_LENGTH,
            model_args=model_args,
            output_path_param=output_file_path,
            tensor_parallel=tensor_parallel,
        )
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = "0"

        result = subprocess.run(
            cmd_list, shell=False, capture_output=True, text=True, check=False, env=env
        )

        if expect_error:
            assert result.returncode != 0, (
                f"Expected an error but FusedScaleMaskSoftmax script passed. "
                f"Stdout:\n{result.stdout}\n"
                f"Stderr:\n{result.stderr}"
            )
            if model_args.get("input_in_fp16") and model_args.get("input_in_bf16"):
                assert "Both fp16 and bf16 flags cannot be active" in result.stderr or \
                       "Both fp16 and bf16 flags cannot be active" in result.stdout
            elif model_args.get("scale") is not None and not model_args.get("softmax_in_fp32"):
                assert "softmax should be in fp32 when scaled" in result.stderr or \
                    "softmax should be in fp32 when scaled" in result.stdout
        else:
            assert result.returncode == 0, (
                f"FusedScaleMaskSoftmax script failed with non-zero exit code: "
                f"{result.returncode}.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"
            )
            assert output_file_path.exists(), (
                f"Output file {output_file_path} was not created."
            )
            output_ms_dict = np.load(output_file_path)
            self.check_acc(output_ms_dict, golden_data_key)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.parametrize(
        SINGLE_CARD_TEST_PARAM,
        SINGLE_CARD_TEST_CASES
    )
    def test_single_card_softmax_cases(
            self,
            model_args,
            golden_data_key,
            expect_error,
            tmp_path
        ):
        """Test FusedScaleMaskSoftmax on single card with various configurations."""
        self.run_softmax_test(
            worker_num=1, local_worker_num=1,
            model_args=model_args,
            golden_data_key=golden_data_key,
            expect_error=expect_error,
            tmp_path=tmp_path
        )

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.parametrize(
        MULTI_CARD_TEST_PARAM,
        MULTI_CARD_TEST_CASES
    )
    def test_multi_card_softmax_cases(
            self,
            model_args,
            golden_data_key,
            expect_error,
            tensor_parallel,
            tmp_path
        ):
        """Test FusedScaleMaskSoftmax on multiple cards with various configurations."""
        self.run_softmax_test(
            worker_num=4, local_worker_num=4,
            model_args=model_args,
            golden_data_key=golden_data_key,
            tensor_parallel=tensor_parallel,
            expect_error=expect_error,
            tmp_path=tmp_path
        )
