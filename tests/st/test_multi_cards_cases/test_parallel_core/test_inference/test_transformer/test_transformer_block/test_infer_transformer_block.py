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
"""mcore transformer block UT of inference"""

import os
import random
import pytest

from mindformers.tools.logger import logger

from tests.st.test_multi_cards_cases.utils import TaskType
from tests.st.test_ut.test_parallel_core.test_inference.test_transformer.test_transformer_block.test_infer_transformer_block import TestInferTransformerLayer


_LEVEL_0_TASK_TIME = 73
_LEVEL_1_TASK_TIME = 0
_TASK_TYPE = TaskType.TWO_CARDS_TASK


TWO_CARD_TEST_PARAM = "model_args, data_keys, expect_error, tensor_parallel"
TWO_CARD_TEST_CASES = [
    (
        # 并行策略: 双卡tp=2, Transformer Layer组成模块: Standard Norm, SelfAttention, MLP, num_layers: 1
        # expected result: 功能跑通, 精度对齐。
        {"num_experts": None, "moe_grouped_gemm": False, "qk_layernorm": False, "multi_latent_attention": False,
         "qk_l2_norm": False, "sandwich_norm": False, "num_layers": 1},
        {"output": "output_standard_layer1"},
        False,
        2
    ),
    (
        # 并行策略: 双卡tp=2, Transformer Layer组成模块: Standard Norm, SelfAttention, MLP, num_layers: 2
        # expected result: 功能跑通, 精度对齐。
        {"num_experts": None, "moe_grouped_gemm": False, "qk_layernorm": False, "multi_latent_attention": False,
         "qk_l2_norm": False, "sandwich_norm": False, "num_layers": 2},
        {"output": "output_standard_layer2"},
        False,
        2
    ),
]


class TestInferTransformerLayerParallel(TestInferTransformerLayer):
    """Test class for Transformer Block with different configurations"""
    @pytest.mark.level0
    @pytest.mark.parametrize(
        TWO_CARD_TEST_PARAM,
        TWO_CARD_TEST_CASES
    )
    def test_two_cards_cases(self, model_args, data_keys, expect_error, tensor_parallel, tmp_path):
        """Test four cards cases with various configurations."""
        logger.info(
            f"--- Running Multi-Card Test: model_args={model_args}, TP={tensor_parallel} ---")
        self.run_test(
            worker_num=tensor_parallel, local_worker_num=tensor_parallel,
            model_args=model_args, expect_error=expect_error,
            data_keys=data_keys,
            tmp_path=tmp_path,
            tensor_parallel=tensor_parallel,
            port=int(os.environ.get("ASCEND_PORT_ID", random.randint(50000, 65535)))
        )
