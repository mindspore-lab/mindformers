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
import os
import random
import pytest
from tests.st.test_multi_cards_cases.utils import TaskType
from tests.st.test_ut.test_parallel_core.test_training_graph.test_transformer.test_fused_scale_mask_softmax.test_fused_scale_mask_softmax import TestFusedScaleMaskSoftmax


_LEVEL_0_TASK_TIME = 45
_LEVEL_1_TASK_TIME = 0
_TASK_TYPE = TaskType.TWO_CARDS_TASK

MULTI_CARD_TEST_PARAM = "model_args, golden_data_key, expect_error, tensor_parallel"
MULTI_CARD_TEST_CASES = [
    (
        # input: 并行策略: 2卡dp1tp2并行, input_in_fp16: FALSE, input_in_bf16: TRUE, attn_mask_type: AttnMaskType.causal, mask_func: AttnMaskFill, softmax_in_fp32: TRUE, scale: None, mask: None
        # expected result: 功能跑通，精度对齐。
        {"input_in_fp16": False, "input_in_bf16": True, "attn_mask_type": "causal", "mask_func_name": "attn_mask_fill",
         "softmax_in_fp32": True, "scale": None, "use_construct_mask": False},
        "output_base",
        False,
        2
    ),
]


class TestFusedScaleMaskSoftmaxFourCards(TestFusedScaleMaskSoftmax):
    """Test class for FusedScaleMaskSoftmax with four cards configurations"""
    @pytest.mark.level0
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
            worker_num=2, local_worker_num=2,
            model_args=model_args,
            golden_data_key=golden_data_key,
            tensor_parallel=tensor_parallel,
            expect_error=expect_error,
            tmp_path=tmp_path,
            port=int(os.environ.get("ASCEND_PORT_ID", random.randint(50000, 65535)))
        )
