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
"""Test SelfAttention with various configurations"""
import os
import random
import pytest
from tests.st.test_multi_cards_cases.utils import TaskType
from tests.st.test_ut.test_parallel_core.test_training_graph.test_transformer.test_attention.test_self_attention.test_self_attention import TestSelfAttention


_LEVEL_0_TASK_TIME = 90
_LEVEL_1_TASK_TIME = 0
_TASK_TYPE = TaskType.TWO_CARDS_TASK

TWO_CARD_TEST_PARAM = "model_args, data_keys, expect_error, tensor_parallel"
TWO_CARD_TEST_CASES = [
    (
        {"use_flash_attn": True, "num_query_groups": 4, "q_layernorm": None, "k_layernorm": None},
        {"output": "output_query_group_4", "bias": "bias_query_group_4"},
        False,
        2
    ),
    (
        {"use_flash_attn": True, "num_query_groups": 8, "q_layernorm": None, "k_layernorm": None},
        {"output": "output_query_group_8", "bias": "bias_query_group_8"},
        False,
        2
    ),
    (
        {"use_flash_attn": True, "num_query_groups": 4, "q_layernorm": "Norm", "k_layernorm": "Norm"},
        {"output": "output_query_group_4", "bias": "bias_query_group_4"},
        False,
        2
    ),
]

class TestSelfAttentionTwoCards(TestSelfAttention):
    """Test SelfAttention with parallel configurations."""

    @pytest.mark.level0
    @pytest.mark.parametrize(
        TWO_CARD_TEST_PARAM,
        TWO_CARD_TEST_CASES
    )
    def test_two_cards_configurations(self, model_args, data_keys, expect_error, tensor_parallel, tmp_path):
        """Test two cards with various configurations."""
        self.run_test(
            worker_num=2, local_worker_num=2,
            model_args=model_args, data_keys=data_keys,
            expect_error=expect_error, tmp_path=tmp_path,
            tensor_parallel=tensor_parallel,
            port=int(os.environ.get("ASCEND_PORT_ID", random.randint(50000, 65535)))
        )
