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
"""mcore Attention UT of inference"""
import os
import random
import pytest

from tests.st.test_multi_cards_cases.utils import TaskType
from tests.st.test_ut.test_parallel_core.test_inference.test_transformer.test_attention.test_infer_attention import TestSelfAttention
from tests.st.test_ut.test_parallel_core.test_inference.test_transformer.test_attention.data_gen_utils import (
    BATCH_SIZE,
    PREFILL_SEQ_LEN,
    DECODE_SEQ_LEN,
    NUM_HEADS,
    HIDDEN_SIZE
)


_LEVEL_0_TASK_TIME = 43
_LEVEL_1_TASK_TIME = 0
_TASK_TYPE = TaskType.TWO_CARDS_TASK
TWO_CARD_TEST_PARAM = "model_args, data_keys, expect_error, tensor_parallel"
TWO_CARD_TEST_CASES = [
    (
        # 并行策略: 双卡, batch_size: 2, prefill_seq_len: 2, decode_seq_len: 1,
        # num_heads: 2, num_query_groups: 2, hidden_size: 32, use_flash_attention: TRUE
        # expected_result: 功能跑通。
        {"batch_size": BATCH_SIZE, "prefill_seq_len": PREFILL_SEQ_LEN,
         "decode_seq_len": DECODE_SEQ_LEN, "num_heads": NUM_HEADS, "num_query_groups": 2,
         "hidden_size": HIDDEN_SIZE, "use_flash_attention": True},
        {"prefill_output": "prefill_output_1", "decode_output": "decode_output_1"},
        False,
        2
    )
]
class TestInferAttentionParallel(TestSelfAttention):
    """Test class for Attention with different configurations"""
    @pytest.mark.level0
    @pytest.mark.parametrize(
        TWO_CARD_TEST_PARAM,
        TWO_CARD_TEST_CASES
    )
    def test_two_cards_configurations(self, model_args, data_keys, expect_error, tensor_parallel, tmp_path):
        """Test two cards with various configurations."""
        os.environ['MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST'] = 'PagedAttention'
        self.run_test(
            worker_num=tensor_parallel, local_worker_num=tensor_parallel,
            model_args=model_args, expect_error=expect_error,
            data_keys=data_keys,
            tmp_path=tmp_path,
            tensor_parallel=tensor_parallel,
            port=int(os.environ.get("ASCEND_PORT_ID", random.randint(50000, 65535)))
        )
