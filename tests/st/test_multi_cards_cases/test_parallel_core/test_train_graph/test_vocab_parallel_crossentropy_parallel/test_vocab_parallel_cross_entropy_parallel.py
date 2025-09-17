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
"""Run VocabParallelCrossEntropy accuracy test with configurable parameters via args"""

import os
import random
import pytest
from tests.st.test_ut.test_parallel_core.test_training_graph.test_vocab_parallel_crossentropy.test_vocab_parallel_cross_entropy import TestVocabParallelCrossEntropy
from tests.st.test_multi_cards_cases.utils import TaskType


_LEVEL_0_TASK_TIME = 43
_LEVEL_1_TASK_TIME = 0
_TASK_TYPE = TaskType.TWO_CARDS_TASK

TWO_CARD_TEST_PARAM = "model_args, data_keys, expect_error, tensor_parallel"
TWO_CARD_TEST_CASES = [
    # Test Case 3: Four Cards (DP=1, TP=2), check_for_nan=False, calculate_per_token=True
    (
        {"check_for_nan_in_loss_and_grad": False, "calculate_per_token_loss": True},
        {"numerator": "numerator", "denominator": "denominator"},
        False,
        2,
    ),
]

class TestVocabParallelCrossEntropyTwoCards(TestVocabParallelCrossEntropy):
    """Test VocabParallelCrossEntropy with two cards and various configurations."""
    @pytest.mark.level0
    @pytest.mark.parametrize(
        TWO_CARD_TEST_PARAM,
        TWO_CARD_TEST_CASES
    )
    def test_two_cards_case(self, model_args, data_keys, expect_error, tensor_parallel, tmp_path):
        """Test two cards with various configurations."""
        self.run_test(
            worker_num=2,
            local_worker_num=2,
            model_args=model_args,
            data_keys=data_keys,
            expect_error=expect_error,
            tmp_path=tmp_path,
            tensor_parallel=tensor_parallel,
            port=int(os.environ.get("ASCEND_PORT_ID", random.randint(50000, 65535)))
        )
