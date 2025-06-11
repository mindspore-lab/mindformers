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
"""Test RowParallelLinear with various configurations"""
import os
import random
import pytest
from tests.st.test_multi_cards_cases.utils import TaskType
from tests.st.test_ut.test_parallel_core.test_training_graph.test_tensor_parallel.test_row_parallel_linear.test_row_parallel_linear import TestRowParallelLinear


_LEVEL_0_TASK_TIME = 89
_LEVEL_1_TASK_TIME = 0
_TASK_TYPE = TaskType.TWO_CARDS_TASK

TWO_CARD_TEST_PARAM = "model_args, data_keys, tensor_parallel"
TWO_CARD_TEST_CASES = [
    (
        {"bias": True, "skip_bias_add": True},
        {"output": "output_only", "bias": "output_bias"},
        2
    ),
]

class TestRowParallelLinearTwoCards(TestRowParallelLinear):
    """Test class for RowParallelLinear with two cards configurations"""
    @pytest.mark.level0
    @pytest.mark.parametrize(
        TWO_CARD_TEST_PARAM,
        TWO_CARD_TEST_CASES
    )
    def test_row_tp_2_case(self, model_args, data_keys, tensor_parallel, tmp_path):
        """Test two cards with various configurations."""
        self.run_test(
            worker_num=2, local_worker_num=2,
            model_args=model_args,
            data_keys=data_keys,
            tmp_path=tmp_path,
            tensor_parallel=tensor_parallel,
            port=int(os.environ.get("ASCEND_PORT_ID", random.randint(50000, 65535)))
        )
