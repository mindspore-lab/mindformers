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
"""Test Multi-Token Prediction (MTP) with various configurations."""
import os
import random
import pytest

from tests.st.test_multi_cards_cases.utils import TaskType
from tests.st.test_ut.test_parallel_core.test_training_graph.test_transformer.test_multi_token_prediction.test_multi_token_prediction import TestMTP


_LEVEL_0_TASK_TIME = 141
_LEVEL_1_TASK_TIME = 0
_TASK_TYPE = TaskType.TWO_CARDS_TASK

BATCH_SIZE = 2
SEQ_LENGTH = 4
HIDDEN_SIZE = 16

MULTI_CARD_TEST_PARAM = 'model_args, golden_data_key, tensor_parallel'
MULTI_CARD_TEST_CASES = [
    (
        # case 1
        # The multi-card mtp baseline (dp=2, tp=2).
        # expected result: The results of multi-card test should comply with the results of single-card test.
        {'position_embedding_type': 'learned_absolute'},
        'single_card_baseline',
        2
    ),
]

class TestMultiCardsMTP(TestMTP):
    """Test class for Multi-Token Prediction (MTP)"""
    @pytest.mark.level0
    @pytest.mark.parametrize(
        MULTI_CARD_TEST_PARAM,
        MULTI_CARD_TEST_CASES
    )
    def test_multi_card_mtp_cases(
            self,
            model_args,
            golden_data_key,
            tensor_parallel,
            tmp_path
    ):
        """Test Multi-Token Prediction (MTP) on multiple cards with various configurations."""
        self.run_mtp_test(
            worker_num=2,
            local_worker_num=2,
            model_args=model_args,
            golden_data_key=golden_data_key,
            tensor_parallel=tensor_parallel,
            tmp_path=tmp_path,
            port=int(os.environ.get("ASCEND_PORT_ID", random.randint(50000, 65535)))
        )
