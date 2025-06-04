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
import pytest
from .test_row_parallel_linear import TestRowParallelLinear

FOUR_CARD_TEST_PARAM = "model_args, data_keys, tensor_parallel"
FOUR_CARD_TEST_CASES = [
    (
        {"bias": True, "skip_bias_add": True},
        {"output": "output_only", "bias": "output_bias"},
        2
    ),
]

class TestRowParallelLinearFourCards(TestRowParallelLinear):
    """Test class for RowParallelLinear with four cards configurations"""
    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.parametrize(
        FOUR_CARD_TEST_PARAM,
        FOUR_CARD_TEST_CASES
    )
    def test_row_tp_4_case(self, model_args, data_keys, tensor_parallel, tmp_path):
        """Test four cards with various configurations."""
        self.run_test(
            worker_num=4, local_worker_num=4,
            model_args=model_args,
            data_keys=data_keys,
            tmp_path=tmp_path,
            tensor_parallel=tensor_parallel
        )
