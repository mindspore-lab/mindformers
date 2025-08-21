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
"""mcore MoE tp parallel UT of inference"""
import os
import random
from pathlib import Path

import pytest

from tests.st.test_multi_cards_cases.utils import TaskType
from tests.st.test_ut.test_parallel_core.test_inference.test_transformer.test_moe.test_infer_moe import TestInferMoE
from tests.st.test_ut.test_parallel_core.test_inference.test_transformer.test_moe.data_gen_utils import GOLDEN_DATA
from tests.utils.precision_utils import PrecisionChecker


_LEVEL_0_TASK_TIME = 78
_LEVEL_1_TASK_TIME = 0
_TASK_TYPE = TaskType.FOUR_CARDS_TASK

MOE_CONFIG_WITH_SHARED_EXPERTS = {
    "batch_size": 4, "seq_len": 4, "hidden_size": 32, "moe_intermediate_size": 8, "num_experts": 8,
    "moe_shared_expert_intermediate_size": 8, "n_shared_experts": 1, "routed_scaling_factor": 2.5,
    "num_experts_per_tok": 2, "n_group": 2, "topk_group": 2
}

MOE_CONFIG_WITHOUT_SHARED_EXPERTS = {
    "batch_size": 4, "seq_len": 4, "hidden_size": 32, "moe_intermediate_size": 8, "num_experts": 8,
    "moe_shared_expert_intermediate_size": None, "n_shared_experts": 0, "routed_scaling_factor": 2.5,
    "num_experts_per_tok": 2, "n_group": 2, "topk_group": 2
}

FOUR_CARD_TEST_PARAM = "model_args, data_keys, expect_error, tensor_parallel, expert_parallel"
FOUR_CARD_DP1TP4EP1_TEST_CASES = [
    (
        # 并行策略: DP=1 TP=4 EP=1
        # eg: input (4 * 4, H) -> [moe] -> output (4 * 4,H)
        # seq_len: 4, batch_size: 4, hidden_size: 32, num_experts: 8,
        # moe_intermediate_size: 8, moe_shared_expert_intermediate_size: 8
        # expected result: 功能跑通, 精度对齐。
        MOE_CONFIG_WITH_SHARED_EXPERTS,
        {"output": "tp1"},
        False,
        4, 1),
    (
        # 并行策略: DP=1 TP=4 EP=1
        # eg: input (4 * 4, H) -> [moe] -> output (4 * 4,H)
        # seq_len: 4, batch_size: 4, hidden_size: 32, num_experts: 8,
        # moe_intermediate_size: 8, moe_shared_expert_intermediate_size: None
        # expected result: 功能跑通, 精度对齐。
        MOE_CONFIG_WITHOUT_SHARED_EXPERTS,
        {"output": "tp1_no_shared"},
        False,
        4, 1)
]


class TestInferMoELayerTP(TestInferMoE):
    """Test class for MoELayer with tp parallel"""

    def setup_method(self):
        """Setup method to prepare test environment"""
        self.sh_path = Path(__file__).parent.resolve()
        self.run_script_path = self.sh_path / "run_infer_moe_layer.py"

    @staticmethod
    def check_acc(output_ms_dict, data_keys):
        """Compare output_ms with GOLDEN_DATA and GPU_DATA."""

        checker = PrecisionChecker()
        for key, data_key in data_keys.items():
            npu_data = output_ms_dict.get(key)
            golden_data = GOLDEN_DATA.get(data_key)
            checker.check_precision(golden_data, npu_data)

    @pytest.mark.level0
    @pytest.mark.parametrize(FOUR_CARD_TEST_PARAM, FOUR_CARD_DP1TP4EP1_TEST_CASES)
    def test_four_cards_tp4ep1_cases(
            self, model_args, data_keys, expect_error,
            tensor_parallel, expert_parallel, tmp_path
    ):
        """Test four-card cases with dp1-tp4-ep1 parallel configurations."""
        self.run_test(
            worker_num=4,
            local_worker_num=4,
            model_args=model_args,
            expect_error=expect_error,
            data_keys=data_keys,
            tensor_parallel=tensor_parallel,
            expert_parallel=expert_parallel,
            tmp_path=tmp_path,
            port=int(os.environ.get("ASCEND_PORT_ID", random.randint(50000, 65535)))
        )
