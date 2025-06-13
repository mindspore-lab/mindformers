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
"""Test TransformerLayer with various configurations"""
import os
import random
import pytest
from tests.st.test_multi_cards_cases.utils import TaskType
from tests.st.test_ut.test_parallel_core.test_training_graph.test_transformer.test_transformer_layer.test_transformer_layer import TestTransformerLayer


_LEVEL_0_TASK_TIME = 46
_LEVEL_1_TASK_TIME = 0
_TASK_TYPE = TaskType.TWO_CARDS_TASK

# Test parameters for four cards (Distributed)
# Format: (model_args_dict, data_keys_dict, expect_error_bool, tensor_parallel_int)
TWO_CARD_TEST_PARAM = "model_args, data_keys, expect_error, tensor_parallel"
TWO_CARD_TEST_CASES = [
    # Case 1: DP=1, TP=2 for Norm, SelfAttention, Norm, MLP
    (
        {
            "input_layernorm": "Norm", "self_attention": "SelfAttention",
            "pre_cross_attn_layernorm": "IdentityOp", "cross_attention": "IdentityOp",
            "pre_mlp_layernorm": "Norm", "mlp": "MLP"
        },
        {"output": "output_default", "extra_loss": "extra_loss_default"},
        False,
        2
    ),
]

class TestTransformerLayerTwoCards(TestTransformerLayer):
    """Test class for TransformerLayer with four cards configurations"""
    @pytest.mark.level0
    @pytest.mark.parametrize(TWO_CARD_TEST_PARAM, TWO_CARD_TEST_CASES)
    def test_multi_card_configurations(self, model_args, data_keys, expect_error, tensor_parallel, tmp_path):
        """Test four cards with various configurations for TransformerLayer."""
        num_devices = 2
        self.run_test(
            worker_num=num_devices, local_worker_num=num_devices,
            model_args=model_args,
            data_keys=data_keys,
            expect_error=expect_error,
            tmp_path=tmp_path,
            tensor_parallel=tensor_parallel,
            port=int(os.environ.get("ASCEND_PORT_ID", random.randint(50000, 65535)))
        )
