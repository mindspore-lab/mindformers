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
"""Test GPTModel with various configurations"""
import random
import pytest
from tests.st.test_multi_cards_cases.utils import TaskType
from tests.st.test_ut.test_parallel_core.test_inference.test_base_models.test_gpt_model.test_infer_gpt_model import TestInferGPTModel


_LEVEL_0_TASK_TIME = 95
_LEVEL_1_TASK_TIME = 0
_TASK_TYPE = TaskType.TWO_CARDS_TASK

# Test parameters for four cards (Distributed)
# Format: (model_args_dict, data_keys_dict, expect_error_bool, tensor_parallel_int)
TWO_CARD_TEST_PARAM = "model_args, data_keys, expect_error, tensor_parallel, pipeline_parallel"
TWO_CARD_TEST_CASES = [
    (
        # prefill, 并行策略: PP2, Transformer Layer组成模块: Standard Norm, SelfAttention, MLP, num_layers: 2
        # expected result: 功能跑通, 精度对齐。
        {"is_prefill": True, "num_experts": None, "moe_grouped_gemm": False, "qk_layernorm": False,
         "multi_latent_attention": False, "qk_l2_norm": False, "sandwich_norm": False, "num_layers": 2},
        {"output": "output_standard_layer1"},
        False,
        1,
        2,
    ),
    (
        # decode, 并行策略: PP2, Transformer Layer组成模块: Standard Norm, SelfAttention, MLP, num_layers: 2
        # expected result: 功能跑通, 精度对齐。
        {"is_prefill": False, "num_experts": None, "moe_grouped_gemm": False, "qk_layernorm": False,
         "multi_latent_attention": False, "qk_l2_norm": False, "sandwich_norm": False, "num_layers": 2},
        {"output": "output_standard_layer1"},
        False,
        1,
        2,
    ),
]

class TestGPTModelTwoCards(TestInferGPTModel):
    """Test class for GPTModel with two cards configurations"""
    @pytest.mark.level0
    @pytest.mark.parametrize(TWO_CARD_TEST_PARAM, TWO_CARD_TEST_CASES)
    def test_multi_card_configurations(self, model_args, data_keys, expect_error, tensor_parallel, pipeline_parallel,
                                       tmp_path):
        """Test two cards with various configurations for GPTModel."""
        num_devices = tensor_parallel * pipeline_parallel
        self.run_test(
            worker_num=num_devices, local_worker_num=num_devices,
            model_args=model_args,
            data_keys=data_keys,
            expect_error=expect_error,
            tmp_path=tmp_path,
            tensor_parallel=tensor_parallel,
            pipeline_parallel=pipeline_parallel,
            port=random.randint(50000, 65535)
        )
