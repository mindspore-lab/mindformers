#  Copyright 2025 Huawei Technologies Co., Ltd
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ============================================================================
"""Test Multi-head Latent Attention (MLA) with various configurations."""
import os
import random
import pytest
from tests.st.test_multi_cards_cases.utils import TaskType
from tests.st.test_ut.test_parallel_core.test_training_graph.test_transformer.test_multi_latent_attention.test_multi_latent_attention import TestMultiLatentAttention


_LEVEL_0_TASK_TIME = 80
_LEVEL_1_TASK_TIME = 0
_TASK_TYPE = TaskType.TWO_CARDS_TASK

MULTI_CARD_TEST_PARAM = 'model_args, golden_data_key, tensor_parallel'
MULTI_CARD_TEST_CASES = [
    (
        # case 1
        # Input:
        #   - model_args: {
        #       'q_lora_rank': 8,        # Enable q_lora_rank with rank 8 for query projection
        #       'use_flash_attn': True,  # Use flash attention optimization
        #       'q_layernorm': 'RMSNorm', # Use RMSNorm for query layer normalization
        #       'k_layernorm': 'RMSNorm'  # Use RMSNorm for key layer normalization
        #       'mscale': 1.0  # Setting mscale as 1.0
        #     }
        #   - golden_data_key: 'multi_q8_flash_ql_kl' (reference output key for multi-card scenario)
        #   - tensor_parallel: 2 (use 2 GPUs with tensor parallelism)
        # Expected Output: Model output from two structures should match with each other
        {
            'q_lora_rank': 8,
            'use_flash_attn': True,
            'q_layernorm': 'RMSNorm',
            'k_layernorm': 'RMSNorm',
            'mscale': 1.0
        },
        'multi_q8_flash_ql_kl',
        2
    ),
]

class TestMultiLatentAttentionFourCards(TestMultiLatentAttention):
    """Test class for Multi-head Latent Attention with four cards configurations."""
    @pytest.mark.level0
    @pytest.mark.parametrize('struct', ['megatron', 'a2'])
    @pytest.mark.parametrize(
        MULTI_CARD_TEST_PARAM,
        MULTI_CARD_TEST_CASES
    )
    def test_multi_card_mla_cases(
            self,
            struct,
            model_args,
            golden_data_key,
            tensor_parallel,
            tmp_path
    ):
        """Test Multi-head Latent Attention on multiple cards with various configurations."""
        self.run_mla_test(
            worker_num=2,
            local_worker_num=2,
            struct=struct,
            model_args=model_args,
            golden_data_key=golden_data_key,
            tensor_parallel=tensor_parallel,
            tmp_path=tmp_path,
            port=int(os.environ.get("ASCEND_PORT_ID", random.randint(50000, 65535)))
        )
