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
"""Test module for testing SharedKVCrossAttention used for mindformers."""
import pytest
import mindspore as ms
from tests.st.test_ut.test_parallel_core.test_training_graph.test_transformer.test_attention.test_shared_kv_cross_attention.data_gen_utils import get_init_params, GOLDEN_DATA, GPU_DATA
from tests.utils.double_benchmark import DoubleBenchmarkComparator
from mindformers.parallel_core.training_graph.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from mindformers.parallel_core.training_graph.transformer.attention import SharedKVCrossAttention, SharedKVCrossAttentionSubmodules
from mindformers.parallel_core.training_graph.transformer.flash_attention import FlashAttention
from mindformers.parallel_core.transformer_config import TransformerConfig


class TestSharedKVCrossAttention:
    """A test class for testing SharedKVCrossAttention"""

    def setup_method(self):
        """Setup method to prepare test environment"""
        self.config = TransformerConfig(
            compute_dtype='bfloat16',
            use_flash_attention=True,
            num_query_groups=2,
            data_parallel_size=1,
            tensor_model_parallel_size=1,
            hidden_size=8,
            num_attention_heads=2,
            add_bias_linear=True,
            add_qkv_bias=True,
            num_layers=1,
            params_dtype='float32',
            attention_dropout=0.0,
            context_parallel_size=1,
            model_architecture="yoco",
            num_encoder_layers=0,
            num_decoder_layers=1
        )

        ms.set_context(mode=ms.GRAPH_MODE)

        submodules = SharedKVCrossAttentionSubmodules(
            linear_q=ColumnParallelLinear,
            core_attention=FlashAttention,
            linear_proj=RowParallelLinear,
        )

        self.net = SharedKVCrossAttention(
            config=self.config,
            submodules=submodules,
            layer_number=1
        )

        self.inputs, weight_dict = get_init_params(self.config)
        self.net.load_state_dict(weight_dict, strict=False)

    def run_test(self, accuracy=True, compare_type=None):
        """Helper function to run test and check results"""

        output, _ = self.net(**self.inputs)
        npu_output = output.asnumpy()
        if accuracy:
            gpu_output = GPU_DATA[compare_type]
            golden_output = GOLDEN_DATA[compare_type]
            assert DoubleBenchmarkComparator.check_pass_or_not(npu_output, gpu_output, golden_output), (
                f"SharedKVCrossAttention compare_type={compare_type} test failed.\n"
                f"NPU output:\n{npu_output}\n\n"
                f"GPU output:\n{gpu_output}\n\n"
                f"Golden output:\n{golden_output}"
            )


    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_shared_kv_cross_attention(self):
        """
        Feature: SharedKVCrossAttention
        Description: Test Case: SharedKVCrossAttention
        """
        self.run_test(compare_type="output_attn")
