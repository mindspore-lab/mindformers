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
"""Test module for testing FlashAttention used for mindformers."""
import pytest
from mindformers.parallel_core.training_graph.transformer.flash_attention import FlashAttention
from mindformers.parallel_core.transformer_config import MLATransformerConfig
from tests.utils.double_benchmark import DoubleBenchmarkComparator
from .data_gen_utils import get_init_params, GOLDEN_DATA, GPU_DATA


class TestFlashAttention:
    """A test class for testing FlashAttention"""

    def setup_method(self):
        """Setup method to prepare test environment"""
        self.config = MLATransformerConfig(multi_latent_attention=False,
                                           hidden_size=4,
                                           num_attention_heads=2,
                                           num_layers=1
                                           )
        self.inputs = get_init_params(self.config)

    def run_test(self, attention_dropout=0.0, soft_max_scale=None, accuracy=True):
        """Helper function to run test and check results"""
        self.flash_attention = FlashAttention(config=self.config, layer_number=1, attention_dropout=attention_dropout,
                                              softmax_scale=soft_max_scale)
        output = self.flash_attention(**self.inputs)
        npu_output = output.asnumpy()
        if accuracy:
            gpu_output = GPU_DATA[str(soft_max_scale)]
            golden_output = GOLDEN_DATA[str(soft_max_scale)]
            assert DoubleBenchmarkComparator.check_pass_or_not(npu_output, gpu_output, golden_output), (
                f"FlashAttention attention_dropout={attention_dropout}, soft_max_scale={soft_max_scale} test failed.\n"
                f"NPU output:\n{npu_output}\n\n"
                f"GPU output:\n{gpu_output}\n\n"
                f"Golden output:\n{golden_output}"
            )

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training # testset
    @pytest.mark.env_onecard
    def test_dropout_0_case(self):
        """
        Feature: FlashAttention
        Description: Test Case: attention_dropout=0.0, softmax_scale=None
        Exception: AssertionError
        """
        self.run_test()

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_softmax_scale_100_case(self):
        """
        Feature: FlashAttention
        Description: Test Case: attention_dropout=0.0, softmax_scale=0.5
        Exception: AssertionError
        """
        self.run_test(soft_max_scale=100.0)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_dropout_0_5_case(self):
        """
        Feature: FlashAttention
        Description: Test Case: attention_dropout=0.5, softmax_scale=None
        Exception: AssertionError
        """
        self.run_test(attention_dropout=0.5, accuracy=False)
