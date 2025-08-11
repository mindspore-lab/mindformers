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
"""Test FusedNorm with various configurations"""
import pytest
import mindspore as ms
from mindformers.parallel_core.training_graph.transformer.norm import FusedNorm
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.training_graph.device_matrix import layout
from tests.utils.double_benchmark import DoubleBenchmarkComparator, DoubleBenchmarkStandard
from .data_gen_utils import get_init_params, GOLDEN_DATA, GPU_DATA


class TestFusedNorm:
    """A test class for testing Fused Norm"""

    def setup_method(self):
        """Setup method to prepare test environment"""
        init_params = get_init_params()
        self.inputs = ms.Tensor(init_params.get("inputs"), dtype=ms.float32)
        self.standard = DoubleBenchmarkStandard(dtype="float32")

    def run_test(self, normalization: str, hidden_size=32):
        """Helper function to run test and check results"""
        self.config = TransformerConfig(normalization=normalization, params_dtype='float32',
                                        layernorm_compute_dtype='float32', num_layers=1, num_attention_heads=2)
        layout.init_layout(self.config)
        self.norm = FusedNorm(config=self.config, dim=hidden_size)
        output = self.norm.construct(self.inputs)
        npu_output = output.asnumpy()
        gpu_output = GPU_DATA[normalization]
        golden_output = GOLDEN_DATA[normalization]
        assert DoubleBenchmarkComparator.check_pass_or_not(npu_output, gpu_output, golden_output, self.standard), (
            f"FusedNorm test failed.\n"
            f"NPU output:\n{npu_output}\n\n"
            f"GPU output:\n{gpu_output}\n\n"
            f"Golden output:\n{golden_output}"
        )

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_layer_norm(self):
        """
        Feature: Activation
        Description: Test Case: normalization='LayerNorm',
        Exception: AssertionError
        """
        self.run_test(normalization='LayerNorm')

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_rms_norm(self):
        """
        Feature: Activation
        Description: Test Case: normalization='LayerNorm',
        Exception: AssertionError
        """
        self.run_test(normalization='RMSNorm')
