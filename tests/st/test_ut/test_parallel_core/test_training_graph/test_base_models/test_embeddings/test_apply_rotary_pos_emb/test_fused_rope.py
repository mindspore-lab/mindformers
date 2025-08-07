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
"""Test module for testing fused RoPE in ApplyRotaryPosEmb used for mindformers."""
import pytest
import mindspore as ms
from mindformers.parallel_core.training_graph.base_models.common.embeddings.rope_utils import ApplyRotaryPosEmb
from mindformers.parallel_core.transformer_config import TransformerConfig
from tests.utils.double_benchmark import DoubleBenchmarkComparator, DoubleBenchmarkStandard

from .data_gen_utils import get_init_params


class TestFusedRoPE:
    """A test class for testing FusedRoPE kernel"""

    def setup_method(self):
        """Setup method to prepare test environment"""
        init_params = get_init_params()
        self.input_t = ms.Tensor(init_params.get("t"), dtype=ms.float32)
        self.input_freqs = ms.Tensor(init_params.get("freqs"), dtype=ms.float32)
        self.mscale = 1.0
        self.freqs = (self.input_freqs, self.mscale)
        self.no_fused_rope_config = TransformerConfig(num_attention_heads=1,
                                                      num_layers=1,
                                                      apply_rope_fusion=False)
        self.fused_rope_config = TransformerConfig(num_attention_heads=1,
                                                   num_layers=1,
                                                   apply_rope_fusion=True)

    def run_test(self):
        """Helper function to run test"""
        fused_rope_output = ApplyRotaryPosEmb(self.fused_rope_config)(self.input_t, self.freqs)
        no_fused_rope_output = ApplyRotaryPosEmb(self.no_fused_rope_config)(self.input_t, self.freqs)
        actual = fused_rope_output.asnumpy()
        golden = no_fused_rope_output.asnumpy()
        standard = DoubleBenchmarkStandard(dtype="float32")
        assert DoubleBenchmarkComparator.check_pass_or_not(actual, golden, golden, standard), (
            f"FusedRoPE test failed.\n"
            f"FusedRoPE output:\n{actual}\n\n"
            f"RoPE output:\n{golden}\n\n"
        )

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_fused_rope(self):
        """
        Feature: ApplyRotaryPosEmb
        Description: Test Fused RoPE,
        Exception: AssertionError
        """
        self.run_test()
