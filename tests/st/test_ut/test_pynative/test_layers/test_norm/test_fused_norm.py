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
from mindspore import context
import numpy as np
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.pynative.layers.layer_norm import get_norm_cls as get_pynative_norm_cls
from mindformers.parallel_core.training_graph.transformer.norm import get_norm_cls as get_graph_norm_cls
from mindformers.parallel_core.training_graph.device_matrix import layout
from .data_gen_utils import get_init_params


# pylint: disable=attribute-defined-outside-init
class TestFusedNorm:
    """A test class for testing Fused Norm"""

    def setup_method(self):
        """Setup method to prepare test environment"""
        init_params = get_init_params()
        # Create only one float32 input, will be cast to appropriate dtype in run_norm_in_mode
        self.inputs = ms.Tensor(init_params.get("inputs"), dtype=ms.float32)

    def run_norm_in_mode(self, normalization: str, mode: int, dtype: str, hidden_size=32):
        """Helper function to run norm in specific mode and return output"""
        # Set execution mode
        context.set_context(mode=mode)

        # Update config with current dtype - both params_dtype and layernorm_compute_dtype are set to the same value
        config = TransformerConfig(
            normalization=normalization,
            params_dtype=dtype,
            layernorm_compute_dtype=dtype,
            num_layers=1,
            num_attention_heads=2
        )
        layout.init_layout(config)

        # Use appropriate norm implementation based on mode
        if mode == context.PYNATIVE_MODE:  # Dynamic graph mode
            # Get norm class and create instance
            norm_cls = get_pynative_norm_cls(normalization=normalization, fused_norm=True)
            norm = norm_cls(
                dim=hidden_size,
                params_dtype=config.params_dtype,
                compute_dtype=config.layernorm_compute_dtype
            )
        else:  # Static graph mode
            # Get norm factory class and create instance
            norm_factory = get_graph_norm_cls(fused_norm=True)
            norm = norm_factory(
                config=config,
                dim=hidden_size
            )

        # Use the same input, cast to appropriate dtype if needed
        # For bfloat16, cast the input to bfloat16 before passing to norm
        if dtype == 'bfloat16':
            inputs = ms.ops.cast(self.inputs, ms.bfloat16)
        else:
            inputs = self.inputs

        # Run forward pass
        if mode == context.PYNATIVE_MODE:
            output = norm.construct(inputs)
        else:
            output = norm(inputs)

        # Convert to numpy array for comparison
        return output.asnumpy()

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    def test_norm_precision_between_modes(self, normalization: str, dtype: str):
        """Test precision consistency between PYNATIVE and GRAPH modes for different dtypes"""
        # Run in PYNATIVE mode (dynamic graph)
        pynative_output = self.run_norm_in_mode(normalization, context.PYNATIVE_MODE, dtype)

        # Run in GRAPH mode (static graph)
        graph_output = self.run_norm_in_mode(normalization, context.GRAPH_MODE, dtype)

        assert (pynative_output == graph_output).all(), (
            f"{normalization} precision inconsistency between modes for {dtype}.\n"
            f"PYNATIVE mode output:\n{pynative_output}\n\n"
            f"GRAPH mode output:\n{graph_output}\n\n"
            f"Max difference: {np.max(np.abs(pynative_output - graph_output))}"
        )
