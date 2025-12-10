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
import mindspore as ms
from tests.st.test_ut.test_parallel_core.test_training_graph.test_transformer.test_attention.test_shared_kv_cross_attention.data_gen_utils import get_init_block_params, GOLDEN_DATA, GPU_DATA
from tests.utils.double_benchmark import DoubleBenchmarkComparator
from mindformers.parallel_core.training_graph.transformer.identity_op import IdentityOp
from mindformers.parallel_core.training_graph.transformer.mlp import MLP, MLPSubmodules
from mindformers.parallel_core.training_graph.transformer.transformer_block import TransformerBlock
from mindformers.parallel_core.training_graph.transformer.transformer_layer import TransformerLayerSubmodules, \
    TransformerLayer
from mindformers.parallel_core.training_graph.device_matrix import layout
from mindformers.parallel_core.utils.spec_utils import ModuleSpec
from mindformers.parallel_core.training_graph.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from mindformers.parallel_core.training_graph.transformer.attention import SharedKVCrossAttention, SharedKVCrossAttentionSubmodules
from mindformers.parallel_core.training_graph.transformer.flash_attention import FlashAttention
from mindformers.parallel_core.training_graph.transformer.norm import RMSNorm
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.core.context.build_context import build_context


class TestTransFormersBlock:
    """A test class for testing TransFormBlock with SharedKVCrossAttention"""

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
            normalization="RMSNorm",
            model_architecture="yoco",
            num_encoder_layers=0,
            num_decoder_layers=1,
            hidden_act="swiglu",
            gated_linear_unit=True
        )

        build_context({"use_legacy": False})
        ms.context.set_context(deterministic="ON")
        ms.set_context(mode=ms.GRAPH_MODE)

        submodules = SharedKVCrossAttentionSubmodules(
            linear_q=ColumnParallelLinear,
            core_attention=FlashAttention,
            linear_proj=RowParallelLinear,
        )
        layout.init_layout(self.config)
        layer_submodules = TransformerLayerSubmodules(
            input_layernorm=IdentityOp,
            cross_attention=ModuleSpec(
                module=SharedKVCrossAttention,
                submodules=submodules,
            ),
            pre_cross_attn_layernorm=RMSNorm,
            self_attention=IdentityOp,
            pre_mlp_layernorm=RMSNorm,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear)
            )
        )

        self.submodules_spec = ModuleSpec(
            module=TransformerLayer,
            submodules=layer_submodules
        )

        self.net = TransformerBlock(
            config=self.config,
            spec=self.submodules_spec,
            post_layer_norm=False
        )

        self.inputs, weight_dict = get_init_block_params(self.config)
        self.net.load_state_dict(weight_dict, strict=False)
        self.hidden_states = self.inputs.get("hidden_states")
        self.attention_mask = self.inputs.get("attention_mask")
        self.rotary_pos_emb = self.inputs.get("rotary_pos_emb")

    def run_test(self, accuracy=True, compare_type=None):
        """Helper function to run test and check results"""

        output, _ = self.net(self.hidden_states,
                          attention_mask=self.attention_mask,
                          rotary_pos_emb=self.rotary_pos_emb
                          )
        npu_output = output.asnumpy()
        if accuracy:
            gpu_output = GPU_DATA[compare_type]
            golden_output = GOLDEN_DATA[compare_type]
            assert DoubleBenchmarkComparator.check_pass_or_not(npu_output, gpu_output, golden_output), (
                f"TransformerBlock with SharedKVCrossAttention compare_type={compare_type} test failed.\n"
                f"NPU output:\n{npu_output}\n\n"
                f"GPU output:\n{gpu_output}\n\n"
                f"Golden output:\n{golden_output}"
            )


    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_transformer_block_with_cache_cross_attention(self):
        """
        Feature: TransformerBlock
        Description: Test Case: TransformerBlock with SharedKVCrossAttention
        """
        self.run_test(compare_type="output_block")
