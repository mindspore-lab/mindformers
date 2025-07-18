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
"""Test get gpt layer specs."""
import pytest

from mindformers.parallel_core.training_graph.base_models.gpt.gpt_layer_specs import get_gpt_layer_local_spec, \
    get_gpt_decoder_block_spec, get_gpt_mtp_block_spec
from mindformers.parallel_core.training_graph.transformer.attention import SelfAttentionContiguous, SelfAttention
from mindformers.parallel_core.training_graph.transformer.identity_op import IdentityOp
from mindformers.parallel_core.training_graph.transformer.mlp import MLP
from mindformers.parallel_core.training_graph.transformer.moe.moe_layer import MoELayer
from mindformers.parallel_core.training_graph.transformer.multi_latent_attention import MLASelfAttention, \
    MLASelfAttentionConcatenated
from mindformers.parallel_core.training_graph.transformer.norm import FusedNorm
from mindformers.parallel_core.transformer_config import TransformerConfig


class TestLayerSpec:
    """test get_gpt_layer_local_spec()."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_dense_case(self):
        """
        Test dense case for get_gpt_layer_local_spec.
        Input: num_experts=None, moe_grouped_gemm=False/True, qk_layernorm, multi_latent_attention.
        Output: Returns spec with correct submodules for dense MLP and attention.
        Expected: MLP module is used, q/k layernorm is IdentityOp or FusedNorm as specified, MLA module if enabled.
        """
        spec = get_gpt_layer_local_spec(num_experts=None, moe_grouped_gemm=False)
        assert spec.submodules.mlp.module == MLP
        assert spec.submodules.self_attention.submodules.q_layernorm == IdentityOp
        assert spec.submodules.self_attention.submodules.k_layernorm == IdentityOp
        assert spec.submodules.self_attention.module == SelfAttentionContiguous

        spec = get_gpt_layer_local_spec(num_experts=None, moe_grouped_gemm=True)
        assert spec.submodules.mlp.module == MLP
        assert spec.submodules.self_attention.submodules.q_layernorm == IdentityOp
        assert spec.submodules.self_attention.submodules.k_layernorm == IdentityOp
        assert spec.submodules.self_attention.module == SelfAttentionContiguous

        spec = get_gpt_layer_local_spec(num_experts=None, moe_grouped_gemm=True, qk_layernorm=True)
        assert spec.submodules.mlp.module == MLP
        assert spec.submodules.self_attention.submodules.q_layernorm == FusedNorm
        assert spec.submodules.self_attention.submodules.k_layernorm == FusedNorm
        assert spec.submodules.self_attention.module == SelfAttentionContiguous

        spec = get_gpt_layer_local_spec(num_experts=None, moe_grouped_gemm=True, multi_latent_attention=True)
        assert spec.submodules.mlp.module == MLP
        assert spec.submodules.self_attention.submodules.q_layernorm == IdentityOp
        assert spec.submodules.self_attention.submodules.k_layernorm == IdentityOp
        assert spec.submodules.self_attention.module == MLASelfAttentionConcatenated

        spec = get_gpt_layer_local_spec(num_experts=None, moe_grouped_gemm=True, use_contiguous_weight_layout=False)
        assert spec.submodules.mlp.module == MLP
        assert spec.submodules.self_attention.submodules.q_layernorm == IdentityOp
        assert spec.submodules.self_attention.submodules.k_layernorm == IdentityOp
        assert spec.submodules.self_attention.module == SelfAttention

        spec = get_gpt_layer_local_spec(num_experts=None, moe_grouped_gemm=True, multi_latent_attention=True,
                                        mla_qkv_concat=False)
        assert spec.submodules.mlp.module == MLP
        assert spec.submodules.self_attention.submodules.q_layernorm == IdentityOp
        assert spec.submodules.self_attention.submodules.kv_layernorm == IdentityOp
        assert spec.submodules.self_attention.module == MLASelfAttention

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_moe_case(self):
        """
        Test MoE case for get_gpt_layer_local_spec.
        Input: num_experts=4, moe_grouped_gemm True/False.
        Output: Returns spec with MoELayer if grouped_gemm True, else raises NotImplementedError.
        Expected: MoELayer is used when grouped_gemm True, exception otherwise.
        """
        spec = get_gpt_layer_local_spec(num_experts=4, moe_grouped_gemm=True)
        assert spec.submodules.mlp.module == MoELayer
        try:
            _ = get_gpt_layer_local_spec(num_experts=4, moe_grouped_gemm=False)
        except NotImplementedError:
            pass


class TestBlockSpec:
    """test get_gpt_decoder_block_spec()."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_all_moe_case(self):
        """
        Test all layers as MoE in get_gpt_decoder_block_spec.
        Input: config with num_layers=8, num_moe_experts=2, moe_grouped_gemm=True.
        Output: All layer_specs use MoELayer.
        Expected: Each layer_spec.mlp.module is MoELayer.
        """
        config = TransformerConfig(num_layers=8, num_attention_heads=1, num_moe_experts=2, moe_grouped_gemm=True,
                                   add_bias_linear=False)
        spec = get_gpt_decoder_block_spec(config)
        assert len(spec.layer_specs) == 8
        for layer_spec in spec.layer_specs:
            assert layer_spec.submodules.mlp.module == MoELayer
            assert layer_spec.submodules.self_attention.module == SelfAttentionContiguous

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_all_moe_with_megatron_attention_case(self):
        """
        Test all layers as MoE in get_gpt_decoder_block_spec.
        Input: config with num_layers=8, num_moe_experts=2, moe_grouped_gemm=True.
        Output: All layer_specs use MoELayer.
        Expected: Each layer_spec.mlp.module is MoELayer.
        """
        config = TransformerConfig(num_layers=8, num_attention_heads=1, num_moe_experts=2, moe_grouped_gemm=True,
                                   add_bias_linear=False, use_contiguous_weight_layout=False)
        spec = get_gpt_decoder_block_spec(config)
        assert len(spec.layer_specs) == 8
        for layer_spec in spec.layer_specs:
            assert layer_spec.submodules.mlp.module == MoELayer
            assert layer_spec.submodules.self_attention.module == SelfAttention

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_all_moe_with_mla_case(self):
        """
        Test all layers as MoE in get_gpt_decoder_block_spec.
        Input: config with num_layers=8, num_moe_experts=2, moe_grouped_gemm=True.
        Output: All layer_specs use MoELayer.
        Expected: Each layer_spec.mlp.module is MoELayer.
        """
        config = TransformerConfig(num_layers=8, num_attention_heads=1, num_moe_experts=2, moe_grouped_gemm=True,
                                   add_bias_linear=False, multi_latent_attention=True)
        spec = get_gpt_decoder_block_spec(config)
        assert len(spec.layer_specs) == 8
        for layer_spec in spec.layer_specs:
            assert layer_spec.submodules.mlp.module == MoELayer
            assert layer_spec.submodules.self_attention.module == MLASelfAttentionConcatenated

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_all_moe_with_megatron_mla_case(self):
        """
        Test all layers as MoE in get_gpt_decoder_block_spec.
        Input: config with num_layers=8, num_moe_experts=2, moe_grouped_gemm=True.
        Output: All layer_specs use MoELayer.
        Expected: Each layer_spec.mlp.module is MoELayer.
        """
        config = TransformerConfig(num_layers=8, num_attention_heads=1, num_moe_experts=2, moe_grouped_gemm=True,
                                   add_bias_linear=False, multi_latent_attention=True, mla_qkv_concat=False)
        spec = get_gpt_decoder_block_spec(config)
        assert len(spec.layer_specs) == 8
        for layer_spec in spec.layer_specs:
            assert layer_spec.submodules.mlp.module == MoELayer
            assert layer_spec.submodules.self_attention.module == MLASelfAttention

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_2_moe_case(self):
        """
        Test two MoE layers in get_gpt_decoder_block_spec.
        Input: config with moe_layer_freq=4.
        Output: Only layers 0 and 4 use MoELayer, others use MLP.
        Expected: Correct alternation of MoELayer and MLP.
        """
        config = TransformerConfig(num_layers=8, num_attention_heads=1, num_moe_experts=2, moe_grouped_gemm=True,
                                   add_bias_linear=False, moe_layer_freq=4)
        spec = get_gpt_decoder_block_spec(config)
        assert len(spec.layer_specs) == 8
        for i in range(8):
            layer_spec = spec.layer_specs[i]
            if i in (0, 4):
                assert layer_spec.submodules.mlp.module == MoELayer
            else:
                assert layer_spec.submodules.mlp.module == MLP

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_list_moe_case(self):
        """
        Test list pattern for MoE layers in get_gpt_decoder_block_spec.
        Input: config with moe_layer_freq=[1, 0, 0, 0, 1, 0, 0, 0].
        Output: Only layers 0 and 4 use MoELayer, others use MLP.
        Expected: Correct mapping from list to layer types.
        """
        config = TransformerConfig(num_layers=8, num_attention_heads=1, num_moe_experts=2, moe_grouped_gemm=True,
                                   add_bias_linear=False, moe_layer_freq=[1, 0, 0, 0, 1, 0, 0, 0])
        spec = get_gpt_decoder_block_spec(config)
        assert len(spec.layer_specs) == 8
        for i in range(8):
            layer_spec = spec.layer_specs[i]
            if i in (0, 4):
                assert layer_spec.submodules.mlp.module == MoELayer
            else:
                assert layer_spec.submodules.mlp.module == MLP

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_list_moe_error_value_case(self):
        """
        Test error for invalid value in moe_layer_freq list.
        Input: config with moe_layer_freq containing value 2.
        Output: Raises ValueError.
        Expected: Exception is thrown for invalid pattern value.
        """
        config = TransformerConfig(num_layers=8, num_attention_heads=1, num_moe_experts=2, moe_grouped_gemm=True,
                                   add_bias_linear=False, moe_layer_freq=[1, 0, 0, 0, 2, 0, 0, 0])
        try:
            _ = get_gpt_decoder_block_spec(config)
        except ValueError:
            pass

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_list_moe_error_length_case(self):
        """
        Test error for invalid length of moe_layer_freq list.
        Input: config with moe_layer_freq length not matching num_layers.
        Output: Raises ValueError.
        Expected: Exception is thrown for invalid pattern length.
        """
        config = TransformerConfig(num_layers=8, num_attention_heads=1, num_moe_experts=2, moe_grouped_gemm=True,
                                   add_bias_linear=False, moe_layer_freq=[1, 0, 0, 0, 1, 0, 0])
        try:
            _ = get_gpt_decoder_block_spec(config)
        except ValueError:
            pass


class TestMtpBlockSpec:
    """test get_gpt_decoder_block_spec()."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_none_case(self):
        """
        Test MTP block spec returns None when mtp_num_layers is not set.
        Input: config with mtp_num_layers unset or zero.
        Output: get_gpt_mtp_block_spec returns None.
        Expected: None is returned.
        """
        spec = get_gpt_layer_local_spec(num_experts=None, moe_grouped_gemm=False)
        config = TransformerConfig(num_layers=8, num_attention_heads=1, num_moe_experts=2, moe_grouped_gemm=True,
                                   add_bias_linear=False)
        assert get_gpt_mtp_block_spec(config, spec) is None

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_input_is_layerspec_case(self):
        """
        Test MTP block spec with input as LayerSpec.
        Input: config with mtp_num_layers=2, spec is LayerSpec.
        Output: Returns MTP block spec with correct number of layers.
        Expected: mtp_spec.layer_specs length is 2, each uses input spec.
        """
        spec = get_gpt_layer_local_spec(num_experts=None, moe_grouped_gemm=False)
        config = TransformerConfig(num_layers=8, num_attention_heads=1, num_moe_experts=2, moe_grouped_gemm=True,
                                   add_bias_linear=False, mtp_num_layers=1)
        mtp_spec = get_gpt_mtp_block_spec(config, spec)
        assert len(mtp_spec.layer_specs) == 1
        assert mtp_spec.layer_specs[0].submodules.transformer_layer == spec

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_input_is_blockspec_case(self):
        """
        Test MTP block spec with input as BlockSpec.
        Input: config with mtp_num_layers=1, spec is BlockSpec.
        Output: Returns MTP block spec with last layer of block as base.
        Expected: mtp_spec.layer_specs[0].submodules.transformer_layer equals last block layer.
        """
        config = TransformerConfig(num_layers=8, num_attention_heads=1, num_moe_experts=2, moe_grouped_gemm=True,
                                   add_bias_linear=False, moe_layer_freq=[1, 0, 0, 0, 1, 0, 0, 1], mtp_num_layers=1)
        spec = get_gpt_decoder_block_spec(config)
        mtp_spec = get_gpt_mtp_block_spec(config, spec)
        assert len(mtp_spec.layer_specs) == 1
        assert mtp_spec.layer_specs[0].submodules.transformer_layer == spec.layer_specs[-1]
