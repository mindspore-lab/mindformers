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
"""test model_mixin.py"""
import threading
from unittest.mock import patch, MagicMock
import numpy as np
import pytest
from mindspore import Tensor
import mindspore.common.dtype as mstype

from mindformers.parallel_core.utils.model_mixin import ModelMixin, TrainModelMixin


class TestModelMixin:
    """Test ModelMixin class"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_init(self):
        """Test __init__ method"""
        mixin = ModelMixin()
        assert mixin.transformer_config is None

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_convert_concat_name(self):
        """Test convert_concat_name method"""
        mixin = ModelMixin()

        # Patch the is_legacy_model function correctly
        with patch('mindformers.parallel_core.utils.model_mixin.is_legacy_model', return_value=False):
            # Test with linear_q
            assert mixin.convert_concat_name(".linear_q.") == ".linear_qkv."
            # Test with linear_k
            assert mixin.convert_concat_name(".linear_k.") == ".linear_qkv."
            # Test with linear_v
            assert mixin.convert_concat_name(".linear_v.") == ".linear_qkv."
            # Test with mlp.gating
            assert mixin.convert_concat_name(".mlp.gating.") == ".mlp.linear_fc1."
            # Test with mlp.hidden
            assert mixin.convert_concat_name(".mlp.hidden.") == ".mlp.linear_fc1."
            # Test with unchanged name
            assert mixin.convert_concat_name(".other.") == ".other."

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_convert_name(self):
        """Test convert_name method"""

        # Create a subclass with weight_mapping
        class TestModel(ModelMixin):
            weight_mapping = [("hf_name", "mcore_name")]

        mixin = TestModel()
        with patch('mindformers.parallel_core.utils.model_mixin.is_legacy_model', return_value=False):
            assert mixin.convert_name("test.hf_name.test") == "test.mcore_name.test"
            assert mixin.convert_name("test.other.test") == "test.other.test"

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_set_dynamic_inputs(self):
        """Test set_dynamic_inputs method"""
        mixin = ModelMixin()
        with pytest.raises(RuntimeError):
            mixin.set_dynamic_inputs()

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_convert_to_transformer_config(self):
        """Test convert_to_transformer_config method"""
        mixin = ModelMixin()

        # Mock the convert_to_transformer_config function
        mock_transformer_config = MagicMock()
        with patch('mindformers.parallel_core.utils.model_mixin.convert_to_transformer_config',
                   return_value=mock_transformer_config):
            # Create a simple config dict
            config = {
                "hidden_size": 768,
                "num_attention_heads": 12
            }

            transformer_config = mixin.convert_to_transformer_config(config)
            assert transformer_config is mock_transformer_config
            assert mixin.transformer_config is mock_transformer_config

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_gpt_transformer_config(self):
        """Test get_gpt_transformer_config method"""
        mixin = ModelMixin()

        # Test without converting first
        with pytest.raises(ValueError):
            mixin.get_gpt_transformer_config()

        # Test after converting
        mock_transformer_config = MagicMock()
        mixin.transformer_config = mock_transformer_config
        assert mixin.get_gpt_transformer_config() == mock_transformer_config

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_is_mtp_model(self):
        """Test is_mtp_model method"""
        mixin = ModelMixin()

        # Test with mtp_num_layers = 0
        class TestConfig1:
            def __init__(self):
                self.mtp_num_layers = 0

        mixin.transformer_config = TestConfig1()
        assert not mixin.is_mtp_model()

        # Test with mtp_num_layers > 0
        class TestConfig2:
            def __init__(self):
                self.mtp_num_layers = 2

        mixin.transformer_config = TestConfig2()
        assert mixin.is_mtp_model()

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_is_moe_model(self):
        """Test is_moe_model method"""
        mixin = ModelMixin()

        # Test with num_moe_experts = 0
        class TestConfig1:
            def __init__(self):
                self.num_moe_experts = 0

        mixin.transformer_config = TestConfig1()
        assert not mixin.is_moe_model()

        # Test with num_moe_experts > 0
        class TestConfig2:
            def __init__(self):
                self.num_moe_experts = 8

        mixin.transformer_config = TestConfig2()
        assert mixin.is_moe_model()

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_gpt_model(self):
        """Test get_gpt_model method"""
        # Test without model attribute
        mixin = ModelMixin()
        with pytest.raises(RuntimeError):
            mixin.get_gpt_model()

        # Test with model attribute
        class TestModel(ModelMixin):
            def __init__(self):
                super().__init__()
                self.model = "test_model"

        mixin = TestModel()
        assert mixin.get_gpt_model() == "test_model"

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_convert_weight_dict(self):
        """Test convert_weight_dict method"""
        mixin = ModelMixin()
        with pytest.raises(RuntimeError):
            mixin.convert_weight_dict({})

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_convert_map_dict(self):
        """Test convert_map_dict method"""
        mixin = ModelMixin()
        with pytest.raises(RuntimeError):
            mixin.convert_map_dict({})

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_obtain_name_map(self):
        """Test obtain_name_map method"""
        mixin = ModelMixin()
        with pytest.raises(RuntimeError):
            mixin.obtain_name_map([])

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_obtain_qkv_ffn_concat_keys(self):
        """Test obtain_qkv_ffn_concat_keys method"""
        mixin = ModelMixin()
        # This method should not raise any exceptions
        result = mixin.obtain_qkv_ffn_concat_keys()
        # The method doesn't return anything, so just check that it runs without errors
        assert result is None


class TestTrainModelMixin:
    """Test TrainModelMixin class"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_concat_qkv_contiguous(self):
        """Test concat_qkv_contiguous method"""
        mixin = TrainModelMixin()

        # Create fake weights
        q_value = np.random.rand(10, 20)
        k_value = np.random.rand(10, 20)
        v_value = np.random.rand(10, 20)
        q_name = "test.linear_q.weight"

        result = mixin.concat_qkv_contiguous(q_value, k_value, v_value, q_name)
        assert "test.linear_qkv.weight" in result
        assert result["test.linear_qkv.weight"].shape == (30, 20)
        # Check if values are concatenated correctly
        assert np.array_equal(result["test.linear_qkv.weight"][:10], q_value)
        assert np.array_equal(result["test.linear_qkv.weight"][10:20], k_value)
        assert np.array_equal(result["test.linear_qkv.weight"][20:], v_value)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_concat_qkv_interleaved(self):
        """Test concat_qkv_interleaved method"""
        mixin = TrainModelMixin()

        # Create fake weights
        head_dim = 5
        n_kv_heads = 2
        num_attention_heads = 4
        hidden_size = head_dim * num_attention_heads

        q_value = np.random.rand(hidden_size, hidden_size)
        k_value = np.random.rand(head_dim * n_kv_heads, hidden_size)
        v_value = np.random.rand(head_dim * n_kv_heads, hidden_size)
        q_name = "test.linear_q.weight"

        result = mixin.concat_qkv_interleaved(q_value, k_value, v_value, q_name,
                                              head_dim, n_kv_heads, num_attention_heads)
        assert "test.linear_qkv.weight" in result
        assert result["test.linear_qkv.weight"].shape == (
            (num_attention_heads // n_kv_heads + 2) * head_dim * n_kv_heads, hidden_size)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_concat_linear_fc1_contiguous(self):
        """Test concat_linear_fc1_contiguous method"""
        mixin = TrainModelMixin()

        # Create fake weights
        gate_value = np.random.rand(10, 20)
        up_value = np.random.rand(10, 20)
        gate_name = "test.mlp.gating.weight"

        result = mixin.concat_linear_fc1_contiguous(gate_value, up_value, gate_name)
        assert "test.mlp.linear_fc1.weight" in result
        assert result["test.mlp.linear_fc1.weight"].shape == (20, 20)
        # Check if values are concatenated correctly
        assert np.array_equal(result["test.mlp.linear_fc1.weight"][:10], gate_value)
        assert np.array_equal(result["test.mlp.linear_fc1.weight"][10:], up_value)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_concat_linear_fc1_interleaved(self):
        """Test concat_linear_fc1_interleaved method"""
        mixin = TrainModelMixin()

        # Create fake weights
        ffn_hidden_size = 10
        gate_value = np.random.rand(ffn_hidden_size, 20)
        up_value = np.random.rand(ffn_hidden_size, 20)
        gate_name = "test.mlp.gating.weight"

        result = mixin.concat_linear_fc1_interleaved(gate_value, up_value, gate_name, ffn_hidden_size)
        assert "test.mlp.linear_fc1.weight" in result
        assert result["test.mlp.linear_fc1.weight"].shape == (20, 20)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_concat_qkv_weight_infer(self):
        """Test concat_qkv_weight_infer method"""
        mixin = TrainModelMixin()

        # Create fake weights and dicts
        condition = threading.Condition()
        qkv_weight_dict = {}
        ms_weight_dict = {
            "test.linear_q.weight": np.random.rand(10, 20),
            "test.linear_k.weight": np.random.rand(10, 20),
            "test.linear_v.weight": np.random.rand(10, 20)
        }

        # Test with all keys present
        mixin.concat_qkv_weight_infer(
            ["test.linear_q.weight"],
            ["test.linear_k.weight"],
            ["test.linear_v.weight"],
            qkv_weight_dict,
            condition,
            ms_weight_dict
        )
        assert "test.linear_qkv.weight" in ms_weight_dict
        assert ms_weight_dict["test.linear_qkv.weight"].shape == (30, 20)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_concat_ffn_weight_infer(self):
        """Test concat_ffn_weight_infer method"""
        mixin = TrainModelMixin()

        # Create fake weights and dicts
        condition = threading.Condition()
        ffn_weight_dict = {}
        ms_weight_dict = {
            "test.mlp.gating.weight": np.random.rand(10, 20),
            "test.mlp.hidden.weight": np.random.rand(10, 20)
        }

        # Test with all keys present
        mixin.concat_ffn_weight_infer(
            ["test.mlp.gating.weight"],
            ["test.mlp.hidden.weight"],
            ffn_weight_dict,
            condition,
            ms_weight_dict
        )
        assert "test.mlp.linear_fc1.weight" in ms_weight_dict
        assert ms_weight_dict["test.mlp.linear_fc1.weight"].shape == (20, 20)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_check_and_get_model(self):
        """Test check_and_get_model method"""
        # Test without model attribute
        mixin = TrainModelMixin()
        with pytest.raises(RuntimeError):
            mixin.check_and_get_model()

        # Test with model attribute
        class TestModel(TrainModelMixin):
            def __init__(self):
                super().__init__()
                self.model = "test_model"

        mixin = TestModel()
        assert mixin.check_and_get_model() == "test_model"

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_prepare_inputs_for_predict_layout(self):
        """Test prepare_inputs_for_predict_layout method"""
        mixin = TrainModelMixin()
        input_ids = [1, 2, 3, 4, 5]
        result = mixin.prepare_inputs_for_predict_layout(input_ids)

        # Check result types
        assert isinstance(result[0], Tensor)
        assert result[0].dtype == mstype.int32
        assert result[0].shape == (5,)
        # Check that other return values are None
        for i in range(1, 9):
            assert result[i] is None

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_model_parameters(self):
        """Test get_model_parameters method"""

        # Create a mock model with get_model_parameters method
        class MockModel:
            def get_model_parameters(self):
                return ["param1", "param2"]

        class TestModel(TrainModelMixin):
            def __init__(self):
                super().__init__()
                self.model = MockModel()

        mixin = TestModel()
        assert mixin.get_model_parameters() == ["param1", "param2"]

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_make_model_muon_fns(self):
        """Test make_model_muon_fns method"""

        # Create a mock model with make_model_muon_fns method
        class MockModel:
            def make_model_muon_fns(self):
                return ["fn1", "fn2"]

        class TestModel(TrainModelMixin):
            def __init__(self):
                super().__init__()
                self.model = MockModel()

        mixin = TestModel()
        assert mixin.make_model_muon_fns() == ["fn1", "fn2"]

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_muon_filter(self):
        """Test get_muon_filter method"""

        # Create a mock model with get_muon_filter method
        class MockModel:
            def get_muon_filter(self):
                return "filter"

        class TestModel(TrainModelMixin):
            def __init__(self):
                super().__init__()
                self.model = MockModel()

        mixin = TestModel()
        assert mixin.get_muon_filter() == "filter"

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_tp_dims(self):
        """Test get_tp_dims method"""

        # Create a mock model with get_tp_dims method
        class MockModel:
            def get_tp_dims(self, parameters):
                return {param: f"dim_{param}" for param in parameters}

        class TestModel(TrainModelMixin):
            def __init__(self):
                super().__init__()
                self.model = MockModel()

        mixin = TestModel()
        assert mixin.get_tp_dims(["param1", "param2"]) == {"param1": "dim_param1", "param2": "dim_param2"}

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_op_groups_info(self):
        """Test get_op_groups_info method"""

        # Create a mock model with get_op_groups_info method
        class MockModel:
            # pylint: disable=W0613
            def get_op_groups_info(self, parameters, op_size, tp_group, op_group):
                return f"info_{op_size}"

        class TestModel(TrainModelMixin):
            def __init__(self):
                super().__init__()
                self.model = MockModel()

        mixin = TestModel()
        assert mixin.get_op_groups_info(None, 2, None, None) == "info_2"

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_parallel_config_for_muon(self):
        """Test get_parallel_config_for_muon method"""

        # Create a mock model with get_parallel_config_for_muon method
        class MockModel:
            def get_parallel_config_for_muon(self):
                return {"config": "parallel"}

        class TestModel(TrainModelMixin):
            def __init__(self):
                super().__init__()
                self.model = MockModel()

        mixin = TestModel()
        assert mixin.get_parallel_config_for_muon() == {"config": "parallel"}

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_param_layer_indices(self):
        """Test get_param_layer_indices method"""

        # Create a mock model with get_param_layer_indices method
        class MockModel:
            def get_param_layer_indices(self, parameters):
                return {param: i for i, param in enumerate(parameters)}

        class TestModel(TrainModelMixin):
            def __init__(self):
                super().__init__()
                self.model = MockModel()

        mixin = TestModel()
        assert mixin.get_param_layer_indices(["param1", "param2"]) == {"param1": 0, "param2": 1}

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_apply_qk_clip_scaling(self):
        """Test apply_qk_clip_scaling method"""

        # Create a mock model with apply_qk_clip_scaling method
        class MockModel:
            # pylint: disable=W0613
            def apply_qk_clip_scaling(self, parameters, param_names, param_layers,logit_threshold, split_fn, merge_fn):
                return f"scaled_{logit_threshold}"

        class TestModel(TrainModelMixin):
            def __init__(self):
                super().__init__()
                self.model = MockModel()

        mixin = TestModel()
        assert mixin.apply_qk_clip_scaling(None, None, None, 1.0, None, None) == "scaled_1.0"

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_concat_qkv_weight_megatron(self):
        """Test concat_qkv_weight_megatron method"""
        mixin = TrainModelMixin()

        # Create fake weights with correct shapes
        condition = threading.Condition()
        qkv_weight_dict = {}

        # Calculate expected shapes
        head_dim = 5
        n_kv_heads = 2
        num_attention_heads = 4
        n_rep = num_attention_heads // n_kv_heads

        # Create weights with compatible shapes
        # For q: shape should be (n_kv_heads * n_rep * head_dim, hidden_size)
        # For k and v: shape should be (n_kv_heads * head_dim, hidden_size)
        hidden_size = 10
        q_value = np.random.rand(n_kv_heads * n_rep * head_dim, hidden_size)
        k_value = np.random.rand(n_kv_heads * head_dim, hidden_size)
        v_value = np.random.rand(n_kv_heads * head_dim, hidden_size)

        ms_weight_dict = {
            "test.linear_q.weight": q_value,
            "test.linear_k.weight": k_value,
            "test.linear_v.weight": v_value
        }

        # Test with all keys present
        mixin.concat_qkv_weight_megatron(
            ["test.linear_q.weight"],
            ["test.linear_k.weight"],
            ["test.linear_v.weight"],
            qkv_weight_dict,
            condition,
            ms_weight_dict,
            head_dim=head_dim,
            n_kv_heads=n_kv_heads,
            num_attention_heads=num_attention_heads
        )
        assert "test.linear_qkv.weight" in ms_weight_dict

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_concat_ffn_weight_megatron(self):
        """Test concat_ffn_weight_megatron method"""
        mixin = TrainModelMixin()

        # Create fake weights and dicts
        condition = threading.Condition()
        ffn_weight_dict = {}
        ms_weight_dict = {
            "test.mlp.gating.weight": np.random.rand(10, 20),
            "test.mlp.hidden.weight": np.random.rand(10, 20)
        }

        # Test with all keys present
        mixin.concat_ffn_weight_megatron(
            ["test.mlp.gating.weight"],
            ["test.mlp.hidden.weight"],
            ffn_weight_dict,
            condition,
            ms_weight_dict,
            ffn_hidden_size=10
        )
        assert "test.mlp.linear_fc1.weight" in ms_weight_dict

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_concat_expert_weight(self):
        """Test concat_expert_weight method"""
        mixin = TrainModelMixin()

        # Create fake weights and dicts
        condition = threading.Condition()
        expert_weight_dict = {}

        # Create a mock transformer config
        class MockTransformerConfig:
            def __init__(self):
                self.hidden_size = 20
                self.moe_ffn_hidden_size = 40

        mixin.transformer_config = MockTransformerConfig()

        # Create fake weights
        num_layers = 1
        num_experts = 2
        ms_weight_dict = {}
        for layer in range(num_layers):
            for expert_idx in range(num_experts):
                ms_weight_dict[f"decoder.layers.{layer}.mlp.experts.{expert_idx}.linear_fc1.weight"] = np.random.rand(
                    40, 20)
                ms_weight_dict[f"decoder.layers.{layer}.mlp.experts.{expert_idx}.linear_fc2.weight"] = np.random.rand(
                    20, 40)

        # Get w2_keys
        w2_keys = [k for k in ms_weight_dict if "linear_fc2" in k]

        # Test expert weight concatenation
        mixin.concat_expert_weight(
            w2_keys,
            expert_weight_dict,
            condition,
            ms_weight_dict,
            num_layers,
            num_experts
        )

        # Check if concatenated weights are created
        assert "decoder.layers.0.mlp.experts.weight1" in ms_weight_dict
        assert "decoder.layers.0.mlp.experts.weight2" in ms_weight_dict
