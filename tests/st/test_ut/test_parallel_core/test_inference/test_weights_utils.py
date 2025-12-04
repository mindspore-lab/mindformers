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
"""
test weights_utils.py
"""
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

from mindformers.parallel_core.inference.weights_utils import (deal_training_qkv_weight, deal_training_ffn_weight,
                                                               deal_training_moe_weight,
                                                               make_expert_params_mapping_with_expert_dim,
                                                               split_fusion_loaded_weight)


class TestDealTrainingQkvWeight:
    """Test class for testing deal_training_qkv_weight."""
    @pytest.fixture
    def mock_config(self):
        """Create a mock config object"""
        config = MagicMock()
        config.kv_channels = None
        config.hidden_size = 768
        config.num_attention_heads = 12
        config.num_query_groups = 12
        return config

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_2d_weight_tp_size(self, mock_config):
        """Test with 2D weight and tensor parallel size 2"""
        head_dim = mock_config.hidden_size // mock_config.num_attention_heads
        q_channel = mock_config.num_attention_heads * head_dim
        kv_channel = mock_config.num_query_groups * head_dim
        total_channels = q_channel + 2 * kv_channel

        weight = np.random.rand(total_channels, 1024).astype(np.float32)

        with patch('mindformers.parallel_core.inference.parallel_state.get_tensor_model_parallel_world_size',
                   return_value=2),  \
                patch('mindformers.parallel_core.inference.parallel_state.get_tensor_model_parallel_rank',
                      return_value=0), \
                patch('mindformers.parallel_core.inference.weights_utils.split_loaded_weight') as mock_split:
            def mock_split_side_effect(w, axis, start, size):
                if axis == 0:
                    return w[start:start + size, :]
                return w

            mock_split.side_effect = mock_split_side_effect
            result = deal_training_qkv_weight(weight, mock_config)
            assert result.ndim == 2
            assert result.shape[0] == weight.shape[0]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_tp_size_greater_than_num_query_groups(self, mock_config):
        """Test when tensor parallel size is greater than number of query groups"""
        mock_config.num_query_groups = 2
        head_dim = mock_config.hidden_size // mock_config.num_attention_heads
        q_channel = mock_config.num_attention_heads * head_dim
        kv_channel = mock_config.num_query_groups * head_dim
        total_channels = q_channel + 2 * kv_channel
        weight = np.random.rand(total_channels, 1024).astype(np.float32)
        with patch('mindformers.parallel_core.inference.parallel_state.get_tensor_model_parallel_world_size',
                   return_value=4), \
                patch('mindformers.parallel_core.inference.parallel_state.get_tensor_model_parallel_rank',
                      return_value=1), \
                patch('mindformers.parallel_core.inference.weights_utils.split_loaded_weight') as mock_split:
            def mock_split_side_effect(w, axis, start, size):
                if axis == 0:
                    return w[start:start + size, :]
                return w
            mock_split.side_effect = mock_split_side_effect
            result = deal_training_qkv_weight(weight, mock_config)
            assert result is not None
            assert result.ndim == 2


class TestDealTrainingFfnWeight:
    """Test class for testing deal_training_ffn_weight."""
    @pytest.fixture
    def mock_config(self):
        """Create a mock config object"""
        config = MagicMock()
        config.ffn_hidden_size = 4096
        return config

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_1d_weight_tp_size_2(self, mock_config):
        """Test with 1D weight and tensor parallel size 2"""
        w = mock_config.ffn_hidden_size * 2  # For W1 and W3
        weight = np.random.rand(w).astype(np.float32)
        with patch('mindformers.parallel_core.inference.parallel_state.get_tensor_model_parallel_world_size',
                   return_value=2), \
                patch('mindformers.parallel_core.inference.parallel_state.get_tensor_model_parallel_rank',
                      return_value=0), \
                patch('mindformers.parallel_core.inference.weights_utils.split_loaded_weight') as mock_split:
            def mock_split_side_effect(w, axis, start, size):
                if axis == 0:
                    return w[start:start + size]
                return w

            mock_split.side_effect = mock_split_side_effect
            result = deal_training_ffn_weight(weight, mock_config)
            assert result.ndim == 1
            assert result.shape[0] == weight.shape[0]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_2d_weight_tp_size_2(self, mock_config):
        """Test with 2D weight and tensor parallel size 2"""
        w = mock_config.ffn_hidden_size * 2
        h = 1024
        weight = np.random.rand(w, h).astype(np.float32)

        with patch('mindformers.parallel_core.inference.parallel_state.get_tensor_model_parallel_world_size',
                   return_value=2), \
                patch('mindformers.parallel_core.inference.parallel_state.get_tensor_model_parallel_rank',
                      return_value=0), \
                patch('mindformers.parallel_core.inference.weights_utils.split_loaded_weight') as mock_split:
            def mock_split_side_effect(w, axis, start, size):
                if axis == 0:
                    return w[start:start + size, :]
                return w
            mock_split.side_effect = mock_split_side_effect
            result = deal_training_ffn_weight(weight, mock_config)
            assert result.ndim == 2
            assert result.shape[0] == weight.shape[0]
            assert result.shape[1] == weight.shape[1]


class TestDealTrainingMoeWeight:
    """Test class for testing deal_training_moe_weight."""
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_moe_weight_tp_size_4(self):
        """Test with tensor parallel size 4"""
        w = 2048
        h = 1024
        weight = np.random.rand(h, w).astype(np.float32)
        with patch('mindformers.parallel_core.inference.parallel_state.get_tensor_model_parallel_world_size',
                   return_value=4), \
                patch('mindformers.parallel_core.inference.parallel_state.get_tensor_model_parallel_rank',
                      return_value=1), \
                patch('mindformers.parallel_core.inference.weights_utils.split_loaded_weight') as mock_split:
            def mock_split_side_effect(w, axis, start, size):
                if axis == 1:
                    return w[:, start:start + size]
                return w
            mock_split.side_effect = mock_split_side_effect
            result = deal_training_moe_weight(weight)
            assert result.shape[0] == weight.shape[0]
            assert result.shape[1] == weight.shape[1]


class TestMakeExpertParamsMappingWithExpertDim:
    """Test class for testing make_expert_params_mapping_with_expert_dim."""
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_make_expert_params_mapping_basic(self):
        """Test basic expert parameter mapping generation"""
        ckpt_gate_proj_name = "gate_proj"
        ckpt_down_proj_name = "down_proj"
        ckpt_up_proj_name = "up_proj"
        result = make_expert_params_mapping_with_expert_dim(
            ckpt_gate_proj_name, ckpt_down_proj_name, ckpt_up_proj_name
        )
        assert len(result) == 3
        for param_tuple in result:
            assert len(param_tuple) == 3
            assert isinstance(param_tuple[0], str)
            assert isinstance(param_tuple[1], str)
            assert isinstance(param_tuple[2], str)

        expected_shard_ids = ['w1', 'w2', 'w3']
        actual_shard_ids = [item[2] for item in result]
        assert actual_shard_ids == expected_shard_ids

        for param_tuple in result:
            weight_name = param_tuple[1]
            weight_prefix = param_tuple[0]
            if 'gate_proj' in weight_name or 'up_proj' in weight_name:
                assert weight_prefix == "experts.weight1"
            else:
                assert weight_prefix == "experts.weight2"


class TestSplitFusionLoadedWeight:
    """Test class for testing split_fusion_loaded_weight."""
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_split_fusion_loaded_weight_2d(self):
        """Test with 2D array input"""
        loaded_weight = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18]
        ], dtype=np.float32)

        start_idxs = [0, 2, 4]
        shard_sizes = [2, 2, 2]
        result = split_fusion_loaded_weight(loaded_weight, start_idxs, shard_sizes)
        expected = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18]
        ], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
