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
test convert_utils
"""
import numpy as np
import pytest

from mindformers.utils.convert_utils import is_lora_param, qkv_concat_hf2mg, ffn_concat_hf2mg


class TestIsLoraParam:
    """ A test class for testing is_lora_param."""
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_lora_in_key_lowercase(self):
        """Test with 'lora' in lowercase in the key"""
        assert is_lora_param("model.lora.weight") is True

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_no_lora_in_key(self):
        """Test with no 'lora' in the key"""
        assert is_lora_param("model.linear.weight") is False


class TestQkvConcatHf2Mg:
    """ A test class for testing qkv_concat_hf2mg."""
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_qkv_concat_2d_array(self):
        """Test with 2D array input"""
        hidden_size = 768
        num_heads = 12
        n_kv_heads = 12
        n_rep = num_heads // n_kv_heads
        q_channel = hidden_size
        kv_channel = hidden_size // n_rep
        total_channels = q_channel + 2 * kv_channel
        qkv_weights = np.random.rand(total_channels, 1024).astype(np.float32)
        result = qkv_concat_hf2mg(qkv_weights, num_heads, n_kv_heads, hidden_size)
        assert result.shape == qkv_weights.shape

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_qkv_concat_1d_array(self):
        """Test with 1D array input (bias case)"""
        hidden_size = 768
        num_heads = 12
        n_kv_heads = 12
        n_rep = num_heads // n_kv_heads
        q_channel = hidden_size
        kv_channel = hidden_size // n_rep
        total_channels = q_channel + 2 * kv_channel
        qkv_weights = np.random.rand(total_channels).astype(np.float32)
        result = qkv_concat_hf2mg(qkv_weights, num_heads, n_kv_heads, hidden_size)
        assert result.shape == qkv_weights.shape

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_qkv_concat_3d_array_raises_error(self):
        """Test with 3D array input should raise ValueError"""
        qkv_weights = np.random.rand(10, 20, 30).astype(np.float32)
        num_heads = 12
        n_kv_heads = 12
        hidden_size = 768
        with pytest.raises(ValueError, match="qkv_weights shape is not supported."):
            qkv_concat_hf2mg(qkv_weights, num_heads, n_kv_heads, hidden_size)


class TestFfnConcatHf2Mg:
    """ A test class for testing ffn_concat_hf2mg."""
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_ffn_concat_basic_case(self):
        """Test basic FFN concat conversion"""
        ffn_weights = np.array([
            [1, 2, 3],  # gate weights part 1
            [4, 5, 6],  # gate weights part 2
            [7, 8, 9],  # hidden weights part 1
            [10, 11, 12]  # hidden weights part 2
        ], dtype=np.float32)
        ffn_hidden_size = 2
        result = ffn_concat_hf2mg(ffn_weights, ffn_hidden_size)
        assert result.shape == ffn_weights.shape
        assert isinstance(result, np.ndarray)
