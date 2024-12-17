# Copyright 2024 Huawei Technologies Co., Ltd
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
Test module for testing the interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_glm2_model/test_glm2_transformer.py
"""
from unittest.mock import patch, MagicMock
import unittest
from mindformers.models.glm2.glm2_transformer import CoreAttention, ChatGLM2SelfAttention
import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor, nn, ops
ms.set_context(mode=ms.PYNATIVE_MODE)

class MockCoreAttention(nn.Cell):
    """Mock ChatGLM2 core attention."""
    def __init__(self):
        bsz = 1
        num_heads = 8
        seq_len = 1024
        hidden_size_per_head = 4096
        self.apply_query_key_layer_scaling = MagicMock(return_value=False)
        self.attention_softmax_in_fp32 = MagicMock(return_value=False)
        self.head_dim = MagicMock(return_value=128)
        self.n_head = MagicMock(return_value=num_heads)
        self.norm_factor = 1
        # Strided linear layer.
        drpt = nn.Dropout(keep_prob=1.0)
        self.attention_dropout = MagicMock(return_value=drpt)
        input_x = np.random.randn(bsz, num_heads, seq_len, seq_len).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        self.batch_matmul_q_k = MagicMock(return_value=input_x_tensor)
        self.mul_mask = MagicMock(return_value=input_x_tensor)
        self.add = MagicMock(return_value=input_x_tensor)
        input_x = np.random.randn(bsz, num_heads, seq_len, hidden_size_per_head).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        self.batch_matmul = MagicMock(return_value=input_x_tensor)
        input_x = np.random.randn(bsz, num_heads, seq_len, seq_len).astype(np.float32)
        input_x_tensor = Tensor(input_x, dtype=ms.float32)
        self.softmax = MagicMock(return_value=input_x_tensor)
        input_x = np.random.randn(bsz, seq_len, num_heads, hidden_size_per_head).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        self.merger_head_transpose = MagicMock(return_value=input_x_tensor)
        input_x = np.random.randn(bsz, seq_len, num_heads * hidden_size_per_head).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        self.reshape = MagicMock(return_value=input_x_tensor)
        self.compute_dtype = ms.float32
        self.multi_query_attention = MagicMock(return_value=True)
        if self.multi_query_attention:
            self.n_kv_head = MagicMock(return_value=2)
            self.qkv_hidden_size = MagicMock(return_value=4608)


class TestCoreAttention(unittest.TestCase):
    """A test for mock CoreAttention """
    @patch('mindformers.models.glm2.glm2_transformer.CoreAttention.__init__', MockCoreAttention.__init__)
    @pytest.mark.leve1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_construct(self):
        """Feature: construct interface.
        Description: a testcase for construct
        Expectation: success
        """
        bsz = 1
        seq_len = 1024
        num_heads = 8
        hidden_size_per_head = 4096
        q = np.random.randn(bsz, num_heads, seq_len, hidden_size_per_head).astype(np.float16)
        k = np.random.randn(bsz, num_heads, seq_len, hidden_size_per_head).astype(np.float16)
        v = np.random.randn(bsz, num_heads, seq_len, hidden_size_per_head).astype(np.float16)
        q_tensor = Tensor(q, dtype=ms.bfloat16)
        k_tensor = Tensor(k, dtype=ms.bfloat16)
        v_tensor = Tensor(v, dtype=ms.bfloat16)
        m_mask = np.random.randn(bsz, num_heads, seq_len, seq_len).astype(np.float16)
        mask_tensor = Tensor(m_mask, dtype=ms.bfloat16)
        config = None
        layer_number = 1
        coreattn = CoreAttention(config, layer_number)
        out = coreattn.construct(q_tensor, k_tensor, v_tensor, mask_tensor)
        self.assertEqual(out.shape, (bsz, seq_len, num_heads * hidden_size_per_head))

    @patch('mindformers.models.glm2.glm2_transformer.CoreAttention.__init__', MockCoreAttention.__init__)
    @pytest.mark.leve1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    # pylint:disable=W0212
    def test_merge_heads(self):
        """Feature: _merge_heads interface.
        Description: a testcase for _merge_heads
        Expectation: success
        """
        bsz = 1
        seq_len = 1024
        num_head = 8
        hidden_size = 4096
        hidden_dim = num_head * hidden_size
        input_x = np.random.randn(bsz, num_head, seq_len, hidden_size).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        config = None
        layer_number = 1
        coreattn = CoreAttention(config, layer_number)
        out = coreattn._merge_heads(input_x_tensor)
        self.assertEqual(out.shape, (bsz, seq_len, hidden_dim))


class MockChatGLM2SelfAttention(nn.Cell):
    """Mock ChatGLM2 self-attention."""
    def __init__(self):
        bsz = 1
        seq_len = 1024
        num_heads = 8
        hidden_size_per_head = 4096
        self.is_first_iteration = True
        self.head_dim = 128
        self.projection_size = 128 * num_heads
        # Per attention head and per partition values.
        self.apply_query_key_layer_scaling = True
        self.norm_factor = 1.0
        self.n_head = num_heads
        self.params_dtype = 'float16'
        self.compute_dtype = 'float16'
        self.batch_size = bsz
        self.pre_seq_len = seq_len
        self.n_rep = 1
        self.multi_query_group_num = 2

        self.multi_query_attention = True
        self.qkv_hidden_size = 3 * self.projection_size
        self.kv_hidden_size = self.projection_size
        self.use_rearrange_rope = True
        self.mask_generate = 'inmap'  # "inmap", "compress_reset"

        if self.multi_query_attention:
            self.n_kv_head = 2
            self.qkv_hidden_size = self.projection_size + 2 * self.head_dim * self.multi_query_group_num
            self.kv_hidden_size = self.n_kv_head * self.head_dim
        self.qkv_concat = False
        input_x = np.random.randn(bsz,
                                  num_heads,
                                  seq_len,
                                  hidden_size_per_head).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        self.tile_kv = MagicMock(return_value=input_x_tensor)
        self.reshape = MagicMock(return_value=input_x_tensor)
        self.shape = MagicMock(return_value=[1, 1, 1, 1])
        self.merger_head_transpose = MagicMock(return_value=input_x_tensor)
        self.sub = MagicMock(return_value=input_x_tensor)
        self.add = MagicMock(return_value=input_x_tensor)
        self.stack = MagicMock(return_value=input_x_tensor)
        self.cast = MagicMock(return_value=input_x_tensor)
        self.concat = MagicMock(return_value=input_x_tensor)
        input_x = np.random.randn(bsz, num_heads, seq_len, seq_len).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        self.mul = MagicMock(return_value=input_x_tensor)


class TestChatGLM2SelfAttention(unittest.TestCase):
    """A test for mock GLM2SelfAttention """
    @patch('mindformers.models.glm2.glm2_transformer.ChatGLM2SelfAttention.__init__',
           MockChatGLM2SelfAttention.__init__)
    @pytest.mark.leve1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    # pylint:disable=W0212
    def test_repeat_kv(self):
        """Feature: _repeat_kv interface.
        Description: a testcase for _repeat_kv
        Expectation: success
        """
        bsz = 1
        seq_len = 1024
        num_heads = 8
        hidden_size_per_head = 4096
        rep = 2
        input_x = np.random.randn(bsz,
                                  num_heads,
                                  seq_len,
                                  hidden_size_per_head).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        self.tile_kv = MagicMock(return_value=input_x_tensor)
        self.reshape = MagicMock(return_value=(1, 1, 1, 1))
        config = None
        layer_number = 1
        selfattn = ChatGLM2SelfAttention(config, layer_number)
        out = selfattn._repeat_kv(input_x_tensor, rep)
        self.assertEqual(out.shape, (bsz, num_heads, seq_len, hidden_size_per_head))


    @patch('mindformers.models.glm2.glm2_transformer.ChatGLM2SelfAttention.__init__',
           MockChatGLM2SelfAttention.__init__)
    @pytest.mark.leve1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    # pylint:disable=W0212
    def test_merge_heads(self):
        """Feature: _merge_heads interface.
        Description: a testcase for _merge_heads
        Expectation: success
        """
        bsz = 1
        seq_len = 1024
        num_heads = 8
        hidden_size_per_head = 4096
        input_x = np.random.randn(bsz,
                                  seq_len,
                                  num_heads,
                                  hidden_size_per_head).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        self.reshape = MagicMock(return_value=input_x_tensor)
        config = None
        layer_number = 1
        selfattn = ChatGLM2SelfAttention(config, layer_number)
        out = selfattn._merge_heads(input_x_tensor)
        self.assertEqual(out.shape, (bsz, num_heads, seq_len, hidden_size_per_head))

    @patch('mindformers.models.glm2.glm2_transformer.ChatGLM2SelfAttention.__init__',
           MockChatGLM2SelfAttention.__init__)
    @pytest.mark.leve1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_apply_rotary_pos_emb(self):
        """Feature: apply_rotary_pos_emb interface.
        Description: a testcase for apply_rotary_pos_emb
        Expectation: success
        """
        bsz = 1
        seq_len = 1024
        num_heads = 8
        hidden_size_per_head = 4096
        input_x = np.random.randn(bsz,
                                  seq_len,
                                  num_heads,
                                  hidden_size_per_head).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        self.reshape = MagicMock(return_value=input_x_tensor)
        ops.split = MagicMock(return_value=(1, 1))
        config = None
        layer_number = 1
        selfattn = ChatGLM2SelfAttention(config, layer_number)
        out = selfattn.apply_rotary_pos_emb(input_x_tensor,
                                            [input_x_tensor, input_x_tensor, input_x_tensor])
        self.assertEqual(out.shape, (bsz, num_heads, seq_len, hidden_size_per_head))
