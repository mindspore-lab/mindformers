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
"""test llama transformer."""
import unittest
from unittest.mock import patch, MagicMock
import math
from typing import Optional
import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, nn
import mindspore.common.dtype as mstype
from mindformers.models.llama.llama_transformer import LLamaDecodeLayer
from mindformers.models.llama.llama_transformer import LLamaAttention
from mindformers.modules.transformer import TransformerOpParallelConfig
ms.set_context(mode=ms.PYNATIVE_MODE)

class MockLLamaAttention(nn.Cell):
    """A mock class for mocking LLamaAttention."""

    def __init__(self,
                 dim: int = 512,
                 n_heads: int = 8,
                 n_kv_heads: Optional[int] = None,
                 qkv_concat=False,
                 compute_dtype=mstype.float16,
                 use_past=False,
                 use_flash_attention=False,
                 parallel_config=TransformerOpParallelConfig(),
                 ):
        self.use_past = use_past
        self.qkv_concat = qkv_concat
        self.use_flash_attention = use_flash_attention
        self.dtype = compute_dtype
        self.hidden_size = dim
        self.n_head = n_heads
        self.head_dim = dim // n_heads
        self.n_kv_head = n_heads if n_kv_heads is None else n_kv_heads
        self.n_rep = self.n_head // self.n_kv_head
        self.kv_dim = self.n_kv_head * self.head_dim
        cp = parallel_config.context_parallel
        self.context_parallel = cp
        self.cp_ds = 2

        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        attention_np_matrix = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        attention_rt_tensor = Tensor(attention_np_matrix, dtype=ms.bfloat16)
        self.shape = MagicMock(return_value=(bs, seq_len, hidden_dim))
        self.reshape = MagicMock(return_value=attention_rt_tensor)
        self.wo = MagicMock(return_value=attention_rt_tensor)
        self.wq = MagicMock(return_value=attention_rt_tensor)
        self.wk = MagicMock(return_value=attention_rt_tensor)
        self.wv = MagicMock(return_value=attention_rt_tensor)
        self.w_qkv = MagicMock(return_value=attention_rt_tensor)
        self.cast = MagicMock(return_value=attention_rt_tensor)
        self.transpose = MagicMock(return_value=attention_rt_tensor)
        self.apply_rotary_emb = MagicMock(return_value=(attention_rt_tensor, attention_rt_tensor))
        self.split_qkv = MagicMock(return_value=(attention_rt_tensor, attention_rt_tensor, attention_rt_tensor))
        self.infer_attention = MagicMock(return_value=attention_rt_tensor)
        self._repeat_kv = MagicMock(return_value=attention_rt_tensor)
        self._attn = MagicMock(return_value=attention_rt_tensor)
        self.flash_attention = MagicMock(return_value=attention_rt_tensor)
        self._merge_heads = MagicMock(return_value=attention_rt_tensor)
        self.transpose_back = MagicMock(return_value=attention_rt_tensor)
        self._ulysses_q_a2a = MagicMock(return_value=attention_rt_tensor)
        self._ulysses_kv_a2a = MagicMock(return_value=attention_rt_tensor)
        self._ulysses_context_layer_a2a = MagicMock(return_value=attention_rt_tensor)
        self.tile_kv = MagicMock(return_value=attention_rt_tensor)

class TestLLamaAttention(unittest.TestCase):
    """A test class for testing LLamaAttention."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_transformer.LLamaAttention.__init__', MockLLamaAttention.__init__)
    def test_construct_ut_qf(self):
        """
        Feature: LLamaAttention
        Description: Test LLamaAttention construct
        Expectation: No Exception
        """
        llama_attention = LLamaAttention(use_past=True, qkv_concat=False)
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)

        freqs_cis = np.random.randn(seq_len, 128).astype(np.float32)
        freqs_cis_tensor = Tensor(freqs_cis, dtype=ms.float32)
        out = llama_attention.construct(input_x_tensor, freqs_cis_tensor)
        self.assertEqual(out.shape, (bs, seq_len, hidden_dim))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_transformer.LLamaAttention.__init__', MockLLamaAttention.__init__)
    def test_construct_uf_qt(self):
        """
        Feature: LLamaAttention
        Description: Test LLamaAttention construct
        Expectation: No Exception
        """
        llama_attention = LLamaAttention(use_past=False, qkv_concat=True)
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)

        freqs_cis = np.random.randn(seq_len, 128).astype(np.float32)
        freqs_cis_tensor = Tensor(freqs_cis, dtype=ms.float32)
        out = llama_attention.construct(input_x_tensor, freqs_cis_tensor)
        self.assertEqual(out.shape, (bs, seq_len, hidden_dim))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_transformer.LLamaAttention.__init__', MockLLamaAttention.__init__)
    def test_construct_ufa(self):
        """
        Feature: LLamaAttention
        Description: Test LLamaAttention construct
        Expectation: No Exception
        """
        llama_attention = LLamaAttention(use_past=False, qkv_concat=True, use_flash_attention=True)
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)

        freqs_cis = np.random.randn(seq_len, 128).astype(np.float32)
        freqs_cis_tensor = Tensor(freqs_cis, dtype=ms.float32)
        out = llama_attention.construct(input_x_tensor, freqs_cis_tensor)
        self.assertEqual(out.shape, (bs, seq_len, hidden_dim))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_transformer.LLamaAttention.__init__', MockLLamaAttention.__init__)
    def test_construct_cp(self):
        """
        Feature: LLamaAttention
        Description: Test LLamaAttention construct
        Expectation: No Exception
        """
        parallel_conf = TransformerOpParallelConfig(context_parallel=2)
        llama_attention = LLamaAttention(use_past=False, qkv_concat=True, use_flash_attention=True,
                                         parallel_config=parallel_conf)
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)

        freqs_cis = np.random.randn(seq_len, 128).astype(np.float32)
        freqs_cis_tensor = Tensor(freqs_cis, dtype=ms.float32)
        out = llama_attention.construct(input_x_tensor, freqs_cis_tensor)
        self.assertEqual(out.shape, (bs, seq_len, hidden_dim))


class MockLLamaAttention2(nn.Cell):
    """A mock class for mocking LLamaAttention."""

    def __init__(self,
                 dim: int = 512,
                 n_heads: int = 8,
                 n_kv_heads: Optional[int] = None,
                 qkv_concat=False,
                 compute_dtype=mstype.float16,
                 softmax_compute_dtype=mstype.float32,
                 use_past=False,
                 use_flash_attention=False,
                 parallel_config=TransformerOpParallelConfig()
                 ):
        self.use_past = use_past
        self.qkv_concat = qkv_concat
        self.use_flash_attention = use_flash_attention
        self.softmax_dtype = softmax_compute_dtype
        self.dtype = compute_dtype
        self.hidden_size = dim
        self.n_head = n_heads
        self.head_dim = dim // n_heads
        self.n_kv_head = n_heads if n_kv_heads is None else n_kv_heads
        self.n_rep = self.n_head // self.n_kv_head
        self.kv_dim = self.n_kv_head * self.head_dim
        cp = parallel_config.context_parallel
        self.context_parallel = cp
        self.cp_ds = parallel_config.get_ulysses_cp_num()
        self.inv_norm_factor = Tensor(1.0 / math.sqrt(self.head_dim), dtype=compute_dtype)

        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        attention_np_matrix = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        attention_rt_tensor = Tensor(attention_np_matrix, dtype=ms.bfloat16)
        self.shape = MagicMock(return_value=(bs, seq_len, hidden_dim))
        self.reshape = MagicMock(return_value=attention_rt_tensor)
        self.batch_matmul_q_k = MagicMock(return_value=attention_rt_tensor)
        self.batch_matmul = MagicMock(return_value=attention_rt_tensor)
        self.cast_attn = MagicMock(return_value=attention_rt_tensor)
        self.mul = MagicMock(return_value=attention_rt_tensor)
        self.add = MagicMock(return_value=attention_rt_tensor)
        self.softmax = MagicMock(return_value=attention_rt_tensor)
        self.wo = MagicMock(return_value=attention_rt_tensor)
        self.wq = MagicMock(return_value=attention_rt_tensor)
        self.wk = MagicMock(return_value=attention_rt_tensor)
        self.wv = MagicMock(return_value=attention_rt_tensor)
        self.w_qkv = MagicMock(return_value=attention_rt_tensor)
        self.cast = MagicMock(return_value=attention_rt_tensor)
        self.transpose = MagicMock(return_value=attention_rt_tensor)
        self.apply_rotary_emb = MagicMock(return_value=(attention_rt_tensor, attention_rt_tensor))
        self.split_qkv = MagicMock(return_value=(attention_rt_tensor, attention_rt_tensor, attention_rt_tensor))
        self.infer_attention = MagicMock(return_value=attention_rt_tensor)
        self._repeat_kv = MagicMock(return_value=attention_rt_tensor)
        self.flash_attention = MagicMock(return_value=attention_rt_tensor)
        self._merge_heads = MagicMock(return_value=attention_rt_tensor)
        self.transpose_back = MagicMock(return_value=attention_rt_tensor)
        self._ulysses_q_a2a = MagicMock(return_value=attention_rt_tensor)
        self._ulysses_kv_a2a = MagicMock(return_value=attention_rt_tensor)
        self._ulysses_context_layer_a2a = MagicMock(return_value=attention_rt_tensor)


class TestLlamaAttention2(unittest.TestCase):
    """A test class for testing LLamaAttention."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_transformer.LLamaAttention.__init__', MockLLamaAttention2.__init__)
    def test_construct_cp(self):
        """
        Feature: LLamaAttention
        Description: Test LLamaAttention construct
        Expectation: No Exception
        """
        parallel_conf = TransformerOpParallelConfig(context_parallel=2)
        llama_attention = LLamaAttention(use_past=False, qkv_concat=True, use_flash_attention=True,
                                         parallel_config=parallel_conf)
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)

        freqs_cis = np.random.randn(seq_len, 128).astype(np.float32)
        freqs_cis_tensor = Tensor(freqs_cis, dtype=ms.float32)
        out = llama_attention.construct(input_x_tensor, freqs_cis_tensor)
        self.assertEqual(out.shape, (bs, seq_len, hidden_dim))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_transformer.LLamaAttention.__init__', MockLLamaAttention2.__init__)
    def test_attn(self):
        """
        Feature: LLamaAttention
        Description: Test LLamaAttention _attn
        Expectation: No Exception
        """
        llama_attention = LLamaAttention()
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        # pylint: disable=W0212
        out = llama_attention._attn(input_x_tensor, input_x_tensor, input_x_tensor, input_x_tensor)
        self.assertEqual(out.shape, (bs, seq_len, hidden_dim))


class MockLLamaAttention3(nn.Cell):
    """A mock class for mocking LLamaAttention."""

    def __init__(self,
                 dim: int = 512,
                 n_heads: int = 8):
        self.n_head = n_heads
        self.hidden_size = dim
        self.head_dim = dim // n_heads
        self.model_parallel = 1
        self.cp_ds = 1
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        n_kv_heads = 8
        attention_np_matrix = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        attention_rt_tensor = Tensor(attention_np_matrix, dtype=ms.bfloat16)
        self.shape = MagicMock(return_value=(bs, n_kv_heads, seq_len, hidden_dim))
        self.reshape = MagicMock(return_value=attention_rt_tensor)
        self.tile_kv = MagicMock(return_value=attention_rt_tensor)
        self.merger_head_transpose = MagicMock(return_value=attention_rt_tensor)
        self.transpose_ulysses = MagicMock(return_value=attention_rt_tensor)
        self.transpose_a2a = MagicMock(return_value=attention_rt_tensor)
        self.transpose_ulysses_merger_a2a = MagicMock(return_value=attention_rt_tensor)
        self.transpose_ulysses_merger = MagicMock(return_value=attention_rt_tensor)


class TestLLamaAttention3(unittest.TestCase):
    """A test class for testing LLamaAttention."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_transformer.LLamaAttention.__init__', MockLLamaAttention3.__init__)
    def test_repeat_kv(self):
        """
        Feature: LLamaAttention
        Description: Test LLamaAttention _repeat_kv
        Expectation: No Exception
        """
        llama_attention = LLamaAttention()
        rep = 2
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        # pylint: disable=W0212
        out = llama_attention._repeat_kv(input_x_tensor, rep)
        self.assertEqual(out.shape, (bs, seq_len, hidden_dim))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_transformer.LLamaAttention.__init__', MockLLamaAttention3.__init__)
    def test_repeat_kv1(self):
        """
        Feature: LLamaAttention
        Description: Test LLamaAttention _repeat_kv
        Expectation: No Exception
        """
        llama_attention = LLamaAttention()
        rep = 1
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        # pylint: disable=W0212
        out = llama_attention._repeat_kv(input_x_tensor, rep)
        self.assertEqual(out.shape, (bs, seq_len, hidden_dim))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_transformer.LLamaAttention.__init__', MockLLamaAttention3.__init__)
    def test_merge_heads(self):
        """
        Feature: LLamaAttention
        Description: Test LLamaAttention _merge_heads
        Expectation: No Exception
        """
        llama_attention = LLamaAttention()
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        # pylint: disable=W0212
        out = llama_attention._merge_heads(input_x_tensor)
        self.assertEqual(out.shape, (bs, seq_len, hidden_dim))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_transformer.LLamaAttention.__init__', MockLLamaAttention3.__init__)
    def test_ulysses_q_a2a(self):
        """
        Feature: LLamaAttention
        Description: Test LLamaAttention _ulysses_q_a2a
        Expectation: No Exception
        """
        llama_attention = LLamaAttention()
        bs = 1
        seq_len = 1024
        n_head = 8
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, n_head, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        # pylint: disable=W0212
        out = llama_attention._ulysses_q_a2a(input_x_tensor)
        self.assertEqual(out.shape, (bs, seq_len, hidden_dim))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_transformer.LLamaAttention.__init__', MockLLamaAttention3.__init__)
    def test_ulysses_kv_a2a(self):
        """
        Feature: LLamaAttention
        Description: Test LLamaAttention _ulysses_kv_a2a
        Expectation: No Exception
        """
        llama_attention = LLamaAttention()
        bs = 1
        seq_len = 1024
        n_head = 8
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, n_head, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        # pylint: disable=W0212
        out = llama_attention._ulysses_kv_a2a(input_x_tensor)
        self.assertEqual(out.shape, (bs, seq_len, hidden_dim))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_transformer.LLamaAttention.__init__', MockLLamaAttention3.__init__)
    def test_ulysses_context_layer_a2a(self):
        """
        Feature: LLamaAttention
        Description: Test LLamaAttention _ulysses_context_layer_a2a
        Expectation: No Exception
        """
        llama_attention = LLamaAttention(dim=4096)
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        # pylint: disable=W0212
        out = llama_attention._ulysses_context_layer_a2a(input_x_tensor)
        self.assertEqual(out.shape, (bs, seq_len, hidden_dim))


class MockLLamaAttention4(nn.Cell):
    """A mock class for mocking LLamaAttention."""

    def __init__(self,
                 dim: int = 512,
                 n_heads: int = 8,
                 compute_dtype=mstype.float16
                 ):
        self.n_head = n_heads
        self.hidden_size = dim
        self.head_dim = dim // n_heads
        self.dtype = compute_dtype
        self.model_parallel = 1
        self.cp_ds = 1
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        n_kv_heads = 8
        attention_np_matrix = np.random.randn(bs, seq_len, n_kv_heads, hidden_dim).astype(np.float16)
        attention_rt_tensor = Tensor(attention_np_matrix, dtype=ms.bfloat16)
        self.reshape = MagicMock(return_value=attention_rt_tensor)
        self.transpose = MagicMock(return_value=attention_rt_tensor)
        self.cast = MagicMock(return_value=attention_rt_tensor)


class TestLLamaAttention4(unittest.TestCase):
    """A test class for testing LLamaAttention."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_transformer.LLamaAttention.__init__', MockLLamaAttention4.__init__)
    def test_cat_prefix(self):
        """
        Feature: LLamaAttention
        Description: Test LLamaAttention _ulysses_context_layer_a2a
        Expectation: No Exception
        """
        llama_attention = LLamaAttention()
        bs = 1
        seq_len = 1024
        n_head = 8
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, n_head, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        # pylint: disable=W0212
        key, value = llama_attention._cat_prefix(input_x_tensor, input_x_tensor, (input_x_tensor, input_x_tensor))
        self.assertEqual(key.shape, (bs, seq_len, n_head*2, hidden_dim))
        self.assertEqual(value.shape, (bs, seq_len, n_head*2, hidden_dim))


class MockLLamaDecodeLayer(nn.Cell):
    """A mock class for mocking LLamaDecodeLayer."""

    def __init__(self,
                 layer_id,
                 use_past=False
                 ):
        self.use_past = use_past
        self.layer_id = layer_id
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        attention_norm_np_matrix = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        attention_norm_rt_tensor = Tensor(attention_norm_np_matrix, dtype=ms.bfloat16)
        self.attention_norm = MagicMock(return_value=attention_norm_rt_tensor)

        attention_np_matrix = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        attention_rt_tensor = Tensor(attention_np_matrix, dtype=ms.bfloat16)
        self.attention = MagicMock(return_value=attention_rt_tensor)

        self.add = MagicMock(return_value=attention_rt_tensor)

        self.ffn_norm = MagicMock(return_value=attention_rt_tensor)

        self.feed_forward = MagicMock(return_value=attention_rt_tensor)

        self._check_input = MagicMock(return_value=True)


class TestLLamaDecodeLayer(unittest.TestCase):
    """A test class for testing LLamaDecodeLayer."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_transformer.LLamaDecodeLayer.__init__', MockLLamaDecodeLayer.__init__)
    def test_construct_use_past_true(self):
        """
        Feature: LLamaDecodeLayer
        Description: Test LLamaDecodeLayer construct
        Expectation: No Exception
        """
        llama_decode_layer = LLamaDecodeLayer(layer_id=1, use_past=True)
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)

        freqs_cis = np.random.randn(seq_len, 128).astype(np.float32)
        freqs_cis_tensor = Tensor(freqs_cis, dtype=ms.float32)
        out = llama_decode_layer.construct(input_x_tensor, freqs_cis_tensor)
        self.assertEqual(out.shape, (bs, seq_len, hidden_dim))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_transformer.LLamaDecodeLayer.__init__', MockLLamaDecodeLayer.__init__)
    def test_construct_use_past_false(self):
        """
        Feature: LLamaDecodeLayer
        Description: Test LLamaDecodeLayer construct
        Expectation: No Exception
        """
        llama_decode_layer = LLamaDecodeLayer(layer_id=1)
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)

        freqs_cis = np.random.randn(seq_len, 128).astype(np.float32)
        freqs_cis_tensor = Tensor(freqs_cis, dtype=ms.float32)
        out = llama_decode_layer.construct(input_x_tensor, freqs_cis_tensor)
        self.assertEqual(out.shape, (bs, seq_len, hidden_dim))


class MockLLamaDecodeLayer2(nn.Cell):
    """A mock class for mocking LLamaDecodeLayer."""

    def __init__(self,
                 layer_id,
                 ):
        self.layer_id = layer_id


class TestLLamaDecodeLayer2(unittest.TestCase):
    """A test class for testing LLamaDecodeLayer."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_transformer.LLamaDecodeLayer.__init__', MockLLamaDecodeLayer2.__init__)
    def test_check_input(self):
        """
        Feature: LLamaDecodeLayer
        Description: Test LLamaDecodeLayer _check_input
        Expectation: No Exception
        """
        llama_decode_layer = LLamaDecodeLayer(layer_id=1)
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        # pylint: disable=W0212
        out = llama_decode_layer._check_input(input_x_tensor, (input_x_tensor, input_x_tensor, input_x_tensor),
                                              input_x_tensor)
        self.assertEqual(out, True)
