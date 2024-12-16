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
"""test llama layer."""
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pytest
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
import mindspore as ms
from mindspore import Tensor
from mindspore.nn.cell import Cell
import mindspore.common.dtype as mstype
from mindformers.models.llama.llama_layer import LlamaSiLU, LlamaEmbedding, LlamaRMSNorm, LlamaFeedForward
from mindformers.models.llama.llama_layer import LlamaMoeInferFeedForward, LlamaFeedForwardWithMoE

ms.set_context(mode=ms.PYNATIVE_MODE)

class MockLlamaSiLU(Cell):
    """A mock class for mocking LlamaSiLU."""

    def __init__(self):
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        attention_np_matrix = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        attention_rt_tensor = Tensor(attention_np_matrix, dtype=ms.bfloat16)
        self.silu = MagicMock(return_value=attention_rt_tensor)
        self.mul = MagicMock(return_value=attention_rt_tensor)
        self.sigmoid = MagicMock(return_value=attention_rt_tensor)


class TestLlamaSiLU(unittest.TestCase):
    """A test class for testing LlamaSiLU."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_layer.LlamaSiLU.__init__', MockLlamaSiLU.__init__)
    def test_construct(self):
        """
        Feature: LlamaSiLU
        Description: Test LlamaSiLU construct
        Expectation: No Exception
        """
        llama_silu = LlamaSiLU()
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)

        out = llama_silu.construct(input_x_tensor)
        self.assertEqual(out.shape, (bs, seq_len, hidden_dim))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_layer.LlamaSiLU.__init__', MockLlamaSiLU.__init__)
    def test_self_silu(self):
        """
        Feature: LlamaSiLU
        Description: Test LlamaSiLU _self_silu
        Expectation: No Exception
        """
        llama_silu = LlamaSiLU()
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        # pylint: disable=W0212
        out = llama_silu._self_silu(input_x_tensor)
        self.assertEqual(out.shape, (bs, seq_len, hidden_dim))


class MockLlamaEmbedding(Cell):
    """A mock class for mocking LlamaEmbedding."""

    def __init__(self, vocab_table_size, embedding_size):
        self.vocab_table_size = vocab_table_size
        self.embedding_size = embedding_size
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        attention_np_matrix = np.random.randn(bs, seq_len, hidden_dim).astype(np.int32)
        attention_rt_tensor = Tensor(attention_np_matrix, dtype=ms.int32)
        self.embedding_weight = MagicMock(return_value=attention_rt_tensor)
        self.gather = MagicMock(return_value=attention_rt_tensor)


class TestLlamaEmbedding(unittest.TestCase):
    """A test class for testing LlamaEmbedding."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_layer.LlamaEmbedding.__init__', MockLlamaEmbedding.__init__)
    def test_construct(self):
        """
        Feature: LlamaEmbedding
        Description: Test LlamaEmbedding construct
        Expectation: No Exception
        """
        llama_embedding = LlamaEmbedding(vocab_table_size=4096, embedding_size=4096)
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.int32)
        input_x_tensor = Tensor(input_x, dtype=ms.int32)

        out = llama_embedding.construct(input_x_tensor)
        self.assertEqual(out.shape, (bs, seq_len, hidden_dim))


class MockLlamaRMSNorm(Cell):
    """A mock class for mocking LlamaRMSNorm."""

    def __init__(self, dim, eps=1e-6, compute_type=mstype.float32):
        # pylint: disable=E1003
        super(LlamaRMSNorm, self).__init__()
        self.eps = eps
        self.compute_type = compute_type
        self.weight = Parameter(initializer('ones', (dim,), dtype=self.compute_type), parallel_optimizer=False)
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        attention_np_matrix = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        attention_rt_tensor = Tensor(attention_np_matrix, dtype=ms.bfloat16)
        self.rms_norm = MagicMock(return_value=attention_rt_tensor)
        self.norm = MagicMock(return_value=attention_rt_tensor)
        self.cast = MagicMock(return_value=attention_rt_tensor)
        self.square = MagicMock(return_value=attention_rt_tensor)
        self.mean = MagicMock(return_value=attention_rt_tensor)
        self.add = MagicMock(return_value=attention_rt_tensor)
        self.rsqrt = MagicMock(return_value=attention_rt_tensor)
        self.mul = MagicMock(return_value=attention_rt_tensor)
        self.mul2 = MagicMock(return_value=attention_rt_tensor)
        self.rcast = MagicMock(return_value=attention_rt_tensor)

class TestLlamaRMSNorm(unittest.TestCase):
    """A test class for testing LlamaRMSNorm."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_layer.LlamaRMSNorm.__init__', MockLlamaRMSNorm.__init__)
    def test_construct(self):
        """
        Feature: LlamaRMSNorm
        Description: Test LlamaRMSNorm construct
        Expectation: No Exception
        """
        llama_rmsnorm = LlamaRMSNorm(dim=512)
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)

        out = llama_rmsnorm.construct(input_x_tensor)
        self.assertEqual(out.shape, (bs, seq_len, hidden_dim))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_layer.LlamaRMSNorm.__init__', MockLlamaRMSNorm.__init__)
    def test_self_norm(self):
        """
        Feature: LlamaRMSNorm
        Description: Test LlamaRMSNorm _self_norm
        Expectation: No Exception
        """
        llama_rmsnorm = LlamaRMSNorm(dim=512)
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        # pylint: disable=W0212
        out = llama_rmsnorm._self_norm(input_x_tensor)
        self.assertEqual(out.shape, (bs, seq_len, hidden_dim))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_layer.LlamaRMSNorm.__init__', MockLlamaRMSNorm.__init__)
    def test_rms_norm(self):
        """
        Feature: LlamaRMSNorm
        Description: Test LlamaRMSNorm _rms_norm
        Expectation: No Exception
        """
        llama_rmsnorm = LlamaRMSNorm(dim=512)
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        # pylint: disable=W0212
        out = llama_rmsnorm._rms_norm(input_x_tensor)
        self.assertEqual(out.shape, (bs, seq_len, hidden_dim))


class MockLlamaFeedForward(Cell):
    """A mock class for mocking LlamaFeedForward."""

    def __init__(self, dim,
                 hidden_dim=None,
                 compute_dtype=mstype.float16,
                 ffn_concat=False):
        # pylint: disable=E1003
        super(LlamaFeedForward, self).__init__()
        self.dim = dim
        self.dtype = compute_dtype
        self.ffn_concat = ffn_concat
        self.hidden_dim = hidden_dim
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        attention_np_matrix = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        attention_rt_tensor = Tensor(attention_np_matrix, dtype=ms.bfloat16)
        self.w1 = MagicMock(return_value=attention_rt_tensor)
        self.w2 = MagicMock(return_value=attention_rt_tensor)
        self.w3 = MagicMock(return_value=attention_rt_tensor)
        self.mul = MagicMock(return_value=attention_rt_tensor)
        self.w_gate_hidden = MagicMock(return_value=attention_rt_tensor)
        self.activate = MagicMock(return_value=attention_rt_tensor)
        self.split = MagicMock(return_value=(attention_rt_tensor, attention_rt_tensor))


class TestLlamaFeedForward(unittest.TestCase):
    """A test class for testing LlamaFeedForward."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_layer.LlamaFeedForward.__init__', MockLlamaFeedForward.__init__)
    def test_construct_ffn_false(self):
        """
        Feature: LlamaFeedForward
        Description: Test LlamaFeedForward construct
        Expectation: No Exception
        """
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        llama_feed_forward = LlamaFeedForward(dim=hidden_dim)
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)

        out = llama_feed_forward.construct(input_x_tensor)
        self.assertEqual(out.shape, (bs, seq_len, hidden_dim))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_layer.LlamaFeedForward.__init__', MockLlamaFeedForward.__init__)
    def test_construct_ffn_true(self):
        """
        Feature: LlamaFeedForward
        Description: Test LlamaFeedForward construct
        Expectation: No Exception
        """
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        llama_feed_forward = LlamaFeedForward(dim=hidden_dim, ffn_concat=True)
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)

        out = llama_feed_forward.construct(input_x_tensor)
        self.assertEqual(out.shape, (bs, seq_len, hidden_dim))


class MockLlamaMoeInferFeedForward(Cell):
    """A mock class for mocking LlamaMoeInferFeedForward."""

    def __init__(self, dim,
                 hidden_dim=None,
                 compute_dtype=mstype.float16):
        self.dim = dim
        self.dtype = compute_dtype
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        attention_np_matrix = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        attention_rt_tensor = Tensor(attention_np_matrix, dtype=ms.bfloat16)
        self.cast = MagicMock(return_value=attention_rt_tensor)
        self.w1 = MagicMock(return_value=attention_rt_tensor)
        self.w2 = MagicMock(return_value=attention_rt_tensor)
        self.w3 = MagicMock(return_value=attention_rt_tensor)
        self.mul = MagicMock(return_value=attention_rt_tensor)


class TestLlamaMoeInferFeedForward(unittest.TestCase):
    """A test class for testing LlamaMoeInferFeedForward."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_init(self):
        llamamoeinferfeedforward = LlamaMoeInferFeedForward(dim=512, hidden_dim=512)
        self.assertEqual(llamamoeinferfeedforward.dim, 512)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_layer.LlamaMoeInferFeedForward.__init__',
           MockLlamaMoeInferFeedForward.__init__)
    def test_construct(self):
        """
        Feature: LlamaMoeInferFeedForward
        Description: Test LlamaMoeInferFeedForward construct
        Expectation: No Exception
        """
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        llama_moe_infer_feed_forward = LlamaMoeInferFeedForward(dim=hidden_dim)
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)

        out = llama_moe_infer_feed_forward.construct(input_x_tensor)
        self.assertEqual(out.shape, (bs, seq_len, hidden_dim))


class MockLlamaFeedForwardWithMoE(Cell):
    """A mock class for mocking LlamaFeedForwardWithMoE."""

    def __init__(self, hidden_size,
                 compute_dtype=mstype.float16,
                 return_extra_loss=False
                 ):
        self.hidden_size = hidden_size
        self.use_shared_expert_gating = True
        self.return_extra_loss = return_extra_loss
        self.router_dense_type = mstype.float32
        self.compute_dtype = compute_dtype
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        attention_np_matrix = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        attention_rt_tensor = Tensor(attention_np_matrix, dtype=ms.bfloat16)
        self.shared_experts = MagicMock(return_value=attention_rt_tensor)
        self.sigmoid = MagicMock(return_value=attention_rt_tensor)
        self.shared_experts_gate = MagicMock(return_value=attention_rt_tensor)
        self.cast = MagicMock(return_value=attention_rt_tensor)
        self.mul = MagicMock(return_value=attention_rt_tensor)
        self.cast = MagicMock(return_value=attention_rt_tensor)
        self.add = MagicMock(return_value=attention_rt_tensor)
        self.routed_experts = MagicMock(return_value=attention_rt_tensor)


class TestLlamaFeedForwardWithMoE(unittest.TestCase):
    """A test class for testing LlamaFeedForwardWithMoE."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_layer.LlamaFeedForwardWithMoE.__init__',
           MockLlamaFeedForwardWithMoE.__init__)
    def test_construct(self):
        """
        Feature: LlamaFeedForwardWithMoE
        Description: Test LlamaFeedForwardWithMoE construct
        Expectation: No Exception
        """
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        llama_feed_forward_with_moe = LlamaFeedForwardWithMoE(hidden_size=hidden_dim)
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)

        out = llama_feed_forward_with_moe.construct(input_x_tensor)
        self.assertEqual(out.shape, (bs, seq_len, hidden_dim))
