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
"""test llama interleave."""
import unittest
from unittest.mock import patch, MagicMock
import math
from typing import Optional
import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, nn
import mindspore.common.dtype as mstype
from mindformers.models.llama.llama_interleave import LLamaDecodeLayerInterleave, LLamaAttentionInterleave
from mindformers.models.llama.llama_interleave import _MicroBatch
from mindformers.modules.transformer import TransformerOpParallelConfig

ms.set_context(mode=ms.PYNATIVE_MODE)


class TestMicroBatch(unittest.TestCase):
    """A test class for testing _MicroBatch."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_init(self):
        microbatch = _MicroBatch(micro_size=8, input_size=8, axis_list=None)
        self.assertEqual(microbatch.micro_size, 8)

class MockLlamaAttentionInterleave(nn.Cell):
    """A mock class for mocking LlamaAttentionInterleave."""

    def __init__(self,
                 batch_size,
                 seq_length,
                 dim: int = 512,
                 n_heads: int = 8,
                 n_kv_heads: Optional[int] = None,
                 qkv_concat=False,
                 compute_dtype=mstype.float16,
                 softmax_compute_dtype=mstype.float32):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_size = dim
        self.n_head = n_heads
        self.head_dim = dim // n_heads
        self.n_kv_head = n_heads if n_kv_heads is None else n_kv_heads
        self.n_rep = self.n_head // self.n_kv_head
        self.kv_dim = self.n_kv_head * self.head_dim
        self.inv_norm_factor = Tensor(1.0 / math.sqrt(self.head_dim), dtype=compute_dtype)
        self.dtype = compute_dtype
        self.softmax_dtype = softmax_compute_dtype
        self.qkv_concat = qkv_concat

        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        attention_np_matrix = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        attention_rt_tensor = Tensor(attention_np_matrix, dtype=ms.bfloat16)
        self.reshape = MagicMock(return_value=attention_rt_tensor)
        self.batch_matmul_q_k = MagicMock(return_value=attention_rt_tensor)
        self.mul = MagicMock(return_value=attention_rt_tensor)
        self.add = MagicMock(return_value=attention_rt_tensor)
        self.softmax = MagicMock(return_value=attention_rt_tensor)
        self.cast_attn = MagicMock(return_value=attention_rt_tensor)
        self.batch_matmul = MagicMock(return_value=attention_rt_tensor)
        self._merge_heads = MagicMock(return_value=attention_rt_tensor)
        self.tile_kv = MagicMock(return_value=attention_rt_tensor)
        self.cast = MagicMock(return_value=attention_rt_tensor)
        self.slice_qkv = MagicMock(return_value=attention_rt_tensor)
        self.w = MagicMock(return_value=attention_rt_tensor)
        self.wo = MagicMock(return_value=attention_rt_tensor)
        self.wq = MagicMock(return_value=attention_rt_tensor)
        self.wk = MagicMock(return_value=attention_rt_tensor)
        self.wv = MagicMock(return_value=attention_rt_tensor)
        self.merger_head_transpose = MagicMock(return_value=attention_rt_tensor)


class TestLlamaAttentionInterleave(unittest.TestCase):
    """A test class for testing LlamaAttentionInterleave."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_init(self):
        parallelconfig = TransformerOpParallelConfig(context_parallel=2)
        llama_attention_interleave = LLamaAttentionInterleave(batch_size=1, seq_length=1024)
        llama_attention_interleave2 = LLamaAttentionInterleave(batch_size=1, seq_length=1024, use_flash_attention=True,
                                                               qkv_concat=True, parallel_config=parallelconfig)
        self.assertEqual(llama_attention_interleave.batch_size, 1)
        self.assertEqual(llama_attention_interleave2.batch_size, 1)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_interleave.LLamaAttentionInterleave.__init__',
           MockLlamaAttentionInterleave.__init__)
    def test_compute_qkv_false(self):
        """
        Feature: LlamaAttentionInterleave
        Description: Test LlamaAttentionInterleave compute_qkv
        Expectation: No Exception
        """
        llama_attention_inter_leave = LLamaAttentionInterleave(batch_size=1, seq_length=1024)
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)

        query, key, value = llama_attention_inter_leave.compute_qkv(input_x_tensor)
        self.assertEqual(query.shape, (bs, seq_len, hidden_dim))
        self.assertEqual(key.shape, (bs, seq_len, hidden_dim))
        self.assertEqual(value.shape, (bs, seq_len, hidden_dim))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_interleave.LLamaAttentionInterleave.__init__',
           MockLlamaAttentionInterleave.__init__)
    def test_compute_qkv_true(self):
        """
        Feature: LlamaAttentionInterleave
        Description: Test LlamaAttentionInterleave compute_qkv
        Expectation: No Exception
        """
        llama_attention_inter_leave = LLamaAttentionInterleave(batch_size=1, seq_length=1024, qkv_concat=True)
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)

        query, key, value = llama_attention_inter_leave.compute_qkv(input_x_tensor)
        self.assertEqual(query.shape, (bs, seq_len, hidden_dim))
        self.assertEqual(key.shape, (bs, seq_len, hidden_dim))
        self.assertEqual(value.shape, (bs, seq_len, hidden_dim))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_interleave.LLamaAttentionInterleave.__init__',
           MockLlamaAttentionInterleave.__init__)
    def test_cal_output_proj(self):
        """
        Feature: LlamaAttentionInterleave
        Description: Test LlamaAttentionInterleave cal_output_proj
        Expectation: No Exception
        """
        llama_attention_inter_leave = LLamaAttentionInterleave(batch_size=1, seq_length=1024)
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        # pylint: disable=W0212
        out = llama_attention_inter_leave.cal_output_proj(input_x_tensor)
        self.assertEqual(out.shape, (bs, seq_len, hidden_dim))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_interleave.LLamaAttentionInterleave.__init__',
           MockLlamaAttentionInterleave.__init__)
    def test_attn(self):
        """
        Feature: LlamaAttentionInterleave
        Description: Test LlamaAttentionInterleave attn
        Expectation: No Exception
        """
        llama_attention_inter_leave = LLamaAttentionInterleave(batch_size=1, seq_length=1024)
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        # pylint: disable=W0212
        out = llama_attention_inter_leave._attn(input_x_tensor, input_x_tensor, input_x_tensor, input_x_tensor)
        self.assertEqual(out.shape, (bs, seq_len, hidden_dim))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_interleave.LLamaAttentionInterleave.__init__',
           MockLlamaAttentionInterleave.__init__)
    def test_cal_attn(self):
        """
        Feature: LlamaAttentionInterleave
        Description: Test LlamaAttentionInterleave cal_attn
        Expectation: No Exception
        """
        llama_attention_inter_leave = LLamaAttentionInterleave(batch_size=1, seq_length=1024)
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        # pylint: disable=E1003
        out = llama_attention_inter_leave.cal_output_proj(input_x_tensor)
        self.assertEqual(out.shape, (bs, seq_len, hidden_dim))


class MockLlamaAttentionInterleave2(nn.Cell):
    """A mock class for mocking LlamaAttentionInterleave."""

    def __init__(self,
                 batch_size,
                 seq_length,
                 dim: int = 512,
                 n_heads: int = 8,
                 n_kv_heads: Optional[int] = None,
                 parallel_config=TransformerOpParallelConfig()):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_size = dim
        self.n_head = n_heads
        self.head_dim = dim // n_heads
        self.n_kv_head = n_heads if n_kv_heads is None else n_kv_heads
        self.n_rep = self.n_head // self.n_kv_head
        self.kv_dim = self.n_kv_head * self.head_dim
        cp = parallel_config.context_parallel
        self.context_parallel = cp
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        attention_np_matrix = np.random.randn(bs, n_heads, seq_len, hidden_dim).astype(np.float16)
        attention_rt_tensor = Tensor(attention_np_matrix, dtype=ms.bfloat16)
        self.reshape = MagicMock(return_value=attention_rt_tensor)
        self.transpose = MagicMock(return_value=attention_rt_tensor)
        self.apply_rotary_emb = MagicMock(return_value=(attention_rt_tensor, attention_rt_tensor))
        self.tile_kv = MagicMock(return_value=attention_rt_tensor)
        self.merger_head_transpose = MagicMock(return_value=attention_rt_tensor)


class TestLlamaAttentionInterleave2(unittest.TestCase):
    """A test class for testing LlamaAttentionInterleave."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_interleave.LLamaAttentionInterleave.__init__',
           MockLlamaAttentionInterleave2.__init__)
    def test_repeat_kv(self):
        """
        Feature: LlamaAttentionInterleave
        Description: Test LlamaAttentionInterleave _repeat_kv
        Expectation: No Exception
        """
        llama_attention_inter_leave = LLamaAttentionInterleave(batch_size=1, seq_length=1024)
        llama_attention_inter_leave_rep1 = LLamaAttentionInterleave(batch_size=1, seq_length=1024)
        bs = 1
        n_head = 8
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, n_head, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        # pylint: disable=W0212
        out = llama_attention_inter_leave._repeat_kv(input_x_tensor, rep=2)
        out_rep1 = llama_attention_inter_leave_rep1._repeat_kv(input_x_tensor, rep=1)
        self.assertEqual(out.shape, (bs, n_head, seq_len, hidden_dim))
        self.assertEqual(out_rep1.shape, (bs, n_head, seq_len, hidden_dim))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_interleave.LLamaAttentionInterleave.__init__',
           MockLlamaAttentionInterleave2.__init__)
    def test_merge_heads(self):
        """
        Feature: LlamaAttentionInterleave
        Description: Test LlamaAttentionInterleave _merge_heads
        Expectation: No Exception
        """
        llama_attention_inter_leave = LLamaAttentionInterleave(batch_size=1, seq_length=1024)
        bs = 1
        n_head = 8
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, n_head, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        # pylint: disable=W0212
        out = llama_attention_inter_leave._merge_heads(input_x_tensor)
        self.assertEqual(out.shape, (bs, n_head, seq_len, hidden_dim))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_interleave.LLamaAttentionInterleave.__init__',
           MockLlamaAttentionInterleave2.__init__)
    def test_merge_heads_cp2(self):
        """
        Feature: LlamaAttentionInterleave
        Description: Test LlamaAttentionInterleave _merge_heads
        Expectation: No Exception
        """
        parallel_conf = TransformerOpParallelConfig(context_parallel=2)
        llama_attention_inter_leave = LLamaAttentionInterleave(batch_size=1, seq_length=1024,
                                                               parallel_config=parallel_conf)
        bs = 1
        n_head = 8
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, n_head, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        # pylint: disable=W0212
        out = llama_attention_inter_leave._merge_heads(input_x_tensor)
        self.assertEqual(out.shape, (bs, n_head, seq_len, hidden_dim))


class MockLlamaAttentionInterleave3(nn.Cell):
    """A mock class for mocking LlamaAttentionInterleave."""

    def __init__(self,
                 batch_size,
                 seq_length,
                 dim: int = 512,
                 n_heads: int = 8,
                 n_kv_heads: Optional[int] = None,
                 use_flash_attention=False,
                 parallel_config=TransformerOpParallelConfig()):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_size = dim
        self.n_head = n_heads
        self.head_dim = dim // n_heads
        self.n_kv_head = n_heads if n_kv_heads is None else n_kv_heads
        self.n_rep = self.n_head // self.n_kv_head
        self.kv_dim = self.n_kv_head * self.head_dim
        self.use_flash_attention = use_flash_attention
        cp = parallel_config.context_parallel
        self.context_parallel = cp
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        attention_np_matrix = np.random.randn(bs, n_heads, seq_len, hidden_dim).astype(np.float16)
        attention_rt_tensor = Tensor(attention_np_matrix, dtype=ms.bfloat16)
        self.reshape = MagicMock(return_value=attention_rt_tensor)
        self.transpose = MagicMock(return_value=attention_rt_tensor)
        self.apply_rotary_emb = MagicMock(return_value=(attention_rt_tensor, attention_rt_tensor))
        self._merge_heads = MagicMock(return_value=attention_rt_tensor)
        self._repeat_kv = MagicMock(return_value=attention_rt_tensor)
        self._attn = MagicMock(return_value=attention_rt_tensor)
        self.flash_attention = MagicMock(return_value=attention_rt_tensor)


class TestLlamaAttentionInterleave3(unittest.TestCase):
    """A test class for testing LlamaAttentionInterleave."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_interleave.LLamaAttentionInterleave.__init__',
           MockLlamaAttentionInterleave3.__init__)
    def test_cal_attn_ufa_false(self):
        """
        Feature: LlamaAttentionInterleave
        Description: Test LlamaAttentionInterleave cal_attn
        Expectation: No Exception
        """
        parallel_conf = TransformerOpParallelConfig(context_parallel=1)
        llama_attention_inter_leave = LLamaAttentionInterleave(batch_size=1, seq_length=1024,
                                                               parallel_config=parallel_conf)
        bs = 1
        n_head = 8
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, n_head, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        # pylint: disable=W0212
        out = llama_attention_inter_leave.cal_attn(input_x_tensor, input_x_tensor, input_x_tensor,
                                                   input_x_tensor, input_x_tensor)
        self.assertEqual(out.shape, (bs, n_head, seq_len, hidden_dim))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_interleave.LLamaAttentionInterleave.__init__',
           MockLlamaAttentionInterleave3.__init__)
    def test_cal_attn_ufa_true(self):
        """
        Feature: LlamaAttentionInterleave
        Description: Test LlamaAttentionInterleave cal_attn
        Expectation: No Exception
        """
        parallel_conf = TransformerOpParallelConfig(context_parallel=2)
        llama_attention_inter_leave = LLamaAttentionInterleave(batch_size=1, seq_length=1024, use_flash_attention=True,
                                                               parallel_config=parallel_conf)
        bs = 1
        n_head = 8
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, n_head, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        # pylint: disable=W0212
        out = llama_attention_inter_leave.cal_attn(input_x_tensor, input_x_tensor, input_x_tensor,
                                                   input_x_tensor, input_x_tensor)
        self.assertEqual(out.shape, (bs, n_head, seq_len, hidden_dim))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_interleave.LLamaAttentionInterleave.__init__',
           MockLlamaAttentionInterleave3.__init__)
    def test_cal_attn_ufa_true_cp1(self):
        """
        Feature: LlamaAttentionInterleave
        Description: Test LlamaAttentionInterleave cal_attn
        Expectation: No Exception
        """
        parallel_conf = TransformerOpParallelConfig(context_parallel=1)
        llama_attention_inter_leave = LLamaAttentionInterleave(batch_size=1, seq_length=1024, use_flash_attention=True,
                                                               parallel_config=parallel_conf)
        bs = 1
        n_head = 8
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, n_head, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        # pylint: disable=W0212
        out = llama_attention_inter_leave.cal_attn(input_x_tensor, input_x_tensor, input_x_tensor,
                                                   input_x_tensor, input_x_tensor)
        self.assertEqual(out.shape, (bs, n_head, seq_len, hidden_dim))


class MockLLamaDecodeLayerInterleave(nn.Cell):
    """A mock class for mocking LLamaDecodeLayerInterleave."""

    def __init__(self,
                 batch_size,
                 seq_length,
                 layer_id,
                 fine_grain_interleave=2):
        self.interleave_num = fine_grain_interleave
        self.layer_id = layer_id
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_layers = 1
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        attention_np_matrix = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        attention_rt_tensor = Tensor(attention_np_matrix, dtype=ms.bfloat16)

        self._check_input = MagicMock(return_value=True)
        self.reshape = MagicMock(return_value=attention_rt_tensor)
        self.linear_layer1 = MagicMock(return_value=(attention_rt_tensor, attention_rt_tensor, attention_rt_tensor))
        self.linear_layer2 = MagicMock(return_value=attention_rt_tensor)
        self.attention = MagicMock(return_value=attention_rt_tensor)
        self.attention.cal_attn = MagicMock(return_value=attention_rt_tensor)
        self.interleaved_concat1 = MagicMock(return_value=attention_rt_tensor)
        self.interleaved_concat2 = MagicMock(return_value=attention_rt_tensor)


class TestLLamaDecodeLayerInterleave(unittest.TestCase):
    """A test class for testing LLamaDecodeLayerInterleave."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch("mindspore.context.get_context")
    def test_init(self, mock_get):
        """
        Feature: LLamaDecodeLayerInterleave
        Description: Test LLamaDecodeLayerInterleave init
        Expectation: No Exception
        """
        mock_get.return_value = ms.context.GRAPH_MODE
        parallelconfig = TransformerOpParallelConfig(use_seq_parallel=True)
        llama_decode_layer_interleave = LLamaDecodeLayerInterleave(batch_size=1, seq_length=1024, layer_id=0)
        llama_decode_layer_interleave2 = LLamaDecodeLayerInterleave(batch_size=1, seq_length=1024, layer_id=0,
                                                                    parallel_config=parallelconfig)
        self.assertEqual(llama_decode_layer_interleave.seq_length, 1024)
        self.assertEqual(llama_decode_layer_interleave2.seq_length, 1024)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_interleave.LLamaDecodeLayerInterleave.__init__',
           MockLLamaDecodeLayerInterleave.__init__)
    def test_construct(self):
        """
        Feature: LLamaDecodeLayerInterleave
        Description: Test LLamaDecodeLayerInterleave construct
        Expectation: No Exception
        """
        llama_decode_layer_interleave = LLamaDecodeLayerInterleave(batch_size=1, seq_length=1024, layer_id=0)
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)

        freqs_cis = np.random.randn(seq_len, 128).astype(np.float32)
        freqs_cis_tensor = Tensor(freqs_cis, dtype=ms.float32)
        out = llama_decode_layer_interleave.construct(input_x_tensor, freqs_cis_tensor)
        self.assertEqual(out.shape, (bs, seq_len, hidden_dim))


class MockLLamaDecodeLayerInterleave2(nn.Cell):
    """A mock class for mocking LLamaDecodeLayerInterleave."""

    def __init__(self,
                 batch_size,
                 seq_length,
                 layer_id
                 ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.layer_id = layer_id
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        attention_np_matrix = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        attention_rt_tensor = Tensor(attention_np_matrix, dtype=ms.bfloat16)

        self.attention = MagicMock(return_value=attention_rt_tensor)
        self.attention_norm = MagicMock(return_value=attention_rt_tensor)
        self.attention.compute_qkv = MagicMock(return_value=(attention_rt_tensor, attention_rt_tensor,
                                                             attention_rt_tensor))
        self.attention.cal_output_proj = MagicMock(return_value=attention_rt_tensor)
        self.add = MagicMock(return_value=attention_rt_tensor)
        self.ffn_norm = MagicMock(return_value=attention_rt_tensor)
        self.feed_forward = MagicMock(return_value=attention_rt_tensor)


class TestLLamaDecodeLayerInterleave2(unittest.TestCase):
    """A test class for testing LLamaDecodeLayerInterleave."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_interleave.LLamaDecodeLayerInterleave.__init__',
           MockLLamaDecodeLayerInterleave2.__init__)
    def test_check_input(self):
        """
        Feature: LLamaDecodeLayerInterleave
        Description: Test LLamaDecodeLayerInterleave _check_input
        Expectation: No Exception
        """
        llama_decode_layer_interleave = LLamaDecodeLayerInterleave(batch_size=1, seq_length=1024, layer_id=0)
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        # pylint: disable=W0212
        out = llama_decode_layer_interleave._check_input(input_x_tensor, (input_x_tensor, input_x_tensor,
                                                                          input_x_tensor), input_x_tensor)
        self.assertEqual(out, True)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_interleave.LLamaDecodeLayerInterleave.__init__',
           MockLLamaDecodeLayerInterleave2.__init__)
    def test_linear_layer1(self):
        """
        Feature: LLamaDecodeLayerInterleave
        Description: Test LLamaDecodeLayerInterleave linear_layer1
        Expectation: No Exception
        """
        llama_decode_layer_interleave = LLamaDecodeLayerInterleave(batch_size=1, seq_length=1024, layer_id=0)
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)

        query, key, value = llama_decode_layer_interleave.linear_layer1(input_x_tensor)
        self.assertEqual(query.shape, (bs, seq_len, hidden_dim))
        self.assertEqual(key.shape, (bs, seq_len, hidden_dim))
        self.assertEqual(value.shape, (bs, seq_len, hidden_dim))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.llama.llama_interleave.LLamaDecodeLayerInterleave.__init__',
           MockLLamaDecodeLayerInterleave2.__init__)
    def test_linear_layer2(self):
        """
        Feature: LLamaDecodeLayerInterleave
        Description: Test LLamaDecodeLayerInterleave linear_layer2
        Expectation: No Exception
        """
        llama_decode_layer_interleave = LLamaDecodeLayerInterleave(batch_size=1, seq_length=1024, layer_id=0)
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        # pylint: disable=W0212
        out = llama_decode_layer_interleave.linear_layer2(input_x_tensor, input_x_tensor)
        self.assertEqual(out.shape, (bs, seq_len, hidden_dim))
