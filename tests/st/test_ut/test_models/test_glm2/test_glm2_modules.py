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
"""test glm2 modules."""
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, nn, Parameter
from mindspore.common.initializer import initializer
import mindspore.common.dtype as mstype
from mindformers.models.glm2.glm2_modules import FreqsMgr, ChatGLM2RMSNorm
ms.set_context(mode=ms.PYNATIVE_MODE)

class MockFreqsMgr(nn.Cell):
    """A mock class for mocking FreqsMgr."""

    def __init__(self,
                 dim,
                 ):
        self.head_dim = dim
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        attention_np_matrix = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        attention_rt_tensor = Tensor(attention_np_matrix, dtype=ms.bfloat16)
        self.freqs_cos = attention_rt_tensor
        self.freqs_sin = attention_rt_tensor
        self.cache = attention_rt_tensor
        self.slice = MagicMock(return_value=attention_rt_tensor)
        self.tile = MagicMock(return_value=attention_rt_tensor)
        self.gather = MagicMock(return_value=attention_rt_tensor)

class TestFreqsMgr(unittest.TestCase):
    """A test class for testing FreqsMgr."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.glm2.glm2_modules.FreqsMgr.__init__', MockFreqsMgr.__init__)
    def test_construct(self):
        """
        Feature: FreqsMgr
        Description: Test FreqsMgr construct
        Expectation: No Exception
        """
        freqsmgr = FreqsMgr(dim=512)
        bs = 1
        seq_len = 1024
        hidden_dim = 4096

        freqs_cos, freqs_sin, cache = freqsmgr.construct(seq_length=1024)
        self.assertEqual(freqs_cos.shape, (bs, seq_len, hidden_dim))
        self.assertEqual(freqs_sin.shape, (bs, seq_len, hidden_dim))
        self.assertEqual(cache.shape, (bs, seq_len, hidden_dim))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.glm2.glm2_modules.FreqsMgr.__init__', MockFreqsMgr.__init__)
    def test_prefill(self):
        """
        Feature: FreqsMgr
        Description: Test FreqsMgr prefill
        Expectation: No Exception
        """
        freqsmgr = FreqsMgr(dim=512)
        bs = 1
        seq_len = 1024
        hidden_dim = 4096

        freqs_cos, freqs_sin, cache = freqsmgr.prefill(bsz=1, seq_length=1024)
        self.assertEqual(freqs_cos.shape, (bs, seq_len, hidden_dim))
        self.assertEqual(freqs_sin.shape, (bs, seq_len, hidden_dim))
        self.assertEqual(cache.shape, (bs, seq_len, hidden_dim))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.glm2.glm2_modules.FreqsMgr.__init__', MockFreqsMgr.__init__)
    def test_increment(self):
        """
        Feature: FreqsMgr
        Description: Test FreqsMgr increment
        Expectation: No Exception
        """
        freqsmgr = FreqsMgr(dim=512)
        bs = 1
        seq_len = 1024
        hidden_dim = 4096

        freqs_cos, freqs_sin, cache = freqsmgr.increment(batch_valid_length=1024)
        self.assertEqual(freqs_cos.shape, (bs, seq_len, hidden_dim))
        self.assertEqual(freqs_sin.shape, (bs, seq_len, hidden_dim))
        self.assertEqual(cache.shape, (bs, seq_len, hidden_dim))


class MockChatGLM2RMSNorm(nn.Cell):
    """A mock class for mocking ChatGLM2RMSNorm."""

    def __init__(self, dim, eps=1e-6, param_init_type=mstype.float32):
        # pylint: disable=E1003
        super(ChatGLM2RMSNorm, self).__init__()
        self.eps = Tensor(float(eps), dtype=param_init_type)
        self.weight = Parameter(initializer('ones', (dim,), dtype=param_init_type))
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        attention_np_matrix = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        attention_rt_tensor = Tensor(attention_np_matrix, dtype=ms.bfloat16)
        self.square = MagicMock(return_value=attention_rt_tensor)
        self.mean = MagicMock(return_value=attention_rt_tensor)
        self.add = MagicMock(return_value=attention_rt_tensor)
        self.rsqrt = MagicMock(return_value=attention_rt_tensor)
        self.mul = MagicMock(return_value=attention_rt_tensor)
        self.mul2 = MagicMock(return_value=attention_rt_tensor)
        self.rms_norm = MagicMock(return_value=attention_rt_tensor)


class TestChatGLM2RMSNorm(unittest.TestCase):
    """A test class for testing ChatGLM2RMSNorm."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.glm2.glm2_modules.ChatGLM2RMSNorm.__init__', MockChatGLM2RMSNorm.__init__)
    def test_self_norm(self):
        """
        Feature: ChatGLM2RMSNorm
        Description: Test ChatGLM2RMSNorm _self_norm
        Expectation: No Exception
        """
        chatglm2rmsnorm = ChatGLM2RMSNorm(dim=512)
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        # pylint: disable=W0212
        out = chatglm2rmsnorm._self_norm(input_x_tensor)
        self.assertEqual(out.shape, (bs, seq_len, hidden_dim))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.glm2.glm2_modules.ChatGLM2RMSNorm.__init__', MockChatGLM2RMSNorm.__init__)
    def construct(self):
        """
        Feature: ChatGLM2RMSNorm
        Description: Test ChatGLM2RMSNorm construct
        Expectation: No Exception
        """
        chatglm2rmsnorm = ChatGLM2RMSNorm(dim=512)
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.bfloat16)
        # pylint: disable=W0212
        out = chatglm2rmsnorm.construct(input_x_tensor)
        self.assertEqual(out.shape, (bs, seq_len, hidden_dim))
