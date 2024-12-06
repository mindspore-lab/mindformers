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
"""test glm layers."""
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor
from mindspore import nn
from mindspore import ops
from mindspore.ops import operations as P
from mindformers.models.glm.layers import GEGLU, MLPWithGEGLU
ms.set_context(mode=ms.PYNATIVE_MODE)

class MockGEGLU(nn.Cell):
    """A mock class for mocking GEGLU."""

    def __init__(self, parallel_config):
        # pylint: disable=E1003
        super(GEGLU, self).__init__()
        self.split = ops.Split(output_num=2, axis=-1)
        self.activation_fn = P.GeLU()
        self.parallel_config = parallel_config


class TestGEGLU(unittest.TestCase):
    """A test class for testing GEGLU."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.glm.layers.GEGLU.__init__', MockGEGLU.__init__)
    def test_construct(self):
        """
        Feature: GEGLU
        Description: Test GEGLU construct
        Expectation: No Exception
        """
        geglu = GEGLU(parallel_config=1)
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.float32)

        out = geglu.construct(input_x_tensor)
        self.assertEqual(out.shape, (bs, seq_len, hidden_dim/2))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_init(self):
        geglu = GEGLU(parallel_config=1)
        self.assertEqual(geglu.parallel_config, 1)


class MockMLPWithGEGLU(nn.Cell):
    """A mock class for mocking MLPWithGEGLU."""

    def __init__(self,
                 hidden_size,
                 output_dropout_prob):
        self.hidden_size = hidden_size
        self.output_dropout_prob = output_dropout_prob
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        attention_np_matrix = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        attention_rt_tensor = Tensor(attention_np_matrix, dtype=ms.float32)
        self.dense_h_to_4h = MagicMock(return_value=attention_rt_tensor)
        self.activation_func = MagicMock(return_value=attention_rt_tensor)
        self.dense_4h_to_h = MagicMock(return_value=attention_rt_tensor)


class TestMLPWithGEGLU(unittest.TestCase):
    """A test class for testing MLPWithGEGLU."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.glm.layers.MLPWithGEGLU.__init__', MockMLPWithGEGLU.__init__)
    def test_mlp_forward(self):
        """
        Feature: MLPWithGEGLU
        Description: Test MLPWithGEGLU mlp_forward
        Expectation: No Exception
        """
        mlpwithgeglu = MLPWithGEGLU(hidden_size=512, output_dropout_prob=1)
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.float32)

        out = mlpwithgeglu.mlp_forward(input_x_tensor)
        self.assertEqual(out.shape, (bs, seq_len, hidden_dim))


class MockMLPWithGEGLU2(nn.Cell):
    """A mock class for mocking MLPWithGEGLU."""

    def __init__(self,
                 hidden_size,
                 output_dropout_prob):
        self.hidden_size = hidden_size
        self.output_dropout_prob = output_dropout_prob
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        attention_np_matrix = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        attention_rt_tensor = Tensor(attention_np_matrix, dtype=ms.float32)
        self.mlp_forward = MagicMock(return_value=attention_rt_tensor)
        self.dropout = MagicMock(return_value=attention_rt_tensor)
        self.training = True


class TestMLPWithGEGLU2(unittest.TestCase):
    """A test class for testing MLPWithGEGLU."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.glm.layers.MLPWithGEGLU.__init__', MockMLPWithGEGLU2.__init__)
    def test_construct(self):
        """
        Feature: MLPWithGEGLU
        Description: Test MLPWithGEGLU construct
        Expectation: No Exception
        """
        mlpwithgeglu = MLPWithGEGLU(hidden_size=512, output_dropout_prob=1)
        bs = 1
        seq_len = 1024
        hidden_dim = 4096
        input_x = np.random.randn(bs, seq_len, hidden_dim).astype(np.float16)
        input_x_tensor = Tensor(input_x, dtype=ms.float32)

        out = mlpwithgeglu.construct(input_x_tensor)
        self.assertEqual(out.shape, (bs, seq_len, hidden_dim))
