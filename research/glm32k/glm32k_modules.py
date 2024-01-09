# Copyright 2023 Huawei Technologies Co., Ltd
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
"""ChatGLM32k Modules."""
import numpy as np

import mindspore.common.dtype as mstype
import mindspore.ops.operations as P
from mindspore import nn, Parameter, Tensor
from mindspore.common.initializer import initializer

from mindformers.modules.layers import Linear

from glm32k_config import ChatGLM32kConfig


def precompute_rotary_emb_cache(seq_len: int, dim: int, dtype=np.float32, rope_ratio: int = 1, base: int = 10000):
    """pre compute rotary emb cache."""
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    base = base * rope_ratio
    theta = 1.0 / (base ** (np.arange(0, dim, 2, dtype=dtype) / dim))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = np.arange(seq_len, dtype=dtype)

    # Calculate the product of position index and $\theta_i$
    idx_theta = np.outer(seq_idx, theta).astype(np.float32)

    cache = np.stack((np.cos(idx_theta), np.sin(idx_theta)), axis=-1).astype(dtype)
    return cache


class ChatGLM32kRMSNorm(nn.Cell):
    r"""
    A self-defined RMSNorm operation using reduce mean.

        Args:
            dim (tuple): The shape of the input tensor
            eps (float): The epsilon value of the denominator. Default 1e-5.
            param_init_type: The param init type.
        Inputs:
            - **x** (Tensor) - Tensor of shape :math:`(batch, seq\_length, hidden\_size)`.

        Outputs:
            Tensor of shape :math:`(batch, seq_length, hidden_size)`.
    """
    def __init__(self, dim, eps=1e-6, param_init_type=mstype.float32):
        super(ChatGLM32kRMSNorm, self).__init__()
        self.eps = Tensor(float(eps), dtype=param_init_type)
        self.weight = Parameter(initializer('ones', (dim,), dtype=param_init_type))
        self.square = P.Square()
        self.mean = P.ReduceMean(keep_dims=True)
        self.add = P.Add()
        self.rsqrt = P.Rsqrt()
        self.mul = P.Mul()
        self.mul2 = P.Mul()

    def _norm(self, x):
        # shard:(dp, 1, 1)
        norm_factor = self.square(x)
        norm_factor = self.mean(norm_factor, -1)
        norm_factor = self.add(norm_factor, self.eps)
        norm_factor = self.rsqrt(norm_factor)
        return self.mul(x, norm_factor)

    def construct(self, x):
        """Forward of RMSNorm."""
        output = self._norm(x)
        output = self.mul2(output, self.weight)
        return output

    def shard(self, strategy):
        """Parallel strategy configuratiuon interface."""
        self.square.shard(strategy)
        self.mean.shard(strategy)
        self.rsqrt.shard(strategy)
        self.add.shard((strategy[0], ()))
        self.mul.shard((strategy[0], strategy[0]))
        self.mul2.shard((strategy[0], (1,)))

    def recompute(self, mode=True):
        self.square.recompute(mode)
        self.mean.recompute(mode)
        self.rsqrt.recompute(mode)
        self.add.recompute(mode)
        self.mul.recompute(mode)
        self.mul2.recompute(mode)


class ChatGLM32kSiLU(nn.Cell):
    r"""
    A self-defined SwiGlu.

        Inputs:
            - **x** (Tensor) - Tensor.

        Outputs:
            Tensor. x = x * sigmod(x).
    """
    def __init__(self):
        super(ChatGLM32kSiLU, self).__init__()
        self.sigmoid = P.Sigmoid()
        self.mul = P.Mul()

    def shard(self, strategy):
        self.sigmoid.shard(strategy)
        self.mul.shard((strategy[0], strategy[0]))

    def construct(self, x):
        return self.mul(x, self.sigmoid(x))


class ChatGLM32kSwiGLU(nn.Cell):
    """SwiGLU activation function."""
    def __init__(self):
        super(ChatGLM32kSwiGLU, self).__init__()
        self.split = P.Split(axis=-1, output_num=2)
        self.silu = ChatGLM32kSiLU()
        self.mul = P.Mul()

    def construct(self, x):
        x0, x1 = self.split(x)
        return self.mul(self.silu(x0), x1)

    def shard(self, strategy):
        self.split.shard(strategy)
        # self.split.shard(((4, 1, 1),))
        self.silu.shard(strategy)
        self.mul.shard((strategy[0], strategy[0]))


class ChatGLM32kMLP(nn.Cell):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """
    def __init__(self, config: ChatGLM32kConfig):
        super(ChatGLM32kMLP, self).__init__()
        self.add_bias = config.add_bias_linear
        self.dense_h_to_4h = Linear(
            config.hidden_size,
            config.ffn_hidden_size * 2,
            has_bias=self.add_bias,
            param_init_type=config.param_init_type,
            compute_dtype=config.compute_dtype,
        )

        self.activation_func = ChatGLM32kSwiGLU()
        self.cast = P.Cast()

        # Project back to h.
        self.dense_4h_to_h = Linear(
            config.ffn_hidden_size,
            config.hidden_size,
            has_bias=self.add_bias,
            param_init_type=config.param_init_type,
            compute_dtype=config.compute_dtype,
        )

    def construct(self, hidden_states):
        # [bs, seq_len, 4 * hidden_size]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [bs, seq_len, hidden_size]
        output = self.dense_4h_to_h(intermediate_parallel)
        return output

    def shard(self, parallel_config):
        """sharding for feedforward"""
        dp = parallel_config.data_parallel

        self.dense_h_to_4h.shard(strategy_matmul=((dp, 1), (1, 1)),
                                 strategy_bias=((dp, 1), (1,)))
        self.activation_func.shard(((dp, 1, 1),))
        self.dense_4h_to_h.shard(strategy_matmul=((dp, 1), (1, 1)),
                                 strategy_bias=((dp, 1), (1,)))
