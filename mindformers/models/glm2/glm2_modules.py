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
"""ChatGLM2 Modules."""
import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.ops.operations as P
from mindspore import nn, Parameter, Tensor
from mindspore.common.initializer import initializer

from mindformers.modules.layers import Linear
from mindformers.tools.utils import is_version_ge

from .glm2_config import ChatGLM2Config


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


class RopeCache(nn.Cell):
    r"""A self-defined RopeCache operation"""
    def __init__(self, config, dim=None, dtype=np.float32, base=10000, is_dynamic=False):
        super().__init__()
        base = base * config.rope_ratio
        theta = 1.0 / (base ** (np.arange(0, dim, 2, dtype=dtype) / dim))
        seq_idx = np.arange(config.seq_length, dtype=dtype)
        idx_theta = np.outer(seq_idx, theta).astype(np.float32)
        self.is_dynamic = is_dynamic
        self.use_past = config.use_past
        self.seq_length = config.seq_length
        self.dim = dim
        self.rotary_pos_emb = np.stack((np.cos(idx_theta), np.sin(idx_theta)), axis=-1).astype(dtype)
        self.rotary_pos_emb = Tensor(self.rotary_pos_emb, config.compute_dtype)

        self.reshape = P.Reshape()
        self.half_dim = dim // 2
        if is_dynamic:
            self.reshape.add_prim_attr("skip_redistribution", True)
        self.slice = P.StridedSlice().shard(((1, 1),))
        self.gather = P.Gather().shard(((1, 1), (1,)))

    def construct(self, seq_len=None):
        rotary_pos_emb = self.rotary_pos_emb
        if self.is_dynamic and self.use_past:
            rotary_pos_emb = self.slice(rotary_pos_emb, (0, 0, 0), (seq_len, self.half_dim, 2), (1, 1, 1))
        return rotary_pos_emb

    def increment(self, batch_valid_length, batch_size):
        rotary_pos_emb = self.gather(self.rotary_pos_emb, batch_valid_length, 0)
        rotary_pos_emb = self.reshape(rotary_pos_emb, (batch_size, 1, self.half_dim, 2))
        return rotary_pos_emb


class ChatGLM2RMSNorm(nn.Cell):
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
        super(ChatGLM2RMSNorm, self).__init__()
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

class ChatGLM2SiLU(nn.Cell):
    r"""
    A self-defined SwiGlu.

        Inputs:
            - **x** (Tensor) - Tensor.

        Outputs:
            Tensor. x = x * sigmod(x).
    """
    def __init__(self):
        super(ChatGLM2SiLU, self).__init__()
        self.sigmoid = P.Sigmoid()
        self.mul = P.Mul()

    def shard(self, strategy):
        self.sigmoid.shard(strategy)
        self.mul.shard((strategy[0], strategy[0]))

    def construct(self, x):
        return self.mul(x, self.sigmoid(x))


class ChatGLM2SwiGLU(nn.Cell):
    """SwiGLU activation function."""
    def __init__(self):
        super(ChatGLM2SwiGLU, self).__init__()
        self.split = P.Split(axis=-1, output_num=2)
        self.silu = ChatGLM2SiLU()
        self.mul = P.Mul()

    def construct(self, x):
        x0, x1 = self.split(x)
        return self.mul(self.silu(x0), x1)

    def shard(self, strategy):
        self.split.shard(strategy)
        self.silu.shard(strategy)
        self.mul.shard((strategy[0], strategy[0]))


class ChatGLM2MLP(nn.Cell):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """
    def __init__(self, config: ChatGLM2Config):
        super(ChatGLM2MLP, self).__init__()
        self.add_bias = config.add_bias_linear
        self.prefix_name = config.prefix_name
        self.dense_h_to_4h = Linear(
            config.hidden_size,
            config.ffn_hidden_size * 2,
            has_bias=self.add_bias,
            param_init_type=config.param_init_type,
            compute_dtype=config.compute_dtype,
            skip_redistribution=config.is_dynamic
        )

        self.activation_func = ChatGLM2SwiGLU()

        # Project back to h.
        self.dense_4h_to_h = Linear(
            config.ffn_hidden_size,
            config.hidden_size,
            has_bias=self.add_bias,
            param_init_type=config.param_init_type,
            compute_dtype=config.compute_dtype,
            skip_redistribution=config.is_dynamic
        )

    def construct(self, hidden_states):
        # [bs, seq_len, 4 * hidden_size]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        origin_dtype = intermediate_parallel.dtype
        intermediate_parallel = self.cast(intermediate_parallel, mstype.float32)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        intermediate_parallel = self.cast(intermediate_parallel, origin_dtype)
        # [bs, seq_len, hidden_size]
        output = self.dense_4h_to_h(intermediate_parallel)
        return output

    def shard(self, parallel_config):
        """sharding for feedforward"""
        dp, mp = parallel_config.data_parallel, parallel_config.model_parallel
        if self.prefix_name.startswith("glm32k"):
            mp = 1
        self.dense_h_to_4h.shard(strategy_matmul=((dp, 1), (mp, 1)),
                                 strategy_bias=((dp, mp), (mp,)))
        self.activation_func.shard(((dp, 1, 1),))
        self.dense_4h_to_h.shard(strategy_matmul=((dp, mp), (1, mp)),
                                 strategy_bias=((dp, 1), (1,)))


# pylint: disable=R1703
def check_promt_flash_attention_version():
    """
    the outputs of prompt flash attention are different when using 2.3 and ms version below 2.3
    """
    cur_ver = ms.__version__
    if is_version_ge(cur_ver, "2.3"):
        pfa_flag = True
    else:
        pfa_flag = False
    return pfa_flag
