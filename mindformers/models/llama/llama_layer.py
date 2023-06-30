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
"""LLaMA Model Layers' APIs."""

import numpy as np

from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore import nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn.cell import Cell

try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator
from mindspore import log as logger
from mindspore.common.initializer import initializer
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindspore.context import ParallelMode
from mindformers.modules.layers import Linear, _check_input_dtype, _args_type_validator_check, _valid_value_checks
from mindformers.modules.transformer.op_parallel_config import _check_config
from mindformers.modules.transformer import TransformerOpParallelConfig

from mindformers.tools.logger import _LogActionOnce


class LlamaSiLU(Cell):
    r"""
    A self-defined SwiGlu.

        Inputs:
            - **x** (Tensor) - Tensor.

        Outputs:
            Tensor. x = x * sigmod(x).
    """
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.mul = P.Mul()

    def shard(self, strategy):
        self.sigmoid.sigmoid.shard(strategy)
        self.mul.shard((strategy[0], strategy[0]))

    def construct(self, x):
        return self.mul(x, self.sigmoid(x))


def get_rotary_mask(head_dim):
    """Rotary embedding help mask."""
    rot_np = np.zeros((head_dim, head_dim), dtype=np.float32)
    for i in range(head_dim):
        if i < head_dim // 2:
            rot_np[2 * i, i] = 1.0
        else:
            rot_np[2 * (i - head_dim // 2) + 1, i] = 1.0
    return rot_np


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, dtype=mstype.float32):
    """
    Precompute of freqs and mask for rotary embedding.
    """
    freqs_base = np.arange(0, dim, 2)[: (dim // 2)].astype(np.float32) # (head_dim // 2, )
    freqs = 1.0 / (theta ** (freqs_base / dim)) # (head_dim // 2, )
    t = np.arange(0, end, 1).astype(np.float32)  # type: ignore # (seq_len,)
    freqs = np.outer(t, freqs)  # type: ignore (seq_len, head_dim // 2)
    freqs = np.expand_dims(freqs, axis=-1)
    freqs = np.tile(freqs, (1, 1, 2))
    emb = freqs.reshape((end, -1))
    freqs_cos = np.cos(emb) # (seq_len, head_dim)
    freqs_sin = np.sin(emb) # (seq_len, head_dim)
    freqs_cos = Tensor(freqs_cos, dtype=dtype)
    freqs_sin = Tensor(freqs_sin, dtype=dtype)

    # minus_mask
    minus_mask = Tensor([[0, 1], [-1, 0]], dtype=dtype)

    # rotary_mask
    rotary_mask = get_rotary_mask(dim)
    rotary_mask = Tensor(rotary_mask, dtype=dtype)

    return freqs_cos, freqs_sin, minus_mask, rotary_mask


class LlamaRotaryEmbedding(Cell):
    r"""
    Rotary Position Embedding.

    Args:
            - **head_dim** (int): The dim of multi head attention.
            - **compute_dtype** (mstype): The compute type, default mstype.float16.
            - **parallel_config** (dict): - Parallel Config.
    Inputs:
            - **x** (Tensor) - Tensor of shape :math:`(batch, seq\_length, hidden\_size)`.

    Outputs:
            Tensor of shape :math:`(batch, seq_length, hidden_size)`.
    """

    def __init__(
            self,
            head_dim=128,
            compute_dtype=mstype.float16,
            parallel_config=TransformerOpParallelConfig()
    ):
        super().__init__(auto_prefix=False)
        self.dtype = compute_dtype
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.rank = P.Rank()
        self.head_dim = head_dim
        mp = parallel_config.model_parallel
        dp = parallel_config.data_parallel

        self.add = P.Add().shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
        self.bmm_minus = P.BatchMatMul().shard(((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1,),
                                                (1, 1)))
        self.bmm_rot = P.BatchMatMul().shard(((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
                                              (1, 1)))
        self.mul = P.Mul().shard(((dp, mp, 1, 1), (1, 1,)))
        self.matmul = P.MatMul().shard(((dp, mp), (mp, 1)))

    # rotary pos emb helpers:
    def rotate_half(self, x, minus_mask):
        # x: b, head_num, T, H_D
        batch_size, head_num, seq_length, head_dim = x.shape
        # shard:(dp, mp, 1, 1)
        x = self.reshape(x, (batch_size, head_num, seq_length * head_dim // 2, 2))
        # shard:(dp, mp, 1, 1)
        x = self.bmm_minus(x, minus_mask)
        # shard:(dp, mp, 1, 1)
        x = self.reshape(x, (batch_size, head_num, seq_length, head_dim))
        return x

    def construct(self, xq: Tensor, xk: Tensor, freqs_cis):
        """Forward of rotary position embedding."""
        # xq, xk: b, t, head_num, head_dim
        freqs_cos, freqs_sin, mins_mask, rotary_mask = freqs_cis

        xq_out = self.add(self.mul(xq, freqs_cos),
                          self.mul(self.rotate_half(xq, mins_mask), freqs_sin))

        xq_out = self.bmm_rot(xq_out, rotary_mask)

        xk_out = self.add(self.mul(xk, freqs_cos),
                          self.mul(self.rotate_half(xk, mins_mask), freqs_sin))

        xk_out = self.bmm_rot(xk_out, rotary_mask)

        return xq_out, xk_out


class LlamaEmbedding(Cell):
    """
    Embedding Layer.

    Args:
            - **vocab_size** (int): Size of the dictionary of embeddings.
            - **embedding_size** (int): The size of each embedding vector.
            - **param_init_type** (mstype): The param init type, default mstype.float32.
            - **parallel_config** (TransformerOpParallelConfig): The parallel config of network. Default
                `default_embedding_parallel_config`, an instance of `EmbeddingOpParallelConfig` with default args.
            - **param_init** (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the embedding_table.
                Refer to class `initializer` for the values of string when a string
                is specified. Default: 'normal'.
    Inputs:
            - **input_ids** (Tensor) - The tokenized inputs with datatype int32 with shape (batch_size, seq_length)

    Outputs:
            - **output** (Tensor) - The embedding vector for the input with shape (batch_size,
              seq_length, embedding_size).
    """

    @_LogActionOnce(m_logger=logger, key='Embedding',
                    no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,))
    @_args_type_validator_check(vocab_table_size=Validator.check_positive_int,
                                embedding_size=Validator.check_positive_int)
    def __init__(self, vocab_table_size, embedding_size, param_init_type=mstype.float32,
                 parallel_config=TransformerOpParallelConfig(), param_init='normal'):
        super().__init__()
        _check_config(parallel_config)
        self.vocab_table_size = vocab_table_size
        self.embedding_size = embedding_size
        self.embedding_weight = Parameter(initializer(param_init,
                                                      [self.vocab_table_size, self.embedding_size],
                                                      dtype=param_init_type),
                                          name='embedding_weight', parallel_optimizer=False)

        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel

        if parallel_config.vocab_emb_dp:
            self.gather = P.Gather().shard(((1, 1), (dp, 1)))
            logger.info(f"Using {dp} data parallel for the embedding lookup.")
        else:
            if self.vocab_table_size % mp != 0:
                raise ValueError(f"The vocab size of the embedding {self.vocab_table_size} must be a "
                                 f"multiple of parallel_config.model_parallel {mp}.")
            self.gather = P.Gather().shard(((mp, 1), (dp, 1)))
            logger.info(f"Using {dp} data parallel and {mp} "
                        f"model parallel for the embedding lookup.")

    def construct(self, input_ids):
        """Forward of vocab embedding."""
        _check_input_dtype(F.dtype(input_ids), "input_ids", [mstype.int32, mstype.int64], self.cls_name)
        output = self.gather(self.embedding_weight, input_ids, 0)
        return output


class LlamaRMSNorm(nn.Cell):
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
        super(LlamaRMSNorm, self).__init__()
        self.eps = eps
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


class LlamaFeedForward(Cell):
    r"""
    LLaMA FeedForward.

    .. math::
            (xW_1 * xW_3)W_2

        Inputs:
            - **x** (Tensor) - should be `[batch, seq_length, hidden_size] or [batch * seq_length, hidden_size]`.
              Float tensor.

        Outputs:
            Tensor, the output of this layer after mapping. The shape is `[batch, seq_length, hidden_size] or
            [batch * seq_length, hidden_size]`.

        Raises:
            ValueError: `hidden_dim` is not a multiple of the model parallel way.
            ValueError: `dim` is not a multiple of the model parallel way.
    """

    @_LogActionOnce(m_logger=logger, key='FeedForward',
                    no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,))
    @_args_type_validator_check(dim=Validator.check_positive_int,
                                hidden_dim=Validator.check_positive_int,
                                multiple_of=Validator.check_positive_int,
                                compute_dtype=_valid_value_checks([mstype.float32, mstype.float16],
                                                                  "FeedForward"),
                                param_init_type=_valid_value_checks([mstype.float32, mstype.float16],
                                                                    "FeedForward"))
    def __init__(self, dim,
                 hidden_dim,
                 multiple_of,
                 hidden_act=LlamaSiLU,
                 compute_dtype=mstype.float16,
                 param_init_type=mstype.float32,
                 parallel_config=TransformerOpParallelConfig()):
        super().__init__()

        if hidden_act is None or not (isinstance(hidden_act, str) or issubclass(hidden_act, nn.Cell)):
            raise TypeError(f"For FeedForward cell, the hidden_act should str type or nn.Cell type, "
                            f"but got {hidden_act}.")

        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * \
            ((hidden_dim + multiple_of - 1) // multiple_of)

        mp = parallel_config.model_parallel
        # ffn use less dp than other ops when use_moe, due to there are ops use dp and ep.
        dp = parallel_config.data_parallel
        if hidden_dim % mp != 0:
            raise ValueError("For 'FeedForward', the class variable 'hidden_dim' must be a multiple of the"
                             "num of model parallel, but got the hidden_dim is {} and the num of model "
                             "parallel is {}.".format(hidden_dim, mp))
        if dim % mp != 0:
            raise ValueError("For 'FeedForward', the class variable 'dim' must be a multiple of the num of "
                             "model parallel, but got the dim is {} and the num of model parallel is {}."
                             .format(dim, mp))

        input_size = dim
        output_size = hidden_dim
        self.dtype = compute_dtype

        _check_config(parallel_config)

        self.w1 = Linear(in_channels=input_size,
                         out_channels=output_size,
                         activation=hidden_act,
                         has_bias=False,
                         outer_batch=dp,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type)

        self.w2 = Linear(in_channels=output_size,
                         out_channels=input_size,
                         has_bias=False,
                         outer_batch=dp,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type)

        self.w3 = Linear(in_channels=input_size,
                         out_channels=output_size,
                         has_bias=False,
                         outer_batch=dp,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type)

        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.w1.shard(strategy_matmul=((dp, 1), (mp, 1)), strategy_activation=((dp, mp),))
            if hidden_act == LlamaSiLU:
                self.w1.activation.shard(((dp, mp),))

        self.w2.shard(strategy_matmul=((dp, mp), (1, mp)))
        self.w3.shard(strategy_matmul=((dp, 1), (mp, 1)))
        self.mul = P.Mul().shard(((dp, 1, mp), (dp, 1, mp)))

        self.cast = P.Cast()

    def construct(self, x):
        """Forward process of the FeedForward"""
        _check_input_dtype(F.dtype(x), "x", [mstype.float32, mstype.float16], self.cls_name)
        x = self.cast(x, self.dtype)
        # returned shape is [bs, seq_length, hidden_dim] or [bs * seq_length, hidden_dim]
        # dp,1 -> dp, mp
        gate = self.w1(x)
        # dp,1 -> dp, mp
        hidden = self.w3(x)
        # dp,mp -> dp, mp
        hidden = self.mul(hidden, gate)
        # dp,mp -> dp, 1
        output = self.w2(hidden)

        return output
