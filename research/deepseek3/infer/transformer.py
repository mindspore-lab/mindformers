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
""" For transformer """
import mindspore.common.dtype as mstype
from mindspore import Parameter, mint, nn, ops
from mindspore.common.initializer import initializer

from mindformers.parallel_core.inference.utils import get_tp_world_size
from mindformers.tools.utils import divide
from research.deepseek3.infer.activation import SiLU
from research.deepseek3.infer.layers import ColumnParallelLinear, RowParallelLinear


class VocabEmbedding(nn.Cell):
    """
    Embedding Layer.

    Args:
            - **num_embeddings** (int): Size of the dictionary of embeddings.
            - **embedding_dim** (int): The size of each embedding vector.
            - **param_init_type** (mstype): The param init type, default mstype.float32.
            - **param_init** (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the embedding_table.
                Refer to class `initializer` for the values of string when a string
                is specified. Default: 'normal'.
    Inputs:
            - **input_ids** (Tensor) - The tokenized inputs with datatype int32 with shape (batch_size, seq_length)

    Outputs:
            - **output** (Tensor) - The embedding vector for the input with shape (batch_size,
              seq_length, embedding_size).
    """

    def __init__(self, num_embeddings, embedding_dim, param_init_type=mstype.float32, param_init='normal',
                 parallel_optimizer=False):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding_weight = Parameter(
            initializer(param_init, [self.num_embeddings, self.embedding_dim], dtype=param_init_type),
            name='embedding_weight', parallel_optimizer=parallel_optimizer)
        self.gather = ops.Gather()

    def construct(self, input_ids):
        """Forward of vocab embedding."""
        # 'embedding' has dynamic shape issue, use gather instead now.
        output = self.gather(self.embedding_weight, input_ids, 0)
        return output


class ParallelMLP(nn.Cell):
    r"""
    Implementation of parallel feedforward block.

    Args:
        config (dict): Configuration.
        is_expert (book): This block is an expert block. Default: False.

    Inputs:
        - **hidden_states** (Tensor) - Tensor of shape :math:`(B, S, H)`.

    Outputs:
        - **output** (Tensor) - Output tensor of shape :math:`(B, S, H)`.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(self, config, is_expert=False):
        super().__init__(config)
        if is_expert:
            raise NotImplementedError("For ParallelMLP, `is_expert` is not supported for now.")
        self.config = config
        self.has_bias = self.config.mlp_has_bias
        self.hidden_size = self.config.hidden_size
        self.ffn_hidden_size = self.config.ffn_hidden_size
        self.mlp_has_gate = self.config.mlp_has_gate
        self.ffn_concat = self.config.ffn_concat

        tp_group_size = get_tp_world_size()
        self.ffn_hidden_size_per_partition = divide(self.ffn_hidden_size, tp_group_size)

        if self.mlp_has_gate:
            if self.ffn_concat:
                self.w_gate_hidden = ColumnParallelLinear(
                    self.hidden_size,
                    self.ffn_hidden_size * 2,
                    config=self.config.parallel_config,
                    bias=self.has_bias,
                    transpose_b=True,
                    gather_output=False,
                    is_expert=is_expert,
                    param_init_type=self.config.param_init_dtype,
                    compute_dtype=self.config.compute_dtype,
                )
            else:
                self.w1 = ColumnParallelLinear(
                    self.hidden_size,
                    self.ffn_hidden_size,
                    config=self.config.parallel_config,
                    bias=self.has_bias,
                    transpose_b=True,
                    gather_output=False,
                    is_expert=is_expert,
                    param_init_type=self.config.param_init_dtype,
                    compute_dtype=self.config.compute_dtype,
                )
                self.w3 = ColumnParallelLinear(
                    self.hidden_size,
                    self.ffn_hidden_size,
                    config=self.config.parallel_config,
                    bias=self.has_bias,
                    transpose_b=True,
                    gather_output=False,
                    is_expert=is_expert,
                    param_init_type=self.config.param_init_dtype,
                    compute_dtype=self.config.compute_dtype,
                )
        else:
            self.w1 = ColumnParallelLinear(
                self.hidden_size,
                self.ffn_hidden_size,
                config=self.config.parallel_config,
                bias=self.has_bias,
                transpose_b=True,
                gather_output=False,
                is_expert=is_expert,
                param_init_type=self.config.param_init_dtype,
                compute_dtype=self.config.compute_dtype,
            )

        self.act_type = self.config.hidden_act
        self.act_func = SiLU()

        # Project back to h.
        self.w2 = RowParallelLinear(
            self.ffn_hidden_size,
            self.hidden_size,
            input_is_parallel=True,
            config=self.config.parallel_config,
            bias=self.has_bias,
            transpose_b=True,
            is_expert=is_expert,
            param_init_type=self.config.param_init_dtype,
            compute_dtype=self.config.compute_dtype,
        )
        self.mul = ops.Mul()
        self.reshape = ops.Reshape()

    def construct(self, x):
        """ Construct function of mlp block. """
        # [B, S, H] -> [B, S, ffn_H]
        if self.mlp_has_gate:
            if self.ffn_concat:
                gate_hidden_out = self.w_gate_hidden(x)  # dp,1 -> dp, mp  # dp,1 -> dp, mp
                gate_hidden_out_shape = gate_hidden_out.shape
                reshape_out = self.reshape(gate_hidden_out,
                                           (*gate_hidden_out_shape[:-1], self.ffn_hidden_size_per_partition, 2))
                gate, hidden = mint.split(reshape_out,
                                          (1, 1), -1)
                gate = self.reshape(gate, (*gate_hidden_out_shape[:-1], self.ffn_hidden_size_per_partition))
                hidden = self.reshape(hidden, (*gate_hidden_out_shape[:-1], self.ffn_hidden_size_per_partition))
            else:
                gate = self.w1(x)  # dp,1 -> dp, mp
                hidden = self.w3(x)  # dp,1 -> dp, mp
            gate = self.act_func(gate)
            hidden = mint.mul(hidden, gate)
        else:
            hidden = self.w1(x)
            hidden = self.act_func(hidden)

        # [B, S, ffn_H] -> [B, S, H]
        output = self.w2(hidden)
        return output
