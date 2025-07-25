# Copyright 2025 Huawei Technologies Co., Ltd
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
DEPRECATED MODULE

This module is deprecated and will be removed in future releases.
For transformer
"""
import math
import os

import numpy as np

import mindspore.common.dtype as mstype
from mindspore import Parameter, Tensor, mint, nn, ops
from mindspore.common.initializer import initializer

from mindformers.parallel_core.inference.utils import divide, get_attn_mask_func
from mindformers.parallel_core.inference.transformer.activation import get_act_func
from mindformers.parallel_core.process_group_config import default_model_comm_pgs
from mindformers.modules.flash_attention import FlashAttention
from mindformers.modules.infer_attention import InferRotaryEmbedding
from mindformers.modules.layers import FreqsMgr, RotaryEmbedding
from mindformers.modules.transformer import LowerTriangularMaskWithDynamic
from mindformers.version_control import need_nz

from research.llama3_1.infer.norm import RMSNorm
from research.llama3_1.infer.parallel_paged_attention_mgr import ParallelPagedAttentionMgr
from research.llama3_1.infer.scale_mask_softmax import ScaleMaskSoftmax
from research.llama3_1.infer.layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding


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
        model_comm_pgs (ModelCommProcessGroups, optional): Model communication process group.
            Default: default_model_comm_pgs.

    Inputs:
        - **hidden_states** (Tensor) - Tensor of shape :math:`(B, S, H)`.

    Outputs:
        - **output** (Tensor) - Output tensor of shape :math:`(B, S, H)`.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(self, config, is_expert=False, model_comm_pgs=default_model_comm_pgs):
        super().__init__(config)
        if is_expert:
            raise NotImplementedError("For ParallelMLP, `is_expert` is not supported for now.")
        self.config = config
        self.has_bias = self.config.mlp_has_bias
        self.hidden_size = self.config.hidden_size
        self.ffn_hidden_size = self.config.ffn_hidden_size
        self.mlp_has_gate = self.config.mlp_has_gate
        self.ffn_concat = self.config.ffn_concat

        self.tp = model_comm_pgs.tp
        tp_group_size = self.tp.size
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
                    tp_group=self.tp,
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
                    tp_group=self.tp,
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
                    tp_group=self.tp,
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
                tp_group=self.tp,
            )

        self.act_type = self.config.hidden_act
        self.act_func = get_act_func(self.act_type)

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
            tp_group=self.tp,
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


class CoreAttention(nn.Cell):
    r"""
    Get the weighted score along the seq_length.

    Args:
        layer_number (int): Number which indicates the index of this transformer layer in the
            whole transformer block.
        config (dict): Configuration.
        attn_type (str): Attention type. Support ['self_attn', 'cross_attn']. Default: 'self_attn'.

    Inputs:
        - **query** (Tensor) - Tensor of query matrix.
        - **key** (Tensor) - Tensor of key matrix.
        - **value** (Tensor) - Tensor of value matrix.
        - **attention_mask** (Tensor) - Tensor of attention mask matrix.

    Outputs:
        - **attn_output** (Tensor) - Tensor of shape :math:`(B, S, H)`.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(self, layer_number, config, attn_mask_type=None):
        super(CoreAttention, self).__init__()
        if attn_mask_type:
            raise NotImplementedError("For CoreAttention, `attn_mask_type` is not supported for now.")
        self.config = config
        self.layer_index = max(1, layer_number)
        self.compute_dtype = self.config.compute_dtype
        self.softmax_compute_dtype = self.config.softmax_compute_dtype
        self.sequence_parallel = self.config.parallel_config.use_sequence_parallel
        self.apply_query_key_layer_scaling = self.config.apply_query_key_layer_scaling
        self.num_heads = self.config.num_heads
        self.hidden_size = self.config.hidden_size
        self.head_dim = divide(self.hidden_size, self.num_heads)

        coeff = None
        norm_factor = math.sqrt(self.head_dim)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_index
            norm_factor *= coeff
        self.inv_norm_factor = Tensor(1.0 / norm_factor, dtype=self.compute_dtype)

        self.mask_func = get_attn_mask_func(self.config.mask_func_type)
        self.scale_mask_softmax = ScaleMaskSoftmax(self.mask_func,
                                                   softmax_compute_type=self.softmax_compute_dtype)

        self.attention_dropout = mint.nn.Dropout(p=self.config.attention_dropout_rate)

    def construct(self, query_layer, key_layer, value_layer, attention_mask):
        """
        Computes the attention scores, applies the attention mask, and returns the weighted
        sum of the value layer based on the attention probabilities.

        Inputs:
        ----------
        query_layer : Tensor
            The query tensor of shape [B, N, S_q, D].
        key_layer : Tensor
            The key tensor of shape [B, N, S_k, D].
        value_layer : Tensor
            The value tensor of shape [B, N, S_k, D].
        attention_mask : Tensor
            The attention mask tensor of shape [B, N, S_q, S_k].

        Returns:
        -------
        Tensor
            The attention output tensor of shape [B, N, S_q, D].
        """
        # score shape: [B, N, S_q, S_k]
        score = ops.bmm(query_layer, key_layer.transpose(0, 1, 3, 2))
        score = mint.mul(score, self.inv_norm_factor)

        # attention scores and attention mask [B, N, S_q, S_k]
        attention_probs = self.scale_mask_softmax(score, attention_mask)

        attention_probs = self.attention_dropout(attention_probs)

        # [B, N, S_q, S_k] * [B, N, S_v, D] -> [B, N, S_q, D]
        weighted_values = ops.bmm(attention_probs, value_layer)

        return weighted_values


class ParallelAttention(nn.Cell):
    r"""
    Parallel attention block.

    Args:
        layer_index (int): Number which indicates the index of this transformer layer in the
            whole transformer block.
        config (dict): Configuration.
        attn_type (str): Attention type. Support ['self_attn', 'cross_attn']. Default: 'self_attn'.
        model_comm_pgs (ModelCommProcessGroups, optional): Model communication process group.
            Default: default_model_comm_pgs.

    Inputs:
        - **hidden_states** (Tensor) - Tensor of shape :math:`(B, S, H)`.
        - **attention_mask** (Tensor) - Tensor of attention mask.
        - **encoder_output** (Tensor) - Tensor of encoder output used for cross attention. Default: None.
        - **rotary_pos_emb** (Tensor) - Tensor of rotary position embedding. Default: None.

    Outputs:
        - **output** (Tensor) - Tensor of shape :math:`(B, S, H)`.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(self, config, layer_number, attention_type="self_attn", attn_mask_type=None,
                 model_comm_pgs=default_model_comm_pgs):
        super().__init__(config)
        if attn_mask_type:
            raise NotImplementedError("For ParallelAttention, `attn_mask_type` is not supported for now.")
        self.config = config
        self.layer_index = max(1, layer_number)
        self.param_init_dtype = self.config.param_init_dtype
        self.compute_dtype = self.config.compute_dtype
        self.is_first_iteration = True
        self.use_past = self.config.use_past
        self.qkv_concat = self.config.qkv_concat

        self.attn_type = attention_type
        self.num_heads = self.config.num_heads
        self.kv_num_heads = self.num_heads if config.n_kv_heads is None else config.n_kv_heads
        self.hidden_size = self.config.hidden_size
        self.head_dim = divide(self.hidden_size, self.num_heads)
        self.kv_hidden_size = self.head_dim * self.kv_num_heads
        self.n_rep = divide(self.num_heads, self.kv_num_heads)

        self.sequence_parallel = self.config.parallel_config.use_sequence_parallel
        self.use_flash_attention = self.config.use_flash_attention
        self.norm_factor = math.sqrt(self.head_dim)

        self.tp = model_comm_pgs.tp
        self.tp_group_size = self.tp.size
        self.num_heads_per_partition = divide(self.num_heads, self.tp_group_size)

        self.use_gqa = (self.num_heads != self.kv_num_heads)

        if self.use_gqa:
            self._check_gqa_valid()
            self.kv_num_heads_per_partition = divide(self.kv_num_heads, self.tp_group_size)
            self.repeat_num = divide(self.num_heads, self.kv_num_heads)
        else:
            self.kv_num_heads_per_partition = self.num_heads_per_partition

        if self.attn_type == "self_attn":
            self._init_self_attn()
        elif self.attn_type == "cross_attn":
            self._init_cross_attn()
        else:
            raise NotImplementedError(
                f"attention_type(str) should be 'self_attn' or 'cross_attn', but got {self.attn_type}")
        self.reshape = ops.Reshape()
        self.cast = ops.Cast()
        self.wo = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            input_is_parallel=True,
            config=self.config.parallel_config,
            bias=self.config.out_proj_has_bias,
            transpose_b=True,
            param_init_type=self.config.param_init_dtype,
            compute_dtype=self.config.compute_dtype,
            tp_group=self.tp,
        )

        if self.use_flash_attention:
            input_layout = "TH" if self.use_past else "BNSD"
            self.flash_attention = FlashAttention(head_num=self.num_heads_per_partition,
                                                  scale_value=1.0 / self.norm_factor,
                                                  next_tokens=0,
                                                  input_layout=input_layout)
        else:
            self.core_attention = CoreAttention(self.layer_index, self.config)

        if self.use_past:
            if need_nz():
                kv_shape = (self.config.num_blocks, self.config.block_size,
                            self.kv_num_heads_per_partition * self.head_dim)
            else:
                kv_shape = (self.config.num_blocks, self.config.block_size,
                            self.kv_num_heads_per_partition, self.head_dim)
            self.npu_mem_size = config.npu_mem_size if hasattr(config, "npu_mem_size") else 2
            self.paged_attention_mgr = ParallelPagedAttentionMgr(self.num_heads_per_partition,
                                                                 self.head_dim,
                                                                 self.kv_num_heads_per_partition,
                                                                 kv_shape,
                                                                 config.seq_length,
                                                                 compute_dtype=self.compute_dtype,
                                                                 npu_mem_size=self.npu_mem_size)
            self.rotary_embedding = InferRotaryEmbedding(rotary_cos_format=2)
        else:
            self.apply_rotary_emb = RotaryEmbedding(self.head_dim, config.rotary_dtype)

    def construct(self, x, batch_valid_length, block_tables, slot_mapping, freqs_cis=None,
                  attn_mask=None, alibi_mask=None, encoder_output=None, prefix_keys_values=None,
                  q_seq_lens=None, key_cache=None, value_cache=None):
        """Construct function of attention block."""
        # hidden states: [B, S, H]
        # apply query, key, value projection
        if self.attn_type == "self_attn":
            if self.qkv_concat:
                qkv = self.cast(self.w_qkv(x), self.compute_dtype)
                reshape_qkv = self.reshape(qkv,
                                           (-1,
                                            self.kv_num_heads_per_partition,
                                            (self.n_rep + 2) * self.head_dim))
                query, key, value = mint.split(reshape_qkv,
                                               (self.head_dim * self.n_rep,
                                                self.head_dim,
                                                self.head_dim), -1)
                if self.use_past:
                    query = self.reshape(query, (-1, self.hidden_size_per_partition))
                    key = self.reshape(key, (-1, self.kv_hidden_size_per_partition))
                    value = self.reshape(value, (-1, self.kv_hidden_size_per_partition))
            else:
                query = self.cast(self.wq(x), self.compute_dtype)
                key = self.cast(self.wk(x), self.compute_dtype)
                value = self.cast(self.wv(x), self.compute_dtype)
                if not self.use_past:
                    # [B, S, H] --> [B, S, N, D]
                    bs, seq_len, _ = x.shape
                    query = self.reshape(query, (bs, seq_len, self.num_heads_per_partition, self.head_dim))
                    key = self.reshape(key, (bs, seq_len, self.kv_num_heads_per_partition, self.head_dim))
                    value = self.reshape(value, (bs, seq_len, self.kv_num_heads_per_partition, self.head_dim))
        else:
            query = self.cast(self.wq(x), self.compute_dtype)
            if self.qkv_concat:
                kv = self.cast(self.w_kv(encoder_output), self.compute_dtype)
                key, value = mint.split(kv, (self.kv_hidden_size_per_partition, self.kv_hidden_size_per_partition), -1)
            else:
                key = self.cast(self.wk(encoder_output), self.compute_dtype)
                value = self.cast(self.wv(encoder_output), self.compute_dtype)

        # qkv shape: [B, S, H]
        if self.use_past:
            if freqs_cis is not None:
                query, key = self.rotary_embedding(query, key, freqs_cis, batch_valid_length)

            if prefix_keys_values is not None:
                prefix_len = prefix_keys_values.shape[2]
                slot_mapping = slot_mapping + self.cast(mint.ne(slot_mapping, -1), mstype.int32) * prefix_len
                if self.is_first_iteration:
                    key, value = self._cat_prefix(key, value, prefix_keys_values)

            key_out = self.paged_attention_mgr(key, value, slot_mapping, batch_valid_length,
                                               key_cache=key_cache, value_cache=value_cache)
            query = ops.depend(query, key_out)

            if self.is_first_iteration:
                if self.use_flash_attention:
                    context_layer = self.flash_attention(query, key, value, attn_mask, alibi_mask, None, None,
                                                         q_seq_lens, batch_valid_length)
                else:
                    bs, seq_len, _ = x.shape
                    # [B, S, H] --> [B, S, N, D]
                    query = query.reshape(bs, seq_len, -1, self.head_dim)
                    key = key.reshape(bs, seq_len, -1, self.head_dim)
                    value = value.reshape(bs, seq_len, -1, self.head_dim)
                    # [B, S, N_kv, D] --> [B, S, N, D]
                    if self.use_gqa:
                        key = mint.repeat_interleave(key, repeats=self.repeat_num, dim=2)
                        value = mint.repeat_interleave(value, repeats=self.repeat_num, dim=2)
                    # [B, S, N, D] --> [B, N, S, D]
                    query = query.transpose(0, 2, 1, 3)
                    key = key.transpose(0, 2, 1, 3)
                    value = value.transpose(0, 2, 1, 3)
                    context_layer = self.core_attention(query, key, value, attn_mask)
                    # [B, N, S, D] --> [B, S, H]
                    context_layer = context_layer.transpose(0, 2, 1, 3).reshape(
                        bs, seq_len, self.hidden_size_per_partition)
            else:
                context_layer = self.paged_attention_mgr.paged_attn(query, batch_valid_length, block_tables,
                                                                    attn_mask, q_seq_lens, key_cache, value_cache)

        # qkv shape: [B, S, N, D]
        else:
            bs, seq_len, _ = x.shape
            # [B, S, N, D] --> [B, N, S, D]
            query = query.transpose(0, 2, 1, 3)
            key = key.transpose(0, 2, 1, 3)
            value = value.transpose(0, 2, 1, 3)
            if freqs_cis is not None:
                query, key = self.apply_rotary_emb(query, key, freqs_cis)
            if self.use_flash_attention:
                if os.getenv('RUN_MODE') == 'predict':
                    raise NotImplementedError(
                        "Conflict detected in predict mode: "
                        "Flash Attention is incompatible when use_past=False")
                context_layer = self.flash_attention(query, key, value, attn_mask)
            else:
                # [B, N_kv, S, D] --> [B, N, S, D]
                if self.use_gqa:
                    key = mint.repeat_interleave(key, repeats=self.repeat_num, axis=1)
                    value = mint.repeat_interleave(value, repeats=self.repeat_num, axis=1)
                context_layer = self.core_attention(query, key, value, attn_mask)
            # [B, N, S, D] --> [B, S, H]
            context_layer = context_layer.transpose(0, 2, 1, 3).reshape(
                bs, seq_len, self.hidden_size_per_partition)

        # apply output projection
        output = self.wo(context_layer)
        output = self.cast(output, x.dtype)

        return output

    def _cat_prefix(self, key, value, prefix_keys_values):
        """
        concat prefix_keys_values to key and value
        prefix_keys_values: shape(2, bs, pre_len, num_heads * kv_channels)
        """
        if prefix_keys_values is not None:
            past_key = prefix_keys_values[0]
            past_value = prefix_keys_values[1]
            past_key = self.cast(past_key, key.dtype)
            past_value = self.cast(past_value, value.dtype)
            key = ops.concat((past_key, key), 1)
            value = ops.concat((past_value, value), 1)
        return key, value

    def _check_gqa_valid(self):
        """check whether the config is valid for grouped-query-attention"""
        if self.num_heads % self.kv_num_heads != 0:
            raise ValueError(
                f"num_heads must be divisible by kv_num_heads, "
                f"but got num_heads {self.num_heads} and kv_num_heads {self.kv_num_heads}"
            )
        if self.kv_num_heads % self.tp_group_size != 0:
            raise ValueError(
                f"kv_num_heads must be divisible by tp_group_size, "
                f"but got kv_num_heads {self.kv_num_heads} and kv_num_heads {self.tp_group_size}"
            )

    def _init_self_attn(self):
        """init qkv linears of self-attention"""
        self.hidden_size_per_partition = divide(self.hidden_size, self.tp_group_size)
        self.kv_hidden_size_per_partition = divide(self.kv_hidden_size, self.tp_group_size)
        if self.qkv_concat:
            self.w_qkv = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size + 2 * self.kv_hidden_size,
                config=self.config.parallel_config,
                bias=self.config.qkv_has_bias,
                gather_output=False,
                transpose_b=True,
                param_init_type=self.config.param_init_dtype,
                compute_dtype=self.config.compute_dtype,
                tp_group=self.tp,
            )
        else:
            self.wq = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                config=self.config.parallel_config,
                bias=self.config.qkv_has_bias,
                gather_output=False,
                transpose_b=True,
                param_init_type=self.config.param_init_dtype,
                compute_dtype=self.config.compute_dtype,
                tp_group=self.tp,
            )
            self.wk = ColumnParallelLinear(
                self.hidden_size,
                self.kv_hidden_size,
                config=self.config.parallel_config,
                bias=self.config.qkv_has_bias,
                gather_output=False,
                transpose_b=True,
                param_init_type=self.config.param_init_dtype,
                compute_dtype=self.config.compute_dtype,
                tp_group=self.tp,
            )
            self.wv = ColumnParallelLinear(
                self.hidden_size,
                self.kv_hidden_size,
                config=self.config.parallel_config,
                bias=self.config.qkv_has_bias,
                gather_output=False,
                transpose_b=True,
                param_init_type=self.config.param_init_dtype,
                compute_dtype=self.config.compute_dtype,
                tp_group=self.tp,
            )

    def _init_cross_attn(self):
        """init qkv linears of cross-attention"""
        if self.hidden_size != self.kv_hidden_size:
            raise ValueError("hidden_size must be equal to kv_hidden_size!")
        self.wq = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            config=self.config.parallel_config,
            bias=self.config.qkv_has_bias,
            gather_output=False,
            transpose_b=True,
            param_init_type=self.config.param_init_dtype,
            compute_dtype=self.config.compute_dtype,
        )
        if self.qkv_concat:
            self.w_kv = ColumnParallelLinear(
                self.hidden_size,
                2 * self.kv_hidden_size,
                config=self.config.parallel_config,
                bias=self.config.qkv_has_bias,
                gather_output=False,
                transpose_b=True,
                param_init_type=self.config.param_init_dtype,
                compute_dtype=self.config.compute_dtype,
            )
        else:
            self.wk = ColumnParallelLinear(
                self.hidden_size,
                self.kv_hidden_size,
                config=self.config.parallel_config,
                bias=self.config.qkv_has_bias,
                gather_output=False,
                transpose_b=True,
                param_init_type=self.config.param_init_dtype,
                compute_dtype=self.config.compute_dtype,
            )
            self.wv = ColumnParallelLinear(
                self.hidden_size,
                self.kv_hidden_size,
                config=self.config.parallel_config,
                bias=self.config.qkv_has_bias,
                gather_output=False,
                transpose_b=True,
                param_init_type=self.config.param_init_dtype,
                compute_dtype=self.config.compute_dtype,
            )


class ParallelTransformerLayer(nn.Cell):
    r"""
    Single parallel transformer layer.

    Args:
        config (dict): Configuration.
        layer_index (int): Number which indicates the index of this transformer layer in the
            whole transformer block.
        model_comm_pgs (ModelCommProcessGroups, optional): Model communication process group.
            Default: default_model_comm_pgs.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(B, S, H)`.
        - **attention_mask** (Tensor) - Tensor of attention mask.
        - **rotary_pos_emb** (Tensor) - Tensor of rotary position embedding. Default: None.

    Outputs:
        - **output** (Tensor) - Tensor of shape :math:`(B, S, H)`.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(
            self,
            config,
            layer_number: int,
            layer_type=None,
            self_attn_mask_type=None,
            drop_path_rate: float = 0.0,
            model_comm_pgs=default_model_comm_pgs,
    ):
        super().__init__(config)
        if layer_type:
            raise NotImplementedError("For ParallelTransformerLayer, only decoder only structure is supported for now.")
        if self_attn_mask_type:
            raise NotImplementedError("For ParallelTransformerLayer, `self_attn_mask_type` is not supported for now.")
        if drop_path_rate > 0.0:
            raise NotImplementedError(
                "For ParallelTransformerLayer, `drop_path_rate > 0` is not supported for now, "
                "but got `drop_path_rate={}`".format(drop_path_rate)
            )
        self.config = config
        self.apply_residual_connection_post_norm = self.config.apply_residual_connection_post_norm
        # Normalize the input data.
        self.attention_norm = RMSNorm(dim=config.hidden_size,
                                      eps=config.layernorm_epsilon,
                                      compute_type=config.layernorm_compute_dtype)
        # Attention.
        self.attention = ParallelAttention(config, layer_number, model_comm_pgs=model_comm_pgs)
        # Normalize the attention output
        self.ffn_norm = RMSNorm(dim=config.hidden_size,
                                eps=config.layernorm_epsilon,
                                compute_type=config.layernorm_compute_dtype)
        # MLP
        self.feed_forward = ParallelMLP(config, model_comm_pgs=model_comm_pgs)

    def construct(self, x, freqs_cis=None, mask=None, batch_valid_length=None, block_tables=None,
                  slot_mapping=None, prefix_keys_values=None, q_seq_lens=None, key_cache=None, value_cache=None):
        """Construct function of transformer layer."""
        # hidden states: [B, S, H]
        # norm at the beginning of the transformer layer.
        norm_output = self.attention_norm(x)
        # attention.
        attention_output = self.attention(norm_output, batch_valid_length, block_tables, slot_mapping, freqs_cis,
                                          mask, prefix_keys_values=prefix_keys_values,
                                          q_seq_lens=q_seq_lens, key_cache=key_cache, value_cache=value_cache)
        # residual-connection.
        if self.apply_residual_connection_post_norm:
            residual = norm_output
        else:
            residual = x
        norm_input = ops.add(residual, attention_output)
        # layernorm post attention.
        norm_output = self.ffn_norm(norm_input)
        # MLP.
        mlp_output = self.feed_forward(norm_output)
        # residual-connection.
        if self.apply_residual_connection_post_norm:
            residual = norm_output
        else:
            residual = norm_input
        output = ops.add(residual, mlp_output)
        return output


class ParallelTransformer(nn.Cell):
    r"""
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`ParallelTransformerLayer`]
    Args:
        config: the config of transformer
        model_comm_pgs (ModelCommProcessGroups, optional): Model communication process group.
            Default: default_model_comm_pgs.

    Returns:
            output: Tensor, the output of transformerlayer
    """

    def __init__(
            self,
            config,
            model_type=None,
            layer_type=None,
            self_attn_mask_type=None,
            post_norm: bool = True,
            pre_process=False,
            post_process=False,
            drop_path_rate: float = 0.0,
            model_comm_pgs=default_model_comm_pgs,
    ):
        super().__init__(config)
        if model_type:
            raise NotImplementedError("For ParallelTransformer, 'model_type' is not support for now.")
        if layer_type:
            raise NotImplementedError("For ParallelTransformer, 'layer_type' is not support for now.")
        if self_attn_mask_type:
            raise NotImplementedError("For ParallelTransformer, 'self_attn_mask_type' is not support for now.")
        if pre_process:
            raise NotImplementedError("For ParallelTransformer, 'pre_process' is not support for now.")
        if post_process:
            raise NotImplementedError("For ParallelTransformer, 'post_process' is not support for now.")
        if drop_path_rate:
            raise NotImplementedError("For ParallelTransformer, 'drop_path_rate' is not support for now.")
        self.config = config
        self.post_norm = post_norm
        self.head_dim = config.hidden_size // config.num_heads
        self.num_layers = config.num_layers
        self.use_past = config.use_past
        self.is_first_iteration = True
        self.use_flash_attention = config.use_flash_attention
        self.compute_dtype = config.compute_dtype

        self.cast = ops.Cast()
        self.shape = ops.Shape()

        self.freqs_mgr = FreqsMgr(head_dim=self.head_dim,
                                  seq_length=config.seq_length,
                                  max_position_embedding=config.max_position_embedding,
                                  rotary_dtype=config.rotary_dtype,
                                  theta=config.theta,
                                  scaling_factor=config.scaling_factor,
                                  extend_method=config.extend_method,
                                  parallel_config=config.parallel_config,
                                  is_dynamic=config.is_dynamic)
        self.casual_mask = LowerTriangularMaskWithDynamic(seq_length=config.seq_length,
                                                          compute_type=config.compute_dtype,
                                                          is_dynamic=config.is_dynamic,
                                                          pad_token_id=config.pad_token_id,
                                                          use_flash_attention=config.use_flash_attention,
                                                          use_attn_mask_compression=config.use_attn_mask_compression,
                                                          use_past=config.use_past)

        self.tp = model_comm_pgs.tp
        self.tp_group_size = self.tp.size
        if config.parallel_config.vocab_emb_dp or self.tp_group_size == 1:
            self.tok_embeddings = VocabEmbedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                param_init_type=config.param_init_dtype,
                param_init="normal",
            )
        else:
            self.tok_embeddings = VocabParallelEmbedding(num_embeddings=config.vocab_size,
                                                         embedding_dim=config.hidden_size,
                                                         parallel_config=config.parallel_config,
                                                         init_method="normal",
                                                         init_type=config.param_init_dtype,
                                                         tp_group=self.tp)

        self.layers = nn.CellList()
        for layer_id in range(config.num_layers):
            layer = ParallelTransformerLayer(
                config=self.config,
                layer_number=layer_id + 1,
                model_comm_pgs=model_comm_pgs
            )
            self.layers.append(layer)

        if self.post_norm:
            # final layernorm before output.
            self.norm_out = RMSNorm(dim=config.hidden_size,
                                    eps=config.layernorm_epsilon,
                                    compute_type=config.layernorm_compute_dtype)

    def construct(self, tokens: Tensor, batch_valid_length=None, batch_index=None, zactivate_len=None,
                  block_tables=None, slot_mapping=None, prefix_keys_values=None, position_ids=None, attention_mask=None,
                  q_seq_lens=None, key_cache=None, value_cache=None):
        """
        Forward of ParallelTransformer.

        Args:
            tokens: the tokenized inputs with datatype int32
            batch_valid_length(Tensor): the past calculated the index with datatype int32, used for incremental
                prediction. Tensor of shape :math:`(batch_size,)`. Default None.
            block_tables (Tensor[int64]): Store mapping tables for each sequence.
            slot_mapping (Tensor[int32]): Store token cache physical slot index.
        Returns:
            output: Tensor, the output of ParallelTransformer
        """
        # preprocess
        mask = attention_mask
        if self.use_past:
            if self.is_first_iteration:
                freqs_cis = self.freqs_mgr.prefill()

                if prefix_keys_values is not None:
                    bs, seq_len = self.shape(tokens)
                    if mask is None:
                        mask = self.casual_mask(tokens)
                    prefix_length = prefix_keys_values[0].shape[2]
                    prefix_mask = Tensor(np.zeros((bs, 1, seq_len, prefix_length)), dtype=mask.dtype)
                    mask = self.concat((prefix_mask, mask))
            else:
                freqs_cis = self.freqs_mgr.chunk_with_decode(position_ids)
        else:
            bs, seq_len = self.shape(tokens)
            mask = self.casual_mask(tokens)
            freqs_cis = self.freqs_mgr(seq_len)
            if prefix_keys_values is not None:
                prefix_length = prefix_keys_values[0].shape[2]
                prefix_mask = Tensor(np.zeros((bs, 1, seq_len, prefix_length)), dtype=mask.dtype)
                mask = self.concat((prefix_mask, mask))

        # tokens shape: [bs, seq / 1]
        hidden_states = self.cast(self.tok_embeddings(tokens), self.compute_dtype)
        # hidden states shape: [bs, seq / 1, hidden_dim]
        for i in range(self.num_layers):
            prefix_kv = prefix_keys_values[i] if prefix_keys_values is not None else None
            key_cache_i = key_cache[i] if key_cache is not None else None
            value_cache_i = value_cache[i] if value_cache is not None else None
            hidden_states = self.layers[i](hidden_states, freqs_cis, mask, batch_valid_length=batch_valid_length,
                                           block_tables=block_tables, slot_mapping=slot_mapping,
                                           prefix_keys_values=prefix_kv, q_seq_lens=q_seq_lens,
                                           key_cache=key_cache_i, value_cache=value_cache_i)

        if self.post_norm:
            hidden_states = self.norm_out(hidden_states)
        return hidden_states
