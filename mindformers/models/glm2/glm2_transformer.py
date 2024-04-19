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
"""ChatGLM2 Transformer."""
import math

import mindspore.ops.functional as F
import mindspore.ops.operations as P
from mindspore import Tensor, nn, ops
from mindspore import dtype as mstype

try:
    from mindspore.nn.layer.flash_attention import FlashAttention

    FLASHATTENTION_IMPORT_VALID = True
except ImportError:
    FLASHATTENTION_IMPORT_VALID = False
try:
    from mindspore.ops.operations.nn_ops import PromptFlashAttention

    PROMPTFLASHATTENTION_VALID = True
except ImportError:
    PROMPTFLASHATTENTION_VALID = False

from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode

from mindformers.modules import LayerNorm, KVCacheMgr
from mindformers.modules.layers import Linear
from mindformers.pet.tuners.ptuning2_adapter import Ptuning2Adapter
from mindformers.version_control import get_dropout, check_valid_flash_attention, choose_flash_attention_dtype, \
    check_valid_paged_attention

from .glm2_config import ChatGLM2Config
from .glm2_modules import ChatGLM2MLP, ChatGLM2RMSNorm


class CoreAttention(nn.Cell):
    """ChatGLM2 core attention."""

    def __init__(self, config: ChatGLM2Config, layer_number):
        super(CoreAttention, self).__init__()
        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)

        projection_size = config.kv_channels * config.num_attention_heads

        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        self.mul_mask = P.Mul()
        self.add = P.Add()

        # Strided linear layer.
        self.attention_dropout = get_dropout(config.attention_dropout)

        parallel_config = config.parallel_config

        self.batch_matmul_q_k = P.BatchMatMul(transpose_b=True)
        self.batch_matmul_q_k.shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
             (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
        self.batch_matmul = P.BatchMatMul().shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
             (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
        self.softmax = nn.Softmax(axis=-1)

        self.merger_head_transpose = P.Transpose().shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),))

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        if config.is_dynamic:
            self.reshape.add_prim_attr("skip_redistribution", True)
        self.expand_dim = P.ExpandDims()
        self.use_prompt_flash_attention = config.use_prompt_flash_attention
        self.use_past = config.use_past

        self.compute_dtype = config.compute_dtype

    def construct(self, query_layer, key_layer, value_layer, attention_mask):
        """
        calculate attention function
        """
        # query_layer [b, heads, seq, hidden_size_per_head]
        # key_layer [b, heads, seq, hidden_size_per_head]
        # value_layer # [bs, heads, seq_len, hidden_size_per_head]

        # seqlen, batch, head, hidden_size

        if self.apply_query_key_layer_scaling:
            query_layer = query_layer / self.norm_factor

        # ===================================
        # Raw attention scores. [b, heads, s, s]
        # ===================================
        # [b, heads, seq_q, hidden_size_per_head] × [b, heads, seq_k, hidden_size_per_head]^T -> [b, heads, seq_q, seq_k]
        matmul_result = self.batch_matmul_q_k(query_layer, key_layer)

        # record original score dtype
        attention_scores_dtype = matmul_result.dtype
        # [b, heads, seq, seq]
        attention_scores = matmul_result

        if attention_mask is not None:
            if self.use_prompt_flash_attention and self.use_past:
                attention_mask = F.cast(attention_mask, mstype.float16)
                attention_mask = self.expand_dim(attention_mask, 1)
                attention_mask = self.mul_mask(attention_mask, -10000)
            attention_scores = self.add(attention_scores, attention_mask)

        if self.attention_softmax_in_fp32:
            attention_scores = F.cast(attention_scores, mstype.float32)

        attention_probs = self.softmax(attention_scores)
        attention_probs = F.cast(attention_probs, attention_scores_dtype)

        attention_probs = self.attention_dropout(attention_probs)

        # [bs, heads, seq_q, seq_k] x [bs, heads, seq_v, hidden_size_per_head] -> [b, heads, seq_q, hidden_size_per_head]
        context_layer = self.batch_matmul(attention_probs, value_layer)
        context_layer = F.cast(context_layer, self.compute_dtype)

        context_layer = self._merge_heads(context_layer)

        return context_layer

    def _merge_heads(self, x):
        """
        convert a 4d input to a 2d output

        Inputs:
            x: input tensor

        Output:
            x_merge: the 2d output
        """
        x = self.merger_head_transpose(x, (0, 2, 1, 3))  # bs, seq_length, head, size_per_head
        bs, seq_len, n_head, head_dim = self.shape(x)
        new_shape = (bs, seq_len, n_head * head_dim)
        x_merge = self.reshape(x, new_shape)
        return x_merge


class ChatGLM2SelfAttention(nn.Cell):
    """ChatGLM2 self-attention."""

    def __init__(self, config: ChatGLM2Config, layer_number):
        super(ChatGLM2SelfAttention, self).__init__()
        self.layer_number = max(1, layer_number)
        self.projection_size = config.kv_channels * config.num_attention_heads
        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads
        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        self.num_attention_heads_per_partition = config.num_attention_heads
        self.params_dtype = config.param_init_type
        self.compute_dtype = config.compute_dtype
        self.batch_size = config.batch_size
        self.seq_length = config.seq_length
        self.pre_seq_len = config.pre_seq_len

        self.multi_query_attention = config.multi_query_attention
        self.qkv_hidden_size = 3 * self.projection_size

        if self.multi_query_attention:
            self.num_multi_query_groups_per_partition = config.multi_query_group_num
            self.qkv_hidden_size = \
                (self.projection_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num)
            self.tile_kv = P.Tile()
            self.n_rep = self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition

        parallel_config = config.parallel_config
        self.query_key_value = Linear(config.hidden_size, self.qkv_hidden_size,
                                      has_bias=config.add_bias_linear or config.add_qkv_bias,
                                      param_init_type=self.params_dtype, compute_dtype=self.compute_dtype,
                                      skip_redistribution=config.is_dynamic)

        self.core_attention = CoreAttention(config, self.layer_number)

        self.dense = Linear(self.projection_size, config.hidden_size, has_bias=config.add_bias_linear,
                            param_init_type=self.params_dtype, compute_dtype=self.compute_dtype,
                            skip_redistribution=config.is_dynamic)

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        if config.is_dynamic:
            self.reshape.add_prim_attr("skip_redistribution", True)
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.use_paged_attention = config.use_paged_attention and check_valid_paged_attention()
        self.slice = P.StridedSlice()
        self.print = P.Print()
        self.kv_channels = config.kv_channels

        self.stack = P.Stack(axis=-1)
        self.gather = P.Gather()
        self.index_0 = Tensor(0)
        self.index_1 = Tensor(1)
        self.mul = P.Mul()
        self.sub = P.Sub()
        self.add = P.Add()
        self.concat = P.Concat(axis=-1)
        self.split_3 = P.Split(axis=-1, output_num=3)
        self.transpose = P.Transpose()
        self.merger_head_transpose = P.Transpose()
        self.cast = P.Cast()

        self.use_past = config.use_past
        if self.use_past:
            self.is_first_iteration = True
            kv_num_partition = config.num_attention_heads
            if config.multi_query_attention:
                kv_num_partition = config.multi_query_group_num

            if self.use_paged_attention:
                from mindformers.modules import PagedAttentionMgr
                self.paged_attention_mgr = PagedAttentionMgr(n_heads=config.num_attention_heads,
                                                             head_dim=self.head_dim, hidden_size=config.hidden_size,
                                                             n_kv_heads=kv_num_partition, block_size=config.block_size,
                                                             num_blocks=config.num_blocks,
                                                             compute_dtype=config.compute_dtype)
                self.paged_attention_mgr.shard(parallel_config)
            else:
                max_seq_length = config.seq_length if not self.pre_seq_len else config.seq_length + self.pre_seq_len
                self.kvcache_mgr = KVCacheMgr(kv_num_partition, self.head_dim,
                                              max_batch_size=config.batch_size, max_seq_length=max_seq_length,
                                              compute_dtype=config.compute_dtype, is_dynamic=config.is_dynamic,
                                              use_kvcache_op=config.use_kvcache_op,
                                              is_flexible_shape=config.is_flexible_shape)
                self.kvcache_mgr.shard(parallel_config)

        self.use_flash_attention = config.use_flash_attention
        self.use_prompt_flash_attention = config.use_prompt_flash_attention
        self.flash_attention, self.prompt_flash_attention = self.init_flash_attention_func(config)

        dp, mp = config.parallel_config.data_parallel, config.parallel_config.model_parallel
        if _get_parallel_mode() not in (ParallelMode.AUTO_PARALLEL,):
            if config.prefix_name.startswith("glm32k"):
                mp = 1
            self.merger_head_transpose.shard(((dp, mp, 1, 1),))
            self.query_key_value.shard(strategy_matmul=((dp, 1), (mp, 1)), strategy_bias=((dp, mp), (mp,)))
            self.dense.shard(strategy_matmul=((dp, 1), (mp, 1)), strategy_bias=((dp, 1), (1,)))


    def init_flash_attention_func(self, config):
        """init the flash attention operator"""
        dp, mp = config.parallel_config.data_parallel, config.parallel_config.model_parallel
        fa_op, pfa_op = None, None

        if self.use_flash_attention:
            self.attention_mask_dtype = choose_flash_attention_dtype()
            fa_op = FlashAttention(head_dim=config.hidden_size // config.num_attention_heads,
                                   head_num=config.num_attention_heads,
                                   dropout_rate=config.attention_dropout,
                                   prev_block_num=65536, next_block_num=0,
                                   dp=dp, mp=mp, high_precision=True)

        if self.use_prompt_flash_attention:
            self.attention_mask_dtype = choose_flash_attention_dtype()
            pfa_op = PromptFlashAttention(num_heads=config.num_attention_heads,
                                          scale_value=1 / self.norm_factor,
                                          num_key_value_heads=0, input_layout='BNSD',
                                          pre_tokens=65536, next_tokens=0)

        return fa_op, pfa_op


    def _merge_heads(self, x):
        """
        convert a 4d input to a 2d output

        Inputs:
            x: input tensor

        Output:
            x_merge: the 2d output
        """
        x = self.merger_head_transpose(x, (0, 2, 1, 3))
        bs, seq_len, n_head, head_dim = self.shape(x)
        new_shape = (bs, seq_len, n_head * head_dim)
        x_merge = self.reshape(x, new_shape)
        return x_merge

    def _repeat_kv(self, x, num_repeat):
        if num_repeat == 1:
            return x
        bs, n_kv_head, seqlen, head_dim = self.shape(x)
        x = self.reshape(x, (bs, n_kv_head, 1, seqlen * head_dim))
        x = self.tile_kv(x, (1, 1, num_repeat, 1))
        x = self.reshape(x, (bs, n_kv_head * num_repeat, seqlen, head_dim))
        return x

    def apply_rotary_pos_emb(self, x: Tensor, rope_cache: Tensor) -> Tensor:
        """apply rotary position embedding to q,k."""
        # x: [b, heads, seq, hidden_size_per_head]
        bs, num_heads, seq_len, head_dim = self.shape(x)
        # rope_cache: first (seq_len, kv_channels//4, 2), other (1, kv_channels//4, 2)
        rot_dim = self.kv_channels // 2
        # rot_dim = rope_cache.shape[-2] * 2
        x1 = self.slice(x, (0, 0, 0, 0), (bs, num_heads, seq_len, rot_dim), (1, 1, 1, 1))
        x_pass = self.slice(x, (0, 0, 0, rot_dim), (bs, num_heads, seq_len, head_dim), (1, 1, 1, 1))
        # ms not support variable sizes
        # truncate to support variable sizes
        # [bs, nh, sq, kv_channels//4, 2]
        xshaped = self.reshape(x1, (bs, num_heads, seq_len, rot_dim // 2, 2))
        _, _, _, kv_shape, _ = self.shape(xshaped)
        # [bs, 1, sq, kv_channels//4, 2]
        rope_cache = self.reshape(rope_cache, (-1, 1, seq_len, kv_shape, 2))
        xshaped_0, xshaped_1 = ops.split(xshaped, 1, -1)
        rope_cache_0, rope_cache_1 = ops.split(rope_cache, 1, -1)
        x_out1 = self.sub(self.mul(xshaped_0, rope_cache_0), self.mul(xshaped_1, rope_cache_1))
        x_out2 = self.add(self.mul(xshaped_1, rope_cache_0), self.mul(xshaped_0, rope_cache_1))
        x_out = self.stack((x_out1, x_out2))
        bs_x, num_heads_x, seq_len_x, _, _, _ = self.shape(x_out)
        x_out = self.reshape(x_out, (bs_x, num_heads_x, seq_len_x, -1))
        # [bs, sq, nh, hidden_size_per_head]
        return self.concat((x_out, x_pass))

    def add_prefix_if_need(self, prefix_key_value, key_layer, value_layer, attention_mask):
        """
        add p-tuning v2 prefix if need
        """
        if not isinstance(self.pre_seq_len, int) or self.pre_seq_len <= 0:
            return key_layer, value_layer, attention_mask

        seq_len = key_layer.shape[2]

        key_layer, value_layer = Ptuning2Adapter.add_prefix(
            prefix_key_value,
            key_layer,
            value_layer
        )

        if attention_mask is not None and getattr(self, "is_first_iteration", True):
            batch_size = attention_mask.shape[0]
            prefix_mask = attention_mask.new_zeros((batch_size, 1, seq_len, self.pre_seq_len))
            m_cat = P.Concat(3)
            # [bs, 1, seq_len, pre_seq_len + seq_len]
            attention_mask = m_cat((prefix_mask, attention_mask))

        return key_layer, value_layer, attention_mask

    def construct(self, hidden_states, attention_mask, rotary_pos_emb, kvcache_inputs=None, prefix_key_value=None):
        """Forward process of self-attention."""
        # hidden_states: [bs, seq_len, hidden_size]
        # attention_mask (bs, 1, seq_len, seq_len)
        # rotary_pos_emb: first: (sen length, kv_channels//4, 2)， after:(1, kv_channels//4, 2]

        # [bs, seq_len, qkv_hidden_size]
        mixed_raw_layer = self.query_key_value(hidden_states)

        # not compatible with ms below 2.0
        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_raw_layer.split(
                [self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                 self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                 self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                 ],
                axis=-1,
            )
            # [bs,seq_len,nh*hidden_size_per_attention_head]
            bs, seq_len, _ = self.shape(query_layer)
            if self.use_past and not self.is_first_iteration:
                query_layer = self.reshape(query_layer, (
                    bs, self.num_attention_heads_per_partition, 1, self.hidden_size_per_attention_head))
                key_layer = self.reshape(key_layer, (
                    bs, self.num_multi_query_groups_per_partition, 1, self.hidden_size_per_attention_head))
                value_layer = self.reshape(value_layer, (
                    bs, self.num_multi_query_groups_per_partition, 1, self.hidden_size_per_attention_head))
            else:
                query_layer = self.reshape(query_layer, (bs, seq_len, self.num_attention_heads_per_partition,
                                                         self.hidden_size_per_attention_head))
                key_layer = self.reshape(key_layer, (bs, seq_len, self.num_multi_query_groups_per_partition,
                                                     self.hidden_size_per_attention_head))
                value_layer = self.reshape(value_layer, (bs, seq_len, self.num_multi_query_groups_per_partition,
                                                         self.hidden_size_per_attention_head))
                # [bs, nh, seq_len, hidden_size_per_attention_head]
                query_layer = self.transpose(query_layer, (0, 2, 1, 3))
                # [bs, multi_query_groups, seq_len, hidden_size_per_attention_head]
                key_layer = self.transpose(key_layer, (0, 2, 1, 3))
                # [bs, multi_query_groups, seq_len, hidden_size_per_attention_head]
                value_layer = self.transpose(value_layer, (0, 2, 1, 3))
        else:
            # [b, seq, (heads * 3 * hidden_size_per_head)] --> [b, seq, heads, 3 * hidden_size_per_head]
            bs, seq_len, _ = self.shape(mixed_raw_layer)
            mixed_raw_layer = self.reshape(mixed_raw_layer, (bs, seq_len, self.num_attention_heads_per_partition,
                                                             3 * self.hidden_size_per_attention_head))
            # [b, seq, heads, hidden_size_per_head]
            (query_layer, key_layer, value_layer) = self.split_3(mixed_raw_layer)
            # [b, seq, heads, hidden_size_per_head] -> [bs, num_heads, seq_len, hidden_size_per_head]
            query_layer = self.transpose(query_layer, (0, 2, 1, 3))
            key_layer = self.transpose(key_layer, (0, 2, 1, 3))
            value_layer = self.transpose(value_layer, (0, 2, 1, 3))

        # rotary_pos_emb: first: (seq_length, kv_channels//4, 2)， after:(1, kv_channels//4, 2)
        if rotary_pos_emb is not None:
            # [b, heads, seq, hidden_size_per_head]
            query_layer = self.apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            # [bs, multi_query_groups, seq_len, hidden_size_per_attention_head]
            key_layer = self.apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        key_layer, value_layer, attention_mask = self.add_prefix_if_need(
            prefix_key_value,
            key_layer,
            value_layer,
            attention_mask
        )

        # key and value for current token(s)
        # [bs, heads, seq_len, hidden_size_per_head]
        if self.use_past:
            if self.use_paged_attention:
                _, _, slot_mapping = kvcache_inputs
                key_out = self.paged_attention_mgr(key_layer, value_layer, slot_mapping)
                query_layer = ops.depend(query_layer, key_out)
            else:
                key_layer, value_layer = self.kvcache_mgr(key_layer, value_layer, kvcache_inputs)

        # tile k,v to num_heads
        if self.multi_query_attention:
            key_layer = self._repeat_kv(key_layer, self.n_rep)
            value_layer = self._repeat_kv(value_layer, self.n_rep)

        context_layer = \
            self.compute_flash_attention_func(query_layer, key_layer, value_layer, attention_mask, kvcache_inputs)

        # Output. [bs, seq_len, hidden_size]
        output = self.dense(context_layer)

        return output

    def compute_flash_attention_func(self, query_layer, key_layer, value_layer, attention_mask, kvcache_inputs):
        """compute context_layer_score with or without flash attention"""
        if not self.training:
            if self.use_prompt_flash_attention and \
                    ((self.use_past and self.is_first_iteration) or (not self.use_past)):
                attention_mask = attention_mask.to(self.attention_mask_dtype)
                context_layer = self.prompt_flash_attention(query_layer, key_layer, value_layer, attention_mask,
                                                            None, None, None, None, None, None, None, None)[0]
                context_layer = self._merge_heads(context_layer)
            elif self.use_paged_attention and (self.use_past and not self.is_first_iteration):
                batch_valid_length, block_tables, _ = kvcache_inputs
                context_layer = self.paged_attention_mgr.paged_attn(query_layer, batch_valid_length, block_tables)
            else:
                context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)

        else:
            if self.use_flash_attention:
                attention_mask = attention_mask.to(self.attention_mask_dtype)
                context_layer = self.flash_attention(query_layer, key_layer, value_layer, attention_mask)
                context_layer = self._merge_heads(context_layer)
            else:
                context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)
        return context_layer


class ChatGLM2Block(nn.Cell):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config: ChatGLM2Config, layer_number: int):
        super(ChatGLM2Block, self).__init__()
        self.layer_number = layer_number
        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.fp32_residual_connection = config.fp32_residual_connection
        self.use_past = config.use_past
        self.params_dtype = config.param_init_type
        self.layernorm_dtype = config.layernorm_compute_type
        self.compute_dtype = config.compute_dtype
        self.seq_length = config.seq_length
        self.use_seq_parallel = config.parallel_config.use_seq_parallel
        self.add = P.Add()

        layer_norm_func = ChatGLM2RMSNorm if config.rmsnorm else LayerNorm
        # Layernorm on the input data.
        self.input_layernorm = layer_norm_func(config.hidden_size, eps=config.layernorm_epsilon,
                                               param_init_type=self.layernorm_dtype)

        self.input_layernorm.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

        # Self attention.
        self.self_attention = ChatGLM2SelfAttention(config, layer_number)
        self.hidden_dropout = config.hidden_dropout

        # Layernorm on the attention output
        self.post_attention_layernorm = layer_norm_func(config.hidden_size, eps=config.layernorm_epsilon,
                                                        param_init_type=self.layernorm_dtype)

        # MLP
        self.mlp = ChatGLM2MLP(config)

        self.dropout = get_dropout(self.hidden_dropout)

        self.cast = P.Cast()

        dp = config.parallel_config.data_parallel
        if _get_parallel_mode() not in (ParallelMode.AUTO_PARALLEL,):
            self.mlp.shard(config.parallel_config)
            self.input_layernorm.shard(((dp, 1, 1),))
            self.post_attention_layernorm.shard(((dp, 1, 1),))
            self.add.shard(((dp, 1, 1), (dp, 1, 1)))
            self.dropout.dropout.shard(((dp, 1, 1),))

    def set_select_recompute(self):
        self.input_layernorm.recompute(False)
        self.post_attention_layernorm.recompute(False)
        self.self_attention.recompute()
        self.mlp.recompute()
        self.dropout.dropout.recompute(False)
        self.cast.recompute(False)

    def construct(self, hidden_states, attention_mask, rotary_pos_emb, kvcache_inputs=None, prefix_key_value=None):
        """Forward process of the transformer layer."""
        # hidden_states: [bs, seq_len, hidden_size]
        # attention_mask first: (bs, 1, seq_len, seq_len), after: (bs, 1, 1, seq_len)
        # rotary_pos_emb: first: (seq_len, kv_channels//4, 2)， after: (1, kv_channels//4, 2)

        # Layer norm at the beginning of the transformer layer.
        hidden_states = self.cast(hidden_states, self.layernorm_dtype)
        layernorm_output = self.input_layernorm(hidden_states)
        # fp32 -> fp16
        layernorm_output = self.cast(layernorm_output, self.compute_dtype)

        # Self attention.
        attention_output = self.self_attention(
            layernorm_output,
            attention_mask,
            rotary_pos_emb,
            kvcache_inputs,
            prefix_key_value
        )

        # Residual connection.
        # False on default.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = self.dropout(attention_output)
        layernorm_input = self.add(residual, layernorm_input)

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        layernorm_output = self.cast(layernorm_output, self.compute_dtype)

        # MLP.
        mlp_output = self.mlp(layernorm_output)

        # Second residual connection.
        # False on default.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = self.dropout(mlp_output)
        output = self.add(residual, output)

        return output


def set_parallel_configure_for_layer(layer, layer_id, offset, parallel_config, n_layers, no_recompute_layers=None):
    r"""
        Default setting for the pipeline is: `(layer_id + offset) // (layers / pipeline_stage)`.

        Args:
            layer(Cell) - Represents the transformer block
            layer_id(int) - Means the layer index for the current module, counts from zero.
            offset(int) - Means the layer_index needs a offset, if there are other modules in the net.
            parallel_config(dict) - Parallel Config
            n_layers(int) - The total layers used for the model.
    """
    pp_dis = max(int((n_layers + 1) / parallel_config.pipeline_stage), 1)
    if isinstance(offset, list):
        if len(offset) != parallel_config.pipeline_stage:
            raise ValueError(f"The length of `offset` {len(offset)} do not match "
                             f"`pipeline stage` {parallel_config.pipeline_stage}.")
        i = min(layer_id // pp_dis, parallel_config.pipeline_stage - 1)
        offset_layer = offset[i]
    elif isinstance(offset, int):
        offset_layer = offset
    else:
        raise TypeError(f"`offset` must be `int` of list of `int`, but got {type(offset)}.")

    pp_id = min((layer_id + offset_layer) // pp_dis, parallel_config.pipeline_stage - 1)
    layer.pipeline_stage = pp_id

    if not parallel_config.recompute.select_recompute:
        if isinstance(parallel_config.recompute, bool):
            if parallel_config.recompute:
                layer.recompute()
        else:
            if parallel_config.recompute.recompute:
                layer.recompute(recompute_slice_activation=parallel_config.recompute.recompute_slice_activation)
    else:
        if not no_recompute_layers:
            layer.set_select_recompute()
        elif layer_id not in no_recompute_layers:
            if parallel_config.recompute.recompute:
                layer.recompute()
            else:
                layer.recompute(recompute_slice_activation=parallel_config.recompute.recompute_slice_activation)


class ChatGLM2Transformer(nn.Cell):
    """Transformer class."""

    def __init__(self, config: ChatGLM2Config):
        super(ChatGLM2Transformer, self).__init__()

        self.post_layer_norm = config.post_layer_norm
        self.compute_dtype = config.compute_dtype

        # Number of layers.
        self.num_layers = config.num_layers

        self.pre_seq_len = config.pre_seq_len

        if config.use_flash_attention:
            config.use_flash_attention = check_valid_flash_attention(FLASHATTENTION_IMPORT_VALID, 'FlashAttention')

        if config.use_prompt_flash_attention:
            config.use_prompt_flash_attention = check_valid_flash_attention(PROMPTFLASHATTENTION_VALID,
                                                                            "PromptFlashAttention")

        # Transformer layers.
        def build_layer(layer_number):
            return ChatGLM2Block(config, layer_number)

        self.layers = nn.CellList()
        for i in range(self.num_layers):
            layer = build_layer(i + 1)

            set_parallel_configure_for_layer(layer, layer_id=i, offset=0, n_layers=self.num_layers,
                                             parallel_config=config.parallel_config,
                                             no_recompute_layers=config.no_recompute_layers)

            self.layers.append(layer)

        if self.post_layer_norm:
            layer_norm_func = ChatGLM2RMSNorm if config.rmsnorm else LayerNorm
            # Final layer norm before output.
            self.final_layernorm = layer_norm_func(config.hidden_size, eps=config.layernorm_epsilon,
                                                   param_init_type=config.layernorm_compute_type)
            if config.parallel_config.pipeline_stage > 1:
                self.final_layernorm.pipeline_stage = config.parallel_config.pipeline_stage - 1
                self.final_layernorm.set_comm_fusion(2)
            else:
                self.final_layernorm.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

            self.final_layernorm.shard(((config.parallel_config.data_parallel, 1, 1),))

    def construct(self,
                  hidden_states,
                  attention_mask,
                  rotary_pos_emb,
                  kvcache_inputs=None,
                  prefix_key_values=None):
        """Forward process of the transformer."""
        # hidden_states (bs, seq_len, hs)
        # attention_mask (bs, 1, seq_len, seq_len)
        # rotary_pos_emb: first: (sen length, kv_channels//2, 2)， after:[1, kv_channels // 2, 2]

        for i in range(self.num_layers):
            prefix_key_value = None
            if prefix_key_values is not None:
                prefix_key_value = prefix_key_values[i]
            layer = self.layers[i]

            hidden_states = layer(
                hidden_states,
                attention_mask,
                rotary_pos_emb,
                kvcache_inputs,
                prefix_key_value=prefix_key_value
            )

        # Final layer norm.
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)
            hidden_states = self.cast(hidden_states, self.compute_dtype)

        return hidden_states
