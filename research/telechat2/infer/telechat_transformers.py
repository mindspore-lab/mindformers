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
import os
import numpy as np

import mindspore.common.dtype as mstype
from mindspore import nn, ops, mint, Tensor

from mindformers.parallel_core.inference.utils import divide
from mindformers.parallel_core.inference.tensor_parallel.mappings import reduce_from_model_parallel_region
from mindformers.parallel_core.inference.parallel_state import get_tensor_model_parallel_group
from mindformers.parallel_core.process_group_config import default_model_comm_pgs
from mindformers.modules.layers import FreqsMgrDynamicNTK
from mindformers.tools.logger import logger
from research.telechat2.infer.moe import ParallelMoE, RoutedParallelMLP
from research.telechat2.infer.layers import ColumnParallelLinear, RowParallelLinear
from research.telechat2.infer.transformer import \
    ParallelAttention, ParallelTransformerLayer, ParallelTransformer, ParallelMLP


class TelechatParallelMoE(ParallelMoE):
    r"""
        TelechatParallelMoE. Routing each tokens to the topk expert and calculating the final output.

        Args:
            ffn (Cell): The FeedForward Module.
            hidden_size (int): The hidden size of each token.
            moe_config (MoEConfig): The configuration of MoE (Mixture of Expert).
            use_fused_op (Bool): Whether use fused kernels.
        Inputs:
            - **input_tensor** (Tensor) - should be `[batch, seq_length, hidden_size].

        Outputs:
            - **output_tensor** (Tensor) - should be `[batch, seq_length, hidden_size].
    """

    def __init__(self,
                 ffn,
                 hidden_size,
                 moe_config,
                 use_fused_op=True):
        super(TelechatParallelMoE, self).__init__(
            ffn=ffn,
            hidden_size=hidden_size,
            moe_config=moe_config,
            use_fused_op=use_fused_op
        )
        self.tp_group = get_tensor_model_parallel_group()
        self.tp_size = self.tp_group.size

    def construct(self, input_tensor):
        """forward process"""
        input_tensor_shape = self.shape(input_tensor)  # (B, S, H)
        input_dtype = input_tensor.dtype
        input_tensor = self.reshape(input_tensor, (-1, self.hidden_size))  # (bs, seq/1, h) -> (bs*seq, h) : use N replace bs*seq

        if self.use_fused_op:
            expert_val, expert_index, row_index = self.gating_topk_softmax(input_tensor)
            sorted_input_tensor, group_list, unsort_map = \
                self.tensor_sort_by_fused_op(input_tensor, expert_index, row_index)
        else:
            gating_logits = self.gating(self.cast(input_tensor, self.router_dense_type)) # (N, h) * (h, E) -> (bs*seq, E)
            routing_weights = self.softmax(self.cast(gating_logits, mstype.float32)) # (N, E) -> (N, E)
            expert_val, expert_index = mint.topk(routing_weights, self.num_experts_chosen)
            sorted_input_tensor, group_list, unsort_map = self.tensor_sort(input_tensor, expert_index)

        if self.moe_config.norm_topk_prob and self.num_experts_chosen > 1:
            expert_val = self.cast(expert_val, mstype.float32)
            expert_weight = self.div(expert_val, self.add(mint.sum(expert_val, -1, True), 1e-9))
        else:
            expert_weight = self.mul(self.moe_config.routed_scaling_factor, expert_val)
        expert_weight = self.cast(expert_weight, input_dtype)

        # moeffn
        group_list[1:] = group_list[1:] - group_list[:-1]
        expert_output = self.ffn(sorted_input_tensor, group_list)  # (N, h) (N, k) -> (N, k, h)

        expert_index = self.cast(expert_index, mstype.int32)
        w2_bias = self.cast(mint.div(self.ffn.w2.bias, self.tp_size), input_dtype)
        moe_output = self.tensor_moe_finalize_routing(expert_output, expert_weight, expert_index, unsort_map, w2_bias)  # -> (N, h)
        moe_output = reduce_from_model_parallel_region(moe_output, self.tp_group)
        output_tensor = self.reshape(moe_output, input_tensor_shape)  # (N, h) -> (bs, seq, h)
        return output_tensor


class TelechatRoutedParallelMLP(RoutedParallelMLP):
    r"""
        TelechatRoutedParallelMLP. Routing each tokens to the topk expert and calculating the final output.

        Args:
            config (Config): The configuration of Model.
        Inputs:
            - **input_tensor** (Tensor) - should be `[batch, seq_length, hidden_size].

        Outputs:
            - **output_tensor** (Tensor) - should be `[batch, seq_length, hidden_size].
    """

    def __init__(self, config):
        super().__init__(config)
        self.w2 = RowParallelLinear(
            self.ffn_hidden_size,
            self.hidden_size,
            input_is_parallel=True,
            config=self.config.parallel_config,
            bias=True,
            skip_bias_add=True,
            transpose_b=True,
            param_init_type=self.config.param_init_dtype,
            compute_dtype=self.config.compute_dtype,
            is_expert=True,
            expert_num=self.config.moe_config.expert_num,
            tp_group=self.tp_group,
        )


class TelechatParallelMLP(ParallelMLP):
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

    def __init__(self, config, is_expert=False, model_comm_pgs=default_model_comm_pgs):
        super().__init__(config, is_expert, model_comm_pgs=model_comm_pgs)
        # Project back to h.
        self.w2 = RowParallelLinear(
            self.ffn_hidden_size,
            self.hidden_size,
            input_is_parallel=True,
            config=self.config.parallel_config,
            bias=True,
            transpose_b=True,
            is_expert=is_expert,
            param_init_type=self.config.param_init_dtype,
            compute_dtype=self.config.compute_dtype,
            tp_group=self.tp
        )

    def construct(self, x):
        """ Construct function of mlp block. """
        # [B, S, H] -> [B, S, ffn_H]
        if self.mlp_has_gate:
            if self.ffn_concat:
                gate_hidden_out = self.w_gate_hidden(x)  # dp,1 -> dp, mp  # dp,1 -> dp, mp
                gate, hidden = mint.split(gate_hidden_out,
                                          (self.ffn_hidden_size_per_partition, self.ffn_hidden_size_per_partition), -1)
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


class TelechatParallelAttention(ParallelAttention):
    r"""
    Parallel attention block.

    Args:
        layer_index (int): Number which indicates the index of this transformer layer in the
            whole transformer block.
        config (dict): Configuration.
        attn_type (str): Attention type. Support ['self_attn', 'cross_attn']. Default: 'self_attn'.

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

    def construct(self, x, batch_valid_length, block_tables, slot_mapping, freqs_cis=None,
                  attn_mask=None, alibi_mask=None, encoder_output=None, prefix_keys_values=None,
                  q_seq_lens=None, key_cache=None, value_cache=None):
        """Construct function of attention block."""
        # hidden_states shape: [B, S, H]
        bs, seq_len, _ = x.shape
        # apply query, key, value projection
        if self.attn_type == "self_attn":
            if self.sequence_parallel:
                seq_len = seq_len * self.tp_group_size
            # [B, S, H] --> [B, S, H + 2 * kv_H]
            if self.qkv_concat:
                qkv = self.cast(self.w_qkv(x), self.compute_dtype)
                query, key, value = mint.split(qkv,
                                               (self.hidden_size_per_partition,
                                                self.kv_hidden_size_per_partition,
                                                self.kv_hidden_size_per_partition), -1)
            else:
                query = self.cast(self.wq(x), self.compute_dtype)
                key_value = self.cast(self.wk_v(x), self.compute_dtype)
                key_value = key_value.reshape(-1, self.kv_num_heads_per_partition, self.head_dim * 2)
                key, value = mint.split(key_value, (self.head_dim, self.head_dim), -1)
                key = key.reshape(bs, seq_len, -1)
                value = value.reshape(bs, seq_len, -1)
        else:
            query = self.cast(self.wq(x), self.compute_dtype)
            if self.qkv_concat:
                kv = self.cast(self.w_kv(encoder_output), self.compute_dtype)
                key, value = mint.split(kv, (self.kv_hidden_size_per_partition, self.kv_hidden_size_per_partition), -1)
            else:
                key = self.cast(self.wk(encoder_output), self.compute_dtype)
                value = self.cast(self.wv(encoder_output), self.compute_dtype)

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
                    query = self.reshape(query, (-1, self.num_heads_per_partition * self.head_dim))
                    key = self.reshape(key, (-1, self.kv_num_heads_per_partition * self.head_dim))
                    value = self.reshape(value, (-1, self.kv_num_heads_per_partition * self.head_dim))
                    context_layer = self.flash_attention(query, key, value, attn_mask, alibi_mask, None, None,
                                                         q_seq_lens, batch_valid_length)
                else:
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
        else:
            query = self.reshape(query, (bs, seq_len, self.num_heads_per_partition, self.head_dim))
            key = self.reshape(key, (bs, seq_len, self.kv_num_heads_per_partition, self.head_dim))
            value = self.reshape(value, (bs, seq_len, self.kv_num_heads_per_partition, self.head_dim))
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

    def _init_self_attn(self):
        """init qkv linears of self-attention"""
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
            self.hidden_size_per_partition = divide(self.hidden_size, self.tp_group_size)
            self.kv_hidden_size_per_partition = divide(self.kv_hidden_size, self.tp_group_size)
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

            self.wk_v = ColumnParallelLinear(
                self.hidden_size,
                self.kv_hidden_size * 2,
                config=self.config.parallel_config,
                bias=self.config.qkv_has_bias,
                gather_output=False,
                transpose_b=True,
                param_init_type=self.config.param_init_dtype,
                compute_dtype=self.config.compute_dtype,
                tp_group=self.tp,
            )
            self.kv_hidden_size_per_partition = divide(self.kv_hidden_size, self.tp_group_size)


class TelechatParallelTransformerLayer(ParallelTransformerLayer):
    r"""
    Single parallel transformer layer.

    Args:
        config (dict): Configuration.
        layer_index (int): Number which indicates the index of this transformer layer in the
            whole transformer block.

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
        super().__init__(
            config,
            layer_number,
            layer_type,
            self_attn_mask_type,
            drop_path_rate,
            model_comm_pgs
        )
        # Attention.
        self.attention = TelechatParallelAttention(config, layer_number)
        # MLP
        self.expert_num = 1 if config.moe_config is None else config.moe_config.expert_num
        self.use_moe_infer = config.use_past and self.expert_num > 1
        if self.use_moe_infer:
            config.moe_config.router_dense_type = config.router_dense_type
            self.feed_forward = TelechatParallelMoE(
                ffn=TelechatRoutedParallelMLP(config),
                hidden_size=config.hidden_size,
                moe_config=config.moe_config,
            )
        else:
            self.feed_forward = TelechatParallelMLP(config, model_comm_pgs=model_comm_pgs)


class TelechatParallelTransformer(ParallelTransformer):
    r"""
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`ParallelTransformerLayer`]
    Args:
        config: the config of transformer

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
        super().__init__(
            config,
            model_type,
            layer_type,
            self_attn_mask_type,
            post_norm,
            pre_process,
            post_process,
            drop_path_rate,
            model_comm_pgs,
        )

        self.enable_dynamic_ntk = False
        if config.extend_method == 'DYNAMIC_NTK':
            self.enable_dynamic_ntk = True
            base_seqlen = config.base_seqlen \
                if hasattr(config, "base_seqlen") and config.base_seqlen else None
            self.freqs_mgr = FreqsMgrDynamicNTK(head_dim=self.head_dim,
                                                max_position_embedding=config.max_position_embedding,
                                                base_seqlen=base_seqlen,
                                                rotary_dtype=config.rotary_dtype,
                                                theta=config.theta,
                                                parallel_config=config.parallel_config,
                                                is_dynamic=config.is_dynamic)
            logger.info("Running with dynamic NTK.")

        self.layers = nn.CellList()
        for layer_id in range(config.num_layers):
            layer = TelechatParallelTransformerLayer(
                config=self.config,
                layer_number=layer_id + 1,
                model_comm_pgs=model_comm_pgs
                )
            self.layers.append(layer)

    # pylint: disable=W0613
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
        bs, seq_len = self.shape(tokens)
        mask = attention_mask
        if self.use_past:
            if self.is_first_iteration:
                if self.enable_dynamic_ntk:
                    freqs_cis = self.freqs_mgr.prefill(bs, batch_valid_length.max())
                else:
                    freqs_cis = self.freqs_mgr.prefill(bs, seq_len)

                if prefix_keys_values is not None:
                    if mask is None:
                        mask = self.casual_mask(tokens)
                    prefix_length = prefix_keys_values[0].shape[2]
                    prefix_mask = Tensor(np.zeros((bs, 1, seq_len, prefix_length)), dtype=mask.dtype)
                    mask = self.concat((prefix_mask, mask))
            else:
                freqs_cis = self.freqs_mgr.increment(batch_valid_length)
        else:
            mask = self.casual_mask(tokens)
            freqs_cis = self.freqs_mgr(seq_len)
            if prefix_keys_values is not None:
                prefix_length = prefix_keys_values[0].shape[2]
                prefix_mask = Tensor(np.zeros((bs, 1, seq_len, prefix_length)), dtype=mask.dtype)
                mask = self.concat((prefix_mask, mask))

        # tokens shape: [bs, seq/1]
        hidden_states = self.cast(self.tok_embeddings(tokens), self.compute_dtype)
        # h shape: [bs, seq/1, hidden_dim]
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
