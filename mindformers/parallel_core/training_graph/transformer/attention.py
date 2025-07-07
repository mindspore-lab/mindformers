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
"""Transformer Attention"""
import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import Union
from mindspore.ops import auto_generate as aclnn_ops
from mindspore.ops import functional as F
from mindspore.parallel.shard import Layout
from mindspore import nn
import mindspore.common.dtype as mstype

from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.training_graph.transformer.enums import AttnMaskType
from mindformers.parallel_core.training_graph.base_models.common.embeddings.rope_utils import ApplyRotaryPosEmb


@dataclass
class SelfAttentionSubmodules:
    """
    Configuration class for specifying the submodules of a self-attention.
    """

    linear_qkv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None


@dataclass
class CrossAttentionSubmodules:
    """
    Configuration class for specifying the submodules of a cross-attention.
    """

    linear_q: Union[ModuleSpec, type] = None
    linear_kv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None


class Attention(nn.Cell):
    """Attention layer abstract class.

    This layer only contains common modules required for the "self attn" and
    "cross attn" specializations.

    Args:
        config (TransformerConfig): The config of the transformer model.
        submodules (Union[SelfAttentionSubmodules, CrossAttentionSubmodules]): The submodules used to construct
            the Attention layer, such as ColumnParallelLinear and RowParallelLinear for query and
            key-value projections.
        layer_number (int): Number which indicates the index of this transformer layer in the
            whole transformer block.
        attn_mask_type (str): attention mask type. Default: None.
        attention_type (str): Attention type. Support ['self', 'cross']. Default: "self".
        cp_comm_type (str): Communication type for context parallelism. Default: None.

    Inputs:
        - **hidden_states** (Tensor) - Tensor of shape :math:`(B, S, H)`.
        - **attention_mask** (Tensor) - Tensor of attention mask.
        - **key_value_states** (Tensor) - Tensor of encoder output used for cross attention. Default: None.
        - **rotary_pos_emb** (Tensor) - Tensor of rotary position embedding. Default: None.

    Outputs:
        - **output** (Tensor) - Tensor of shape :math:`(B, S, H)`.
        - **bias** (Tensor) - Tensor of shape :math:`(B, S, H)`.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(
            self,
            config: TransformerConfig,
            submodules: Union[SelfAttentionSubmodules, CrossAttentionSubmodules],
            layer_number: int,
            attn_mask_type: AttnMaskType = None,
            attention_type: str = "self",
            cp_comm_type: str = None,
    ):
        super().__init__()
        if attn_mask_type:
            # The repository uses the attention_mask passed to GPTModel with default causal implementation.
            # For implementation details, refer to the training_graph.transformer.mask_generate.py code.
            # Note: Specifying attn_mask_type is currently not supported in this implementation.
            raise NotImplementedError("For Attention, 'attn_mask_type' is not supported for now.")
        if cp_comm_type is not None:
            raise NotImplementedError("cp_comm_type is not supported for now.")

        self.config = config
        self.attention_type = attention_type
        self.init_method = config.init_method
        self.output_layer_init_method = config.output_layer_init_method
        self.compute_dtype = self.config.compute_dtype
        self.hidden_size = self.config.hidden_size
        self.use_flash_attention = self.config.use_flash_attention
        self.parallel_config = self.config
        self.use_attn_mask_compression = self.config.use_attn_mask_compression
        self.input_layout = config.input_layout

        # For normal attention without groups, num_query_groups == num_attention_heads,
        # so these two will be the same
        self.use_gqa = self.config.num_query_groups < self.config.num_attention_heads
        self.num_heads = self.config.num_attention_heads
        self.kv_num_heads = self.config.num_query_groups if self.use_gqa else self.num_heads
        self.head_dim = self.config.kv_channels
        self.q_hidden_size = self.head_dim * self.num_heads
        self.kv_hidden_size = self.head_dim * self.kv_num_heads

        # Not Support Graph Mode and key/value with different hidden size for now.
        # attention_hidden_size and num_attention_heads must be evenly divisible
        # by num_heads and tp respectively to enable correct tensor splitting.

        self.dp = 1 if self.config.data_parallel_size is None else self.config.data_parallel_size
        self.tp = 1 if self.config.tensor_model_parallel_size is None else self.config.tensor_model_parallel_size
        self.cp = 1 if self.config.context_parallel_size is None else self.config.context_parallel_size

        self.n_rep = self.num_heads // self.kv_num_heads
        self.layer_number = max(1, layer_number)
        self.norm_factor = math.sqrt(self.head_dim)
        self.compute_2d = (config.sequence_parallel and self.cp == 1)
        self.seq_length = config.seq_length
        self.pre_tokens = 2147483647 if self.config.attention_pre_tokens is None else self.config.attention_pre_tokens
        self.next_tokens = 0 if self.config.attention_next_tokens is None else self.config.attention_next_tokens
        self.keep_prob = 1.0 if self.config.attention_dropout is None else 1 - self.config.attention_dropout
        self.use_attention_mask = True if self.config.use_attention_mask is None else self.config.use_attention_mask

        # Define ulysses context parallel related parameters
        self.cp_ds = 1
        self.cp_co = self.cp // self.cp_ds

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_size' must be a multiple "
                             "of 'num_heads', but got the hidden_size is {} and the num_heads is {}."
                             .format(self.hidden_size, self.num_heads))
        # Check if num_heads and kv_num_heads are multiples of tp * cp_ds
        if self.num_heads % (self.tp * self.cp_ds) != 0:
            raise ValueError("For 'ParallelAttention', the class variable 'num_heads' must be a multiple of "
                             "'tensor_parallel * ulysses_cp_num', but got num_heads is {}, tensor_parallel is {}, "
                             "ulysses_cp_num is {}."
                             .format(self.num_heads, self.tp, self.cp_ds))
        if self.kv_num_heads % (self.tp * self.cp_ds) != 0 and self.kv_num_heads % self.tp != 0:
            raise ValueError("For 'ParallelAttention', the class variable 'kv_num_heads' must be a multiple of "
                             "'tensor_parallel * ulysses_cp_num', but got kv_num_heads is {}, tensor_parallel is {}, "
                             "ulysses_cp_num is {}."
                             .format(self.kv_num_heads, self.tp, self.cp_ds))

        self.core_attention = build_module(
            submodules.core_attention,
            config=self.config,
            layer_number=self.layer_number,
            # attn_mask_type/cp_comm_type useless for this api.
            # attention_type/softmax_scale, this parameter for the corresponding Megatron module
            # is functionally irrelevant, so it can be omitted here.
            # Other similar invocations should follow this same interpretation.
        )

        # Output
        self.linear_proj = build_module(
            submodules.linear_proj,
            input_size=self.q_hidden_size,
            output_size=self.config.hidden_size,
            config=self.config,
            init_method=self.output_layer_init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            # The input_is_parallel/is_expert/tp_comm_buffer_name parameter is unnecessary.
            # tp/ep partitioning and communication of module parameters is implemented by MindSpore's shard mechanism,
            # requiring no awareness from upper layers.
            # Other similar invocations should follow this same interpretation.
        )

        self.apply_rotary_pos_emb = ApplyRotaryPosEmb(self.parallel_config)
        # after rotary
        # If ulysses context parallel is enabled, initialize related operations
        if self.cp_ds > 1:
            self._ulysses_initial()

        self.split_qkv = aclnn_ops.SplitWithSize().add_prim_attr("skip_redistribution", True)
        self.shape = aclnn_ops.Shape()
        self.cast = aclnn_ops.Cast()
        self.reshape = aclnn_ops.Reshape()
        self.merge_head_transpose = aclnn_ops.Transpose()
        self.tile_kv = aclnn_ops.Tile()
        self.cat = aclnn_ops.Concat(2)
        self.shard(self.config)

    def _ulysses_initial(self):
        """Initialize ulysses related operations."""
        self.transpose_back = aclnn_ops.Transpose()
        self.transpose_ulysses = aclnn_ops.Transpose()
        self.transpose_a2a = aclnn_ops.Transpose()
        self.transpose_ulysses_merger_a2a = aclnn_ops.Transpose()
        self.transpose_ulysses_merger = aclnn_ops.Transpose()

        dp = self.dp
        tp = self.tp
        cp = self.cp

        self.linear_proj.matmul.shard(in_strategy=((dp * cp, tp), (1, tp)), out_strategy=((dp * cp * tp, 1),))
        layout = Layout((dp, cp, tp), ("dp", "cp", "tp"))
        layout_transpose_back = (layout("dp", "tp", "cp", "None"),)
        self.transpose_back.shard(in_strategy=layout_transpose_back)
        self.transpose_ulysses.shard(((dp, cp, tp, 1, 1, 1),))
        self.transpose_a2a.shard(((dp, self.cp_co, self.cp_ds, tp, 1, 1),))
        self.transpose_ulysses_merger_a2a.shard(((dp, self.cp_co, self.cp_ds, tp, 1, 1),))
        self.transpose_ulysses_merger.shard(((dp, cp, 1, tp, 1, 1),))

    @abstractmethod
    def get_query_key_value_tensors(self, hidden_states, key_value_states):
        """
        This method needs to be implemented based on whether the derived class
        is "self-attn" or "cross-attn".
        """

    def construct(
            self,
            hidden_states,
            attention_mask,
            key_value_states=None,
            rotary_pos_emb=None,
            prefix_keys_values=None,
            actual_seq_len=None
    ):
        """ Construct function of attention block."""
        ori_dtype = hidden_states.dtype
        if self.compute_2d:
            bs_seq, _ = self.shape(hidden_states)
            seq_len = self.seq_length
            bs = bs_seq // seq_len
        else:
            seq_len, bs, _ = self.shape(hidden_states)

        # apply query, key, value projection
        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)

        # Training-only implementation (no inference logic included).

        # transpose and reshape
        query = self.reshape(query, (seq_len, bs, self.num_heads, self.head_dim))
        key = self.reshape(key, (seq_len, bs, self.kv_num_heads, self.head_dim))

        # apply rotary position embedding
        if rotary_pos_emb is not None:
            query = self.apply_rotary_pos_emb(query, rotary_pos_emb)
            key = self.apply_rotary_pos_emb(key, rotary_pos_emb)

        # with ulysses context parallel, insert all to all before FA
        if self.input_layout == "TND":
            query = query.reshape(query, (-1, self.num_heads, self.head_dim))
            key = key.reshape(key, (-1, self.kv_num_heads, self.head_dim))
            value = value.reshape(value, (-1, self.kv_num_heads, self.head_dim))
        elif self.cp > 1 and self.cp_ds > 1:
            # For query & key, transpose from [B, N, S, D] back to [B, S, N, D]
            query = self.transpose_back(query, (0, 2, 1, 3))
            query = self._ulysses_q_a2a(query)
            key = self.transpose_back(key, (0, 2, 1, 3))
            key = self._ulysses_kv_a2a(key)
            # Value is [B, S, N, D], no need to transpose back
            value = self.reshape(value, (bs, seq_len, self.kv_num_heads, self.head_dim))
            value = self._ulysses_kv_a2a(value)
        elif self.cp > 1:
            # Merge heads for query and key
            query = self._merge_heads(query)
            key = self._merge_heads(key)
        else:
            value = self.reshape(value, (seq_len, bs, self.kv_num_heads, self.head_dim))
            key, value = self._cat_prefix(key, value, prefix_keys_values)

        if not self.use_flash_attention:
            key = self._repeat_kv(key, self.n_rep)
            value = self._repeat_kv(value, self.n_rep)
            context_layer = self.core_attention(query, key, value, attention_mask)
        else:
            if attention_mask is not None:
                attention_mask = self.cast(attention_mask, mstype.uint8)

            if query.dtype not in (mstype.float16, mstype.bfloat16):
                query = self.cast(query, mstype.float16)
            if key.dtype not in (mstype.float16, mstype.bfloat16):
                key = self.cast(key, mstype.float16)
            if value.dtype not in (mstype.float16, mstype.bfloat16):
                value = self.cast(value, mstype.float16)

            output = self.core_attention(
                query, key, value, attention_mask,
                actual_seq_qlen=actual_seq_len, actual_seq_kvlen=actual_seq_len
            )

            # with ulysses context parallel, insert all to all after FA
            if self.input_layout == "TND":
                context_layer = self.reshape(output, (seq_len, bs, -1))
            elif self.cp > 1 and self.cp_ds > 1:
                output = self._ulysses_context_layer_a2a(output)
                context_layer = output
            elif self.cp > 1:
                # If context_parallel > 1 but cp_ds <= 1, no need for all_to_all, proceed without merging heads
                context_layer = output
            else:
                context_layer = self.cast(output, self.compute_dtype)

        # apply output projection
        output, bias = self.linear_proj(context_layer)
        output = self.cast(output, ori_dtype)

        return output, bias

    def _cat_prefix(self, key, value, prefix_keys_values):
        '''
        Concatenate prefix_keys_values to key and value.
        prefix_keys_values: shape (2, bs, pre_len, num_heads * kv_channels)
        '''
        if prefix_keys_values is not None:
            _, bs, n_kv_head, head_dim = key.shape
            past_key = prefix_keys_values[0]
            past_value = prefix_keys_values[1]
            past_key = self.reshape(past_key, (-1, bs, n_kv_head, head_dim))
            past_value = self.reshape(past_value, (-1, bs, n_kv_head, head_dim))
            past_key = self.cast(past_key, self.compute_dtype)
            past_value = self.cast(past_value, self.compute_dtype)
            key = self.cat((past_key, key))
            value = self.cat((past_value, value))
        return key, value

    def _merge_heads(self, x):
        """
        Convert a 4D input tensor to a 3D output tensor.

        Inputs:
            x: input tensor

        Output:
            x_merge: the 3D output tensor
        """
        x = self.merge_head_transpose(x, (0, 2, 1, 3))  # dp,tp,cp,1 -> dp,cp,tp,1
        bs, seq_len, n_head, head_dim = self.shape(x)

        if self.compute_2d:
            new_shape = (bs * seq_len, n_head * head_dim)
        else:
            new_shape = (bs, seq_len, n_head * head_dim)
        x_merge = self.reshape(x, new_shape)
        return x_merge

    def _repeat_kv(self, x, rep):
        if rep == 1:
            return x
        bs, n_kv_head, seqlen, head_dim = self.shape(x)
        x = self.reshape(x, (bs, n_kv_head, 1, seqlen * head_dim))
        x = self.tile_kv(x, (1, 1, rep, 1))
        x = self.reshape(x, (bs, n_kv_head * rep, seqlen, head_dim))
        return x

    def _ulysses_q_a2a(self, qkv):
        """
        Given a qkv tensor with shape (bs, seq_len, n_head, head_dim),
        insert all-to-all communication in the right place using transpose with specific shard strategy.
        Refers to <https://arxiv.org/abs/2309.14509>

        Args:
            qkv (Tensor): qkv after rotary embedding and before attention, with shape (B, S, N, D)

        Returns:
            Tensor: qkv tensor after all-to-all communication.
        """
        bs, seq_len, _, _ = F.shape(qkv)
        new_shape = (bs, seq_len, self.tp, self.cp_ds, -1, self.head_dim)
        # [bs, seq_len, n_head, head_dim] -> [bs, seq_len, n_head/cp_ds, cp_ds, head_dim]
        qkv = self.reshape(qkv, new_shape)
        # [bs, seq_len, n_head/cp_ds, cp_ds, head_dim] -> [bs, seq_len, cp_ds, n_head/cp_ds, head_dim]
        qkv = self.transpose_ulysses(qkv, (0, 1, 3, 2, 4, 5))
        # Insert all-to-all communication
        qkv = self.transpose_a2a(qkv, (0, 1, 2, 3, 4, 5))
        # Reshape to BSH, set -1 for H to accommodate different kv heads
        qkv = F.reshape(qkv, (bs, seq_len, -1))
        return qkv

    def _ulysses_kv_a2a(self, qkv):
        """
        Given a qkv tensor with shape (bs, seq_len, n_head, head_dim),
        insert all-to-all communication in the right place using transpose with specific shard strategy.
        Refers to <https://arxiv.org/abs/2309.14509>

        Args:
            qkv (Tensor): qkv after rotary embedding and before attention, with shape (B, S, N, D)

        Returns:
            Tensor: qkv tensor after all-to-all communication.
        """
        bs, seq_len, _, _ = F.shape(qkv)
        new_shape = (bs, seq_len, self.tp, self.cp_ds, -1, self.head_dim)
        # [bs, seq_len, n_head, head_dim] -> [bs, seq_len, n_head/cp_ds, cp_ds, head_dim]
        qkv = self.reshape(qkv, new_shape)
        # [bs, seq_len, n_head/cp_ds, cp_ds, head_dim] -> [bs, seq_len, cp_ds, n_head/cp_ds, head_dim]
        qkv = self.transpose_ulysses(qkv, (0, 1, 3, 2, 4, 5))
        # Insert all-to-all communication
        qkv = self.transpose_a2a(qkv, (0, 1, 2, 3, 4, 5))
        # Reshape to BSH, set -1 for H to accommodate different kv heads
        qkv = F.reshape(qkv, (bs, seq_len, -1))
        return qkv

    def _ulysses_context_layer_a2a(self, context_layer):
        """
        Given the context_layer tensor after attention, with shape (bs, seq_len, hidden_size),
        insert all-to-all communication in the right place using transpose with specific shard strategy.
        Refers to <https://arxiv.org/abs/2309.14509>

        Args:
            context_layer (Tensor): context layer after attention, with shape (B, S, H)

        Returns:
            Tensor: context layer tensor after all-to-all communication.
        """
        bs, seq_len, _ = F.shape(context_layer)
        new_shape = (bs, seq_len, self.cp_ds, self.tp, -1, self.head_dim)
        context_layer = F.reshape(context_layer, new_shape)
        # Insert all-to-all communication
        context_layer = self.transpose_ulysses_merger_a2a(context_layer, (0, 1, 2, 3, 4, 5))
        context_layer = self.transpose_ulysses_merger(context_layer, (0, 1, 3, 2, 4, 5))
        # Reshape back to BSH
        context_layer = F.reshape(context_layer, (bs, seq_len, self.hidden_size))
        return context_layer

    def shard(self, config: TransformerConfig):
        """Set sharding strategies."""
        dp = 1 if config is None else config.data_parallel_size
        tp = 1 if config is None else config.tensor_model_parallel_size
        cp = 1 if config is None else config.context_parallel_size

        self.merge_head_transpose.shard(((dp, tp, cp, 1),))
        self.tile_kv.shard(((dp, tp, 1, cp),))
        self.cat.shard(((dp, tp, 1, 1), (dp, tp, 1, 1)))

        if config.sequence_parallel and cp == 1:
            self.linear_proj.matmul.shard(in_strategy=((dp, tp), (1, tp)), out_strategy=((dp * tp, 1),))


class SelfAttention(Attention):
    """Self-attention layer class

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.

    Args:
        config (TransformerConfig): The config of the transformer model.
        submodules (SelfAttentionSubmodules): The submodules used to construct the SelfAttention layer,
            such as ColumnParallelLinear and RowParallelLinear for query and key-value projections.
        layer_number (int): Number which indicates the index of this transformer layer in the
            whole transformer block.
        attn_mask_type (str): attention mask type. Default None.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(
            self,
            config: TransformerConfig,
            submodules: SelfAttentionSubmodules,
            layer_number: int,
            attn_mask_type: AttnMaskType = None,
            cp_comm_type: str = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
        )

        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.hidden_size,
            self.q_hidden_size + 2 * self.kv_hidden_size,
            config=self.config,
            init_method=self.init_method,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            # The gather_output/is_expert/tp_comm_buffer_name parameter is unnecessary.
            # tp/ep partitioning and communication of module parameters is implemented by MindSpore's shard mechanism,
            # requiring no awareness from upper layers.
            # Other similar invocations should follow this same interpretation.
        )

        self.reshape_concat = aclnn_ops.Reshape()
        self.shard_self_attention(self.config)

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        if self.compute_2d:
            bs_seq, _ = self.shape(hidden_states)
            seq_len = self.seq_length
            bs = bs_seq // seq_len
        else:
            seq_len, bs, _ = self.shape(hidden_states)

        qkv, _ = self.linear_qkv(hidden_states)
        qkv = self.cast(qkv, self.compute_dtype)

        query, key, value = self.split_qkv(qkv,
                                           (self.head_dim * self.n_rep * self.kv_num_heads,
                                            self.head_dim * self.kv_num_heads,
                                            self.head_dim * self.kv_num_heads), -1)
        query = self.reshape_concat(query, (seq_len, bs, self.kv_num_heads, self.n_rep * self.head_dim))
        key = self.reshape_concat(key, (seq_len, bs, self.kv_num_heads, self.head_dim))
        value = self.reshape_concat(value, (seq_len, bs, self.kv_num_heads, self.head_dim))

        return query, key, value

    def shard_self_attention(self, config: TransformerConfig):
        """Set sharding strategies."""
        dp = 1 if config is None else config.data_parallel_size
        tp = 1 if config is None else config.tensor_model_parallel_size
        cp = 1 if config is None else config.context_parallel_size
        self.split_qkv.shard(((cp, dp, tp),))


class SelfAttentionMegatron(Attention):
    """Self-attention layer class

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.

    Args:
        config (TransformerConfig): The config of the transformer model.
        submodules (SelfAttentionSubmodules): The submodules used to construct the SelfAttentionMegatron layer,
            such as ColumnParallelLinear and RowParallelLinear for query and key-value projections.
        layer_number (int): Number which indicates the index of this transformer layer in the
            whole transformer block.
        attn_mask_type (str): attention mask type. Default None.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(
            self,
            config: TransformerConfig,
            submodules: SelfAttentionSubmodules,
            layer_number: int,
            attn_mask_type: AttnMaskType = None,
            cp_comm_type: str = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
        )

        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.hidden_size,
            self.q_hidden_size + 2 * self.kv_hidden_size,
            config=self.config,
            init_method=self.init_method,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            # The gather_output/is_expert/tp_comm_buffer_name parameter is unnecessary.
            # tp/ep partitioning and communication of module parameters is implemented by MindSpore's shard mechanism,
            # requiring no awareness from upper layers.
            # Other similar invocations should follow this same interpretation.
        )
        self.reshape_concat = aclnn_ops.Reshape()
        self.shard_self_attention(self.config)

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        if self.compute_2d:
            bs_seq, _ = self.shape(hidden_states)
            seq_len = self.seq_length
            bs = bs_seq // seq_len
        else:
            seq_len, bs, _ = self.shape(hidden_states)

        qkv, _ = self.linear_qkv(hidden_states)
        qkv = self.cast(qkv, self.compute_dtype)
        new_tensor_shape = (seq_len, bs, -1, (self.n_rep + 2) * self.head_dim)
        mixed_x_layer = self.reshape_concat(qkv, new_tensor_shape)
        query, key, value = self.split_qkv(mixed_x_layer,
                                           (self.head_dim * self.n_rep, self.head_dim, self.head_dim), 3)
        return query, key, value

    def shard_self_attention(self, config: TransformerConfig):
        """Set sharding strategies."""
        dp = 1 if config is None else config.data_parallel_size
        tp = 1 if config is None else config.tensor_model_parallel_size
        cp = 1 if config is None else config.context_parallel_size
        self.split_qkv.shard(((cp, dp, tp, 1),))


class CrossAttention(Attention):
    """Cross-attention layer class

    Cross-attention layer takes input with size [s, b, h] and context with size
    [s, b, h] and returns output of the same size.

    Args:
        config (TransformerConfig): The config of the transformer model.
        submodules (CrossAttentionSubmodules): The submodules used to construct the CrossAttention layer,
            such as ColumnParallelLinear and RowParallelLinear for query and key-value projections.
        layer_number (int): Number which indicates the index of this transformer layer in the
            whole transformer block.
        attn_mask_type (str): attention mask type. Default None.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(
            self,
            config: TransformerConfig,
            submodules: CrossAttentionSubmodules,
            layer_number: int,
            attn_mask_type: AttnMaskType = None,
            cp_comm_type: str = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="cross",
            cp_comm_type=cp_comm_type,
        )
        if self.use_gqa:
            raise NotImplementedError("Grouped query attention not implemented for cross-attention.")
        if self.hidden_size != self.kv_hidden_size:
            raise ValueError("self.hidden_size and self.kv_hidden_size must be equal in cross_attn!")

        self.linear_q = build_module(
            submodules.linear_q,
            self.hidden_size,
            self.hidden_size,
            config=self.config,
            bias=self.config.add_bias_linear,
            init_method=self.init_method,
            skip_bias_add=False,
            # The gather_output/is_expert parameter is unnecessary.
            # tp/ep partitioning and communication of module parameters is implemented by MindSpore's shard mechanism,
            # requiring no awareness from upper layers.
            # Other similar invocations should follow this same interpretation.
        )

        self.linear_kv = build_module(
            submodules.linear_kv,
            self.hidden_size,
            2 * self.kv_hidden_size,
            config=self.config,
            bias=self.config.add_bias_linear,
            init_method=self.init_method,
            skip_bias_add=False,
            # The gather_output/is_expert parameter is unnecessary.
            # tp/ep partitioning and communication of module parameters is implemented by MindSpore's shard mechanism,
            # requiring no awareness from upper layers.
            # Other similar invocations should follow this same interpretation.
        )

        self.split_kv = aclnn_ops.SplitWithSize().add_prim_attr("skip_redistribution", True)

    def get_query_key_value_tensors(self, hidden_states, key_value_states):
        """
        Derives `query` tensor from `hidden_states`, and `key`/`value` tensors
        from `key_value_states`.
        """
        query, _ = self.linear_q(hidden_states)
        query = self.cast(query, self.compute_dtype)
        kv, _ = self.linear_kv(key_value_states)
        kv = self.cast(kv, self.compute_dtype)
        key, value = self.split_kv(kv, (self.kv_hidden_size, self.kv_hidden_size), 2)

        return query, key, value

    def shard(self, config: TransformerConfig):
        """Set sharding strategies."""
        dp = 1 if config is None else config.data_parallel_size
        tp = 1 if config is None else config.tensor_model_parallel_size
        cp = 1 if config is None else config.context_parallel_size
        self.split_kv.shard(((dp, cp, tp),))
