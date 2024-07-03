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
import math

from mindspore import nn, Tensor
from mindspore import ops
from mindspore.ops import operations as P

from mindformers.experimental.distri_cores.create_comm import (
    get_tp_world_size,
)
from mindformers.experimental.distri_cores.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from mindformers.experimental.distri_cores.transformer.rotary_pos_embedding import (
    apply_rotary_pos_emb,
)
from mindformers.experimental.distri_cores.transformer.scale_mask_softmax import (
    ScaleMaskSoftmax,
)
from mindformers.experimental.distri_cores.transformer import get_norm,\
    get_attn_mask_func, get_act_func
from mindformers.experimental.distri_cores.utils import divide
from mindformers.experimental.distri_cores.random import get_rng_tracer
from .module import Module


__all__ = [
    "ParallelMLP",
    "ParallelAttention",
    "ParallelTransformerLayer",
    "ParallelTransformer",
]


class ParallelMLP(Module):
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
        super(ParallelMLP, self).__init__(config)
        if is_expert:
            raise NotImplementedError("For ParallelMLP, `is_expert=True` is not supported for now.")
        self.config = config
        self.has_bias = self.config.mlp_has_bias
        self.hidden_size = self.config.hidden_size
        self.ffn_hidden_size = self.config.ffn_hidden_size
        self.mlp_has_gate = self.config.mlp_has_gate
        if self.config.mlp_has_gate:
            self.gating = ColumnParallelLinear(
                self.hidden_size,
                self.ffn_hidden_size,
                config=self.config.parallel_config,
                bias=self.has_bias,
                transpose_b=False,
                gather_output=False,
                is_expert=is_expert,
                param_init_type=self.config.param_init_dtype,
                compute_dtype=self.config.compute_dtype,
            )
            self.mul = P.Mul()

        self.mapping = ColumnParallelLinear(
            self.hidden_size,
            self.ffn_hidden_size,
            config=self.config.parallel_config,
            bias=self.has_bias,
            transpose_b=False,
            gather_output=False,
            is_expert=is_expert,
            param_init_type=self.config.param_init_dtype,
            compute_dtype=self.config.compute_dtype,
        )

        self.bias_gelu_fusion = False

        self.act_type = self.config.hidden_act
        self.act_func = get_act_func(self.act_type)

        # Project back to h.
        self.projection = RowParallelLinear(
            self.ffn_hidden_size,
            self.hidden_size,
            input_is_parallel=True,
            config=self.config.parallel_config,
            bias=self.has_bias,
            transpose_b=False,
            is_expert=is_expert,
            param_init_type=self.config.param_init_dtype,
            compute_dtype=self.config.compute_dtype,
        )

    def construct(self, hidden_states):
        """ Construct function of mlp block. """
        # [B, S, H] -> [B, S, ffn_H]
        if self.config.mlp_has_gate:
            gate = self.gating(hidden_states)
            gate = self.act_func(gate)
            intermediate_parallel = self.mapping(hidden_states)
            intermediate_parallel = self.mul(intermediate_parallel, gate)
        else:
            intermediate_parallel = self.mapping(hidden_states)
            intermediate_parallel = self.act_func(intermediate_parallel)

        # [B, S, ffn_H] -> [B, S, H]
        output = self.projection(intermediate_parallel)
        return output


def _merge_heads(x):
    """ Merge attention heads. """
    # [B, N, S, D] -> [B, S, N, D]
    x = x.transpose(0, 2, 1, 3)
    bs, seq_len, num_heads, head_dim = x.shape
    # [B, S, N, D] -> [B, S ,H]
    merged_shape = (bs, seq_len, num_heads * head_dim)
    x_merged = x.reshape(merged_shape)
    return x_merged


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

        self.attention_dropout = nn.Dropout(p=self.config.attention_dropout_rate)

    def construct(self, query_layer, key_layer, value_layer, attention_mask):
        """construct."""
        # score: [B, N, S, S]
        score = ops.bmm(query_layer, key_layer.transpose(0, 1, 3, 2))
        score = score * self.inv_norm_factor

        # attention scores and attention mask [B, N, S_q, S_k]
        attention_probs = self.scale_mask_softmax(score, attention_mask)

        if not self.sequence_parallel:
            with get_rng_tracer().rng_fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # [B, N, S, S] * [B, N, S, D] -> [B, N, S, D]
        weighted_values = ops.bmm(attention_probs, value_layer)
        # [B, N, S, D] -> [B, S, N*D]
        attn_output = _merge_heads(weighted_values)

        return attn_output


class ParallelAttention(Module):
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
    def __init__(self, config, layer_number, attention_type='self_attn', attn_mask_type=None):
        super(ParallelAttention, self).__init__()
        if attn_mask_type:
            raise NotImplementedError("For ParallelAttention, `attn_mask_type` is not supported for now.")
        self.config = config
        self.layer_index = max(1, layer_number)
        self.param_init_dtype = self.config.param_init_dtype
        self.compute_dtype = self.config.compute_dtype

        self.attn_type = attention_type
        self.use_gqa = self.config.use_gqa
        self.num_heads = self.config.num_heads
        self.kv_num_heads = self.config.kv_num_heads if self.use_gqa else self.num_heads
        self.hidden_size = self.config.hidden_size
        self.head_dim = divide(self.hidden_size, self.num_heads)
        self.kv_hidden_size = self.head_dim * self.kv_num_heads

        self.sequence_parallel = self.config.parallel_config.use_sequence_parallel
        self.use_flash_attention = self.config.use_flash_attention
        self.norm_factor = math.sqrt(self.head_dim)

        tp_group_size = get_tp_world_size()
        self.tp_group_size = tp_group_size
        self.num_heads_per_partition = divide(self.num_heads, tp_group_size)

        if self.use_gqa:
            if self.kv_num_heads % tp_group_size != 0:
                raise NotImplementedError(
                    "Currently the kv_num_heads should be "
                    "a multiple of the tensor parallel size"
                )
            self.kv_num_heads_per_partition = divide(self.kv_num_heads, tp_group_size)
        else:
            self.kv_num_heads_per_partition = self.num_heads_per_partition

        if self.attn_type == 'self_attn':
            self.qkv_proj = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size + 2 * self.kv_hidden_size,
                config=self.config.parallel_config,
                bias=self.config.qkv_has_bias,
                gather_output=False,
                transpose_b=False,
                param_init_type=self.config.param_init_dtype,
                compute_dtype=self.config.compute_dtype,
            )
        elif self.attn_type == 'cross_attn':
            assert self.hidden_size == self.kv_hidden_size

            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                config=self.config.parallel_config,
                bias=self.config.qkv_has_bias,
                gather_output=False,
                transpose_b=False,
                param_init_type=self.config.param_init_dtype,
                compute_dtype=self.config.compute_dtype,
            )

            self.kv_proj = ColumnParallelLinear(
                self.hidden_size,
                2 * self.kv_hidden_size,
                config=self.config.parallel_config,
                bias=self.config.qkv_has_bias,
                gather_output=False,
                transpose_b=False,
                param_init_type=self.config.param_init_dtype,
                compute_dtype=self.config.compute_dtype,
            )
        else:
            raise NotImplementedError(f"attention_type should be self_attn or cross_attn, but got {self.attn_type}")

        self.core_attention = CoreAttention(self.layer_index, self.config)

        self.out_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            input_is_parallel=True,
            config=self.config.parallel_config,
            bias=self.config.out_proj_has_bias,
            transpose_b=False,
            param_init_type=self.config.param_init_dtype,
            compute_dtype=self.config.compute_dtype,
        )

        self.cast = ops.Cast()

    def construct(
            self,
            hidden_states,
            attention_mask,
            encoder_output=None,
            rotary_pos_emb=None,
        ):
        """ Construct function of attention block. """
        # hidden_states: [B, S, H]
        ori_dtype = hidden_states.dtype
        bs, seq_len, _ = hidden_states.shape
        # apply query, key, value projection
        if self.attn_type == 'self_attn':
            if self.sequence_parallel:
                seq_len = seq_len * self.tp_group_size
            # [B, S, H] --> [B, S, H + 2 * kv_H]
            qkv = self.cast(self.qkv_proj(hidden_states), self.compute_dtype)

            query = ops.strided_slice(qkv,
                                      (0, 0, 0),
                                      (bs, seq_len, self.num_heads_per_partition * self.head_dim),
                                      (1, 1, 1))
            key = ops.strided_slice(qkv,
                                    (0, 0, self.num_heads_per_partition * self.head_dim),
                                    (bs, seq_len, (self.num_heads_per_partition +
                                                   self.kv_num_heads_per_partition) * self.head_dim),
                                    (1, 1, 1))
            value = ops.strided_slice(qkv,
                                      (0, 0, (self.num_heads_per_partition +
                                              self.kv_num_heads_per_partition) * self.head_dim),
                                      (bs, seq_len, (self.num_heads_per_partition +
                                                     self.kv_num_heads_per_partition * 2) * self.head_dim),
                                      (1, 1, 1))

            # [B, S, H] -> [B, N, S, D]
            query = query.reshape(bs, seq_len, -1, self.head_dim).transpose((0, 2, 1, 3))
            # [B, S, H] -> [B, S, N, D]
            key = key.reshape(bs, seq_len, -1, self.head_dim)
            value = value.reshape(bs, seq_len, -1, self.head_dim)
        else:
            kv = self.cast(self.kv_proj(encoder_output), self.compute_dtype)

            # split tensor along last dimension.
            last_dim = kv.ndim - 1
            (key, value) = ops.Split(axis=last_dim, output_num=2)(kv)

            new_tensor_shape = kv.shape[:-1] + (
                self.num_heads_per_partition,
                self.head_dim,
            )

            key = key.view(*new_tensor_shape)
            value = value.view(*new_tensor_shape)

            q = self.cast(self.q_proj(hidden_states), self.compute_dtype)
            new_tensor_shape = q.shape[:-1] + \
                               (self.num_heads_per_partition,
                                self.head_dim)
            query = q.view(*new_tensor_shape)
            query = query.transpose((0, 2, 1, 3))

        if rotary_pos_emb is not None:
            if isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = rotary_pos_emb
            else:
                rotary_pos_emb = (rotary_pos_emb,) * 2

        # expand the key_layer and value_layer [B, S, kv_N_per_tp, D]
        # to [B, S, N_per_tp, D]
        if self.num_heads_per_partition // self.kv_num_heads_per_partition > 1:
            repeat_num = self.num_heads_per_partition - self.kv_num_heads_per_partition
            key = self._repeat_kv(key, repeat_num)
            value = self._repeat_kv(value, repeat_num)
        else:
            key = key.transpose((0, 2, 1, 3))
            value = value.transpose((0, 2, 1, 3))

        # apply rotary position embedding
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            query = apply_rotary_pos_emb(query, q_pos_emb)
            key = apply_rotary_pos_emb(key, k_pos_emb)

        if not self.use_flash_attention:
            context_layer = self.core_attention(query, key, value, attention_mask)
        else:
            raise NotImplementedError('use_flash_attention is not supported for now.')

        # apply output projection
        output = self.out_proj(context_layer)
        output = self.cast(output, ori_dtype)

        return output

    def _repeat_kv(self, x, rep):
        """ Expand key, value on num_head dimension. """
        if rep == 1:
            return x
        bs, seq_length, num_groups, head_dim = x.shape()
        # [B, S, ng, D] -> [B, ng, S, D]
        x = x.transpose((0, 2, 1, 3))
        # [B, ng, S, D] -> [B, ng, 1, S*D]
        x = x.reshape((bs, num_groups, 1, seq_length * head_dim))
        x = x.tile((1, 1, rep, 1))
        # [B, ng, rep, S*D] -> [B, N, S, D]
        x = x.reshape((bs, num_groups * rep, seq_length, head_dim))
        return x


class ParallelTransformerLayer(Module):
    r"""
    Single parallel transformer layer.

    Args:
        config (dict): Configuration.
        layer_index (int): Number which indicates the index of this transformer layer in the
            whole transformer block.

    Inputs:
        - **hidden_states** (Tensor) - Tensor of shape :math:`(B, S, H)`.
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
            layer_number,
            layer_type=None,
            self_attn_mask_type=None,
            drop_path_rate=0.0,
        ):
        super(ParallelTransformerLayer, self).__init__(config)
        if layer_type:
            raise NotImplementedError("For ParallelTransformerLayer, only decoder only structure is supported for now.")
        if self_attn_mask_type:
            raise NotImplementedError("For ParallelTransformerLayer, `self_attn_mask_type` is not supported for now.")
        if drop_path_rate > 0.0:
            raise NotImplementedError("For ParallelTransformerLayer, `drop_path_rate > 0` is not supported for now, "
                                      "but got `drop_path_rate={}`".format(drop_path_rate))
        self.config = config
        self.layer_index = layer_number

        self.apply_residual_connection_post_norm = (
            self.config.apply_residual_connection_post_norm
        )

        self.residual_connection_dtype = self.config.residual_connection_dtype

        # Normalize the input data.
        self.input_norm = get_norm(config)

        # Attention.
        self.attention = ParallelAttention(config, layer_number)

        # Normalize the attention output
        self.post_attention_norm = get_norm(config)

        # MLP
        self.mlp = ParallelMLP(config)

        self.hidden_states_dropout = nn.Dropout(p=self.config.hidden_dropout_rate)

    def construct(self, hidden_states, attention_mask, rotary_pos_emb=None):
        """ Construct function of transformer layer. """
        # hidden_states: [B, S, H]
        # layernorm at the beginning of the transformer layer.
        norm_output = self.input_norm(hidden_states)

        # attention.
        attention_output = self.attention(
            norm_output, attention_mask, rotary_pos_emb=rotary_pos_emb
        )

        # residual-connection.
        if self.apply_residual_connection_post_norm:
            residual = norm_output
        else:
            residual = hidden_states

        out = self.hidden_states_dropout(attention_output)
        norm_input = residual + out

        # layernorm post attention.
        norm_output = self.post_attention_norm(norm_input)

        # MLP.
        mlp_output = self.mlp(norm_output)

        # residual-connection.
        if self.apply_residual_connection_post_norm:
            residual = norm_output
        else:
            residual = norm_input

        out = self.hidden_states_dropout(mlp_output)
        output = residual + out

        return output


class ParallelTransformer(Module):
    r"""
    Parallel transformer class.

    Args:
        config (dict): Configuration.
        post_norm (bool): Insert normalization layer at the end of transformer block. Default: True.

    Inputs:
        - **hidden_states** (Tensor) - Tensor of shape :math:`(B, S, H)`.
        - **attention_mask** (Tensor) - Tensor of attention mask.
        - **rotary_pos_emb** (Tensor) - Tensor of rotary position embedding. Default: None.

    Outputs:
        - **hidden_states** (Tensor) - Tensor of shape :math:`(B, S, H)`.

    Supported Platforms:
        ``Ascend``
    """
    def __init__(
            self,
            config,
            model_type=None,
            layer_type=None,
            self_attn_mask_type=None,
            post_norm=True,
            pre_process=False,
            post_process=False,
            drop_path_rate=0.0
        ):
        super(ParallelTransformer, self).__init__(config)
        if model_type:
            raise NotImplementedError("For ParallelTransformer, `model_type` is not supported for now.")
        if layer_type:
            raise NotImplementedError("For ParallelTransformer, `layer_type` is not supported for now.")
        if self_attn_mask_type:
            raise NotImplementedError("For ParallelTransformer, `self_attn_mask_type` is not supported for now.")
        if pre_process:
            raise NotImplementedError("For ParallelTransformer, `pre_process=True` is not supported.")
        if post_process:
            raise NotImplementedError("For ParallelTransformer, `post_process=True` is not supported.")
        if drop_path_rate > 0.0:
            raise NotImplementedError("For ParallelTransformer, `drop_path_rate > 0` is not supported for now, "
                                      "but got `drop_path_rate={}`".format(drop_path_rate))
        self.config = config
        self.post_norm = post_norm
        # number of layers.
        self.num_layers = config.num_layers

        # transformer layers.
        def build_layer(layer_index):
            return ParallelTransformerLayer(config, layer_index)

        offset = 0

        self.layers = nn.SequentialCell(
            [build_layer(i + 1 + offset) for i in range(self.num_layers)]
        )

        if self.post_norm:
            # final layernorm before output.
            self.final_norm = get_norm(config)

    def _get_layer(self, layer_number):
        """ Get layer_number-th transformerlayer. """
        return self.layers[layer_number]

    def construct(self, hidden_states, attention_mask, rotary_pos_emb=None):
        """ Construct function of transformer. """
        for index in range(self.num_layers):
            layer = self._get_layer(index)
            hidden_states = layer(hidden_states, attention_mask, rotary_pos_emb)

        # final layernorm.
        if self.post_norm:
            hidden_states = self.final_norm(hidden_states)

        return hidden_states
