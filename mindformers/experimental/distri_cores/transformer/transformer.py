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
import copy
from collections import OrderedDict

import mindspore.common.dtype as mstype
from mindspore import Tensor, nn, ops, mint, Parameter
import mindspore.ops.functional as F

from mindformers.experimental.distri_cores.create_comm import get_pp_rank, get_tp_world_size
from mindformers.experimental.distri_cores.random import get_rng_tracer
from mindformers.experimental.distri_cores.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    CopyToModelParallelRegion,
    GatherFromModelParallelRegion,
    LinearWithGradAccumulationAndAsyncCommunication
)
from mindformers.experimental.distri_cores.tensor_parallel.lora_layers import (
    ColumnParallelLoRA,
    RowParallelLoRA,
)
from mindformers.experimental.distri_cores.transformer.rotary_pos_embedding import (
    apply_rotary_pos_emb,
)
from mindformers.experimental.distri_cores.transformer.scale_mask_softmax import (
    ScaleMaskSoftmax,
)
from mindformers.experimental.distri_cores.transformer.norm import get_norm
from mindformers.experimental.distri_cores.transformer.activation import get_act_func, get_act_func_gated_version
from mindformers.experimental.distri_cores.transformer.utils import get_attn_mask_func
from mindformers.experimental.distri_cores.recompute import CheckpointedRecomputeOrientedCell

from mindformers.experimental.distri_cores.utils import divide

from .module import Module

__all__ = [
    "ParallelMLP",
    "ParallelAttention",
    "ParallelTransformerLayer",
    "ParallelTransformer",
    "ParallelLMLogits"
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
        - **output_bias** (Parameter) - Output projection bias weight when `projection.skip_bias_add=True`.

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
        self.use_lora = config.lora_config.use_lora
        self.act_type = self.config.hidden_act
        self.is_expert = is_expert

        self._init_mapping()
        self.bias_gelu_fusion = False
        self.act_func = get_act_func(self.act_type)

        self._init_projection()

    def _init_mapping(self):
        """ initialize mapping cell """
        mapping_output_size = self.ffn_hidden_size
        if self.config.mlp_has_gate:
            gated_act_type = get_act_func_gated_version(self.act_type)
            if gated_act_type is not None:
                self.mapping_gate_fusion = True
                self.act_type = gated_act_type
                mapping_output_size *= 2
            else:
                self.mapping_gate_fusion = False
                self.gating = ColumnParallelLinear(
                    self.hidden_size,
                    self.ffn_hidden_size,
                    config=self.config,
                    init_method=self.config.init_method,
                    bias=self.has_bias,
                    gather_output=False,
                    is_expert=self.is_expert,
                    bias_init=self.config.bias_init,
                )
        self.mapping = ColumnParallelLinear(
            self.hidden_size,
            mapping_output_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.has_bias,
            gather_output=False,
            is_expert=self.is_expert,
            param_init_dtype=self.config.param_init_dtype,
            compute_dtype=self.config.compute_dtype,
            bias_init=self.config.bias_init,
        )
        if self.use_lora:
            mapping_lora = self._get_cell_lora_config(self.config, 'mapping')
            if mapping_lora is not None:
                self.mapping = ColumnParallelLoRA(
                    self.hidden_size,
                    mapping_output_size,
                    config=self.config,
                    init_method=self.config.init_method,
                    bias=self.has_bias,
                    gather_output=False,
                    is_expert=self.is_expert,
                    param_init_dtype=self.config.param_init_dtype,
                    compute_dtype=self.config.compute_dtype,
                    bias_init=self.config.bias_init,
                    lora_rank=mapping_lora['rank'],
                    lora_alpha=mapping_lora['alpha'],
                    lora_dropout=mapping_lora['dropout'],
                )
            gating_lora = self._get_cell_lora_config(self.config, 'gating')
            if self.config.mlp_has_gate and not self.mapping_gate_fusion and gating_lora is not None:
                self.gating = ColumnParallelLoRA(
                    self.hidden_size,
                    self.ffn_hidden_size,
                    config=self.config,
                    init_method=self.config.init_method,
                    bias=self.has_bias,
                    gather_output=False,
                    is_expert=self.is_expert,
                    bias_init=self.config.bias_init,
                    lora_rank=gating_lora['rank'],
                    lora_alpha=gating_lora['alpha'],
                    lora_dropout=gating_lora['dropout'],
                )

    def _init_projection(self):
        """ initialize projection cell """
        self.projection = RowParallelLinear(
            self.ffn_hidden_size,
            self.hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.has_bias,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=self.is_expert,
            param_init_dtype=self.config.param_init_dtype,
            compute_dtype=self.config.compute_dtype,
            bias_init=self.config.bias_init,
        )
        if self.use_lora:
            projection_lora = self._get_cell_lora_config(self.config, 'projection')
            if projection_lora is not None:
                self.projection = RowParallelLoRA(
                    self.ffn_hidden_size,
                    self.hidden_size,
                    config=self.config,
                    init_method=self.config.init_method,
                    bias=self.has_bias,
                    input_is_parallel=True,
                    skip_bias_add=False,
                    is_expert=self.is_expert,
                    param_init_dtype=self.config.param_init_dtype,
                    compute_dtype=self.config.compute_dtype,
                    bias_init=self.config.bias_init,
                    lora_rank=projection_lora['rank'],
                    lora_alpha=projection_lora['alpha'],
                    lora_dropout=projection_lora['dropout'],
                )

    def construct(self, hidden_states):
        """ Construct function of mlp block. """
        # [B, S, H] -> [B, S, ffn_H]
        if self.config.mlp_has_gate and not self.mapping_gate_fusion:
            gate, _ = self.gating(hidden_states)
            gate = self.act_func(gate)
            intermediate_parallel, _ = self.mapping(hidden_states)
            intermediate_parallel = mint.mul(intermediate_parallel, gate)
        else:
            intermediate_parallel, _ = self.mapping(hidden_states)
            intermediate_parallel = self.act_func(intermediate_parallel)

        # [B, S, ffn_H] -> [B, S, H]
        output, output_bias = self.projection(intermediate_parallel)
        return output, output_bias


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
        - **context_layer** (Tensor) - Tensor of shape :math:`(B, S, H)`.

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
        context_layer = _merge_heads(weighted_values)

        return context_layer


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
        - **bias** (Tensor) - Output out_proj dense layer's bias weight when `projection.skip_bias_add=True`.

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
        if self.use_flash_attention:
            self.fa_config = self.config.fa_config
        self.norm_factor = math.sqrt(self.head_dim)

        tp_group_size = get_tp_world_size()
        self.tp_group_size = tp_group_size
        self.num_heads_per_partition = divide(self.num_heads, tp_group_size)
        self.use_lora = self.config.lora_config.use_lora

        if self.use_gqa:
            if self.kv_num_heads % tp_group_size != 0:
                raise ValueError(
                    "The kv_num_heads should be " "a multiple of the tensor parallel size"
                )
            self.kv_num_heads_per_partition = divide(self.kv_num_heads, tp_group_size)
        else:
            self.kv_num_heads_per_partition = self.num_heads_per_partition

        if self.attn_type == 'self_attn':
            self.qkv_proj = self._init_qkv_proj(
                self.hidden_size,
                self.hidden_size + 2 * self.kv_hidden_size,
                cell_name='qkv_proj',
                gather_output=False,
            )

        elif self.attn_type == 'cross_attn':
            assert self.hidden_size == self.kv_hidden_size
            self.q_proj = self._init_qkv_proj(
                self.hidden_size,
                self.hidden_size,
                cell_name='q_proj',
                gather_output=False,
            )
            self.kv_proj = self._init_qkv_proj(
                self.hidden_size,
                2 * self.kv_hidden_size,
                cell_name='kv_proj',
                gather_output=False,
            )
        else:
            raise NotImplementedError(f"attention_type should be self_attn or cross_attn, but got {self.attn_type}")

        self.core_attention = CoreAttention(self.layer_index, self.config)

        self.out_proj = self._init_out_proj(
            self.hidden_size,
            self.hidden_size,
            cell_name='out_proj',
            input_is_parallel=True,
        )

    def _init_qkv_proj(self, input_size, output_size, cell_name, gather_output=False):
        """Construct qkv projection cell."""
        proj = ColumnParallelLinear(
            input_size,
            output_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.config.qkv_has_bias,
            gather_output=gather_output,
            skip_bias_add=False,
            bias_init=self.config.bias_init,
        )
        if self.use_lora:
            proj_lora = self._get_cell_lora_config(self.config, cell_name)
            if proj_lora is not None:
                proj = ColumnParallelLoRA(
                    input_size,
                    output_size,
                    config=self.config,
                    init_method=self.config.init_method,
                    bias=self.config.qkv_has_bias,
                    gather_output=gather_output,
                    skip_bias_add=False,
                    bias_init=self.config.bias_init,
                    lora_rank=proj_lora['rank'],
                    lora_alpha=proj_lora['alpha'],
                    lora_dropout=proj_lora['dropout'],
                )
        return proj

    def _init_out_proj(self, input_size, output_size, cell_name, input_is_parallel=True):
        """Construct out projection cell."""
        out_proj = RowParallelLinear(
            input_size,
            output_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.config.out_proj_has_bias,
            input_is_parallel=input_is_parallel,
            skip_bias_add=False,
            bias_init=self.config.bias_init,
        )
        if self.use_lora:
            out_proj_lora = self._get_cell_lora_config(self.config, cell_name)
            if out_proj_lora is not None:
                out_proj = RowParallelLoRA(
                    input_size,
                    output_size,
                    config=self.config,
                    init_method=self.config.init_method,
                    bias=self.config.out_proj_has_bias,
                    input_is_parallel=input_is_parallel,
                    skip_bias_add=False,
                    bias_init=self.config.bias_init,
                    lora_rank=out_proj_lora['rank'],
                    lora_alpha=out_proj_lora['alpha'],
                    lora_dropout=out_proj_lora['dropout'],
                )
        return out_proj

    def construct(
            self,
            hidden_states,
            attention_mask,
            encoder_output=None,
            inference_params=None,
            rotary_pos_emb=None,
        ):
        """ Construct function of attention block. """
        if inference_params:
            raise NotImplementedError("inference_params is not supported for now.")

        # hidden_states: [B, S, H]
        ori_dtype = hidden_states.dtype
        bs, seq_len, _ = hidden_states.shape
        # apply query, key, value projection
        if self.attn_type == 'self_attn':
            if self.sequence_parallel:
                seq_len = seq_len * self.tp_group_size
            # [B, S, H] --> [B, S, H + 2 * kv_H]
            qkv, _ = self.qkv_proj(hidden_states)
            qkv = self.cast(qkv, self.compute_dtype)

            split_sections = [
                self.num_heads_per_partition * self.head_dim,
                self.kv_num_heads_per_partition * self.head_dim,
                self.kv_num_heads_per_partition * self.head_dim
            ]
            (query, key, value) = mint.split(qkv, split_sections, dim=-1)

            # [B, S, H] -> [B, N, S, D]
            query = query.reshape(bs, seq_len, -1, self.head_dim).transpose((0, 2, 1, 3))
            # [B, S, H] -> [B, S, N, D]
            key = key.reshape(bs, seq_len, -1, self.head_dim)
            value = value.reshape(bs, seq_len, -1, self.head_dim)
        else:
            kv, _ = self.kv_proj(encoder_output)
            kv = self.cast(kv, self.compute_dtype)

            # split tensor along last dimension.
            last_dim = kv.ndim - 1
            (key, value) = mint.split(kv, split_size_or_sections=kv.shape[last_dim]//2, dim=last_dim)

            new_tensor_shape = kv.shape[:-1] + (
                self.num_heads_per_partition,
                self.head_dim,
            )

            key = key.view(*new_tensor_shape)
            value = value.view(*new_tensor_shape)

            q, _ = self.q_proj(hidden_states)
            q = self.cast(q, self.compute_dtype)
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

        if divide(self.num_heads_per_partition, self.kv_num_heads_per_partition) > 1 and not self.use_flash_attention:
            # expand the key_layer and value_layer [B, S, kv_N_per_tp, D] to [B, S, N_per_tp, D]
            repeat_num = divide(self.num_heads_per_partition, self.kv_num_heads_per_partition)
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
            if attention_mask.ndim == 3:
                attention_mask = attention_mask.expand_dims(axis=1)
            if query.dtype == mstype.float32:
                query = query.astype(mstype.float16)
            if key.dtype == mstype.float32:
                key = key.astype(mstype.float16)
            if value.dtype == mstype.float32:
                value = value.astype(mstype.float16)
            if self.fa_config:
                output = ops.flash_attention_score(
                    query,
                    key,
                    value,
                    self.num_heads_per_partition,
                    attn_mask=attention_mask,
                    scalar_value=1.0 / self.norm_factor,
                    **self.fa_config,
                )
            else:
                output = ops.flash_attention_score(
                    query,
                    key,
                    value,
                    self.num_heads_per_partition,
                    attn_mask=attention_mask,
                    scalar_value=1.0 / self.norm_factor,
                )
            context_layer = _merge_heads(output)

        # apply output projection
        output, bias = self.out_proj(context_layer)
        output = self.cast(output, ori_dtype)

        return output, bias

    def _repeat_kv(self, x, rep):
        """ Expand key, value on num_head dimension. """
        if rep == 1:
            return x
        bs, seq_length, num_groups, head_dim = x.shape
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
        use_lora = config.lora_config.use_lora
        # Normalize the input data.
        self.input_norm = get_norm(config)

        # Attention.
        attention_config = copy.deepcopy(config)
        if use_lora:
            attention_config.update_lora_config('attention')
        self.attention = ParallelAttention(attention_config, layer_number)

        # Normalize the attention output
        self.post_attention_norm = get_norm(config)

        # MLP
        mlp_config = copy.deepcopy(config)
        if use_lora:
            mlp_config.update_lora_config('mlp')
        self.mlp = ParallelMLP(mlp_config)

        self.hidden_states_dropout = mint.nn.Dropout(p=self.config.hidden_dropout_rate)

        # selective recompute
        if self.config.recompute_granularity == "selective":
            self._set_selective_recompute()

    def _set_selective_recompute(self):
        """Set selective recompute for transformer layer."""
        self.attention.core_attention.recompute()

    def construct(self,
                  hidden_states,
                  attention_mask,
                  encoder_output=None,
                  enc_dec_attn_mask=None,
                  retriever_input=None,
                  retriever_output=None,
                  retriever_attn_mask=None,
                  inference_params=None,
                  rotary_pos_emb=None):
        """ Construct function of transformer layer. """
        if encoder_output is not None:
            raise NotImplementedError("encoder_output is not supported for now.")
        if enc_dec_attn_mask is not None:
            raise NotImplementedError("enc_dec_attn_mask is not supported for now.")
        if retriever_input is not None:
            raise NotImplementedError("retriever_input is not supported for now.")
        if retriever_output is not None:
            raise NotImplementedError("retriever_output is not supported for now.")
        if retriever_attn_mask is not None:
            raise NotImplementedError("retriever_attn_mask is not supported for now.")
        if inference_params is not None:
            raise NotImplementedError("inference_params is not supported for now.")

        # hidden_states: [B, S, H]
        # layernorm at the beginning of the transformer layer.
        norm_output = self.input_norm(hidden_states)
        # attention.
        attention_output, _ = self.attention(hidden_states=norm_output,
                                             attention_mask=attention_mask,
                                             rotary_pos_emb=rotary_pos_emb)

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
        mlp_output, _ = self.mlp(norm_output)

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

        offset = get_pp_rank() * self.num_layers

        use_lora = config.lora_config.use_lora

        layers_config = copy.deepcopy(config)
        if use_lora:
            layers_config.update_lora_config('layers')
            layers_index_config = []
            for i in range(self.num_layers):
                layer_config_new = copy.deepcopy(layers_config)
                layer_config_new.update_lora_config(f'{i}')
                layers_index_config.append(layer_config_new)

        # ensure the Parameter of each rank init as correct name
        layers_dict = OrderedDict()
        for i in range(self.num_layers):
            layers_dict[str(i + offset)] = ParallelTransformerLayer(config=layers_index_config[i] \
                if use_lora else layers_config,
                                                                    layer_number=i + 1 + offset)
        self.layers = nn.SequentialCell(layers_dict)

        # gradient checkpointing for recompute.
        self.checkpointed_recompute = (
            self.config.recompute_method is not None
            and self.config.recompute_granularity is not None
            and self.config.recompute_num_layers is not None
            and self.config.recompute_granularity == "full"
        )
        if self.checkpointed_recompute:
            self._set_checkpointed_recompute(self.config.recompute_method, self.config.recompute_num_layers)

        if self.post_norm:
            # final layernorm before output.
            self.final_norm = get_norm(config)

    def _set_checkpointed_recompute(self, recompute_method, recompute_num_layers):
        """Set checkpointed recompute for transformer."""
        self.checkpointed_recompute = True
        self.checkpointed_layer_groups = nn.CellList()
        if recompute_method == "uniform":
            for idx in range(0, self.num_layers, recompute_num_layers):
                checkpointed_layer_group = CheckpointedRecomputeOrientedCell(
                    self.layers[idx : idx + recompute_num_layers]
                )
                checkpointed_layer_group.recompute()
                self.checkpointed_layer_groups.append(checkpointed_layer_group)
        elif recompute_method == "block":
            self.checkpointed_layer_groups = nn.CellList(
                [CheckpointedRecomputeOrientedCell(self.layers[:recompute_num_layers])]
            )
            for layer in self.layers[recompute_num_layers:]:
                self.checkpointed_layer_groups.append(layer)
            self.checkpointed_layer_groups[0].recompute()
        else:
            raise NotImplementedError(f"recompute_method should be uniform or blocks, but got {recompute_method}")

    def construct(self,
                  hidden_states,
                  attention_mask,
                  encoder_output=None,
                  enc_dec_attn_mask=None,
                  retriever_input=None,
                  retriever_output=None,
                  retriever_attn_mask=None,
                  inference_params=None,
                  rotary_pos_emb=None):
        """ Construct function of transformer. """
        if encoder_output is not None:
            raise NotImplementedError("encoder_output is not supported for now.")
        if enc_dec_attn_mask is not None:
            raise NotImplementedError("enc_dec_attn_mask is not supported for now.")
        if retriever_input is not None:
            raise NotImplementedError("retriever_input is not supported for now.")
        if retriever_output is not None:
            raise NotImplementedError("retriever_output is not supported for now.")
        if retriever_attn_mask is not None:
            raise NotImplementedError("retriever_attn_mask is not supported for now.")
        if inference_params is not None:
            raise NotImplementedError("inference_params is not supported for now.")

        if self.checkpointed_recompute:
            layers = self.checkpointed_layer_groups
        else:
            layers = self.layers

        for layer in layers:
            hidden_states = layer(hidden_states=hidden_states,
                                  attention_mask=attention_mask,
                                  rotary_pos_emb=rotary_pos_emb)

        # final layernorm.
        if self.post_norm:
            hidden_states = self.final_norm(hidden_states)

        return hidden_states


class ParallelLMLogits(nn.Cell):
    r"""
    Head to get the logits of each token in the vocab.

    Args:
        config (dict): Parallel configuration.
        bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        transpose_b (bool): Specifies whether the weight parameter will be initialized as a transposed shape.
        compute_dtype (dtype.Number): The computation type. Default: None.

    Inputs:
        - **input_** (Tensor) - Tensor of hidden states.
        - **word_embedding_table** (Parameter) - Weight matrix passed from embedding layer.
        - **parallel_output** (bool) - Specifies whether return paralleled output on each tensor parallel rank.
          Default: True.
        - **bias** (Tensor) - The trainable bias parameter.

    Outputs:
        Tensor of logits.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(self, config, bias=True, compute_dtype=None):
        super(ParallelLMLogits, self).__init__()
        self.compute_dtype = compute_dtype if compute_dtype else config.compute_dtype
        self.sequence_parallel = config.use_sequence_parallel
        self.allreduce_dgrad = (
            get_tp_world_size() > 1 and not self.sequence_parallel
        )

        self.copy_to_mp_region = CopyToModelParallelRegion()
        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion
        self.forward_impl_ = LinearWithGradAccumulationAndAsyncCommunication(
            bias=bias,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            sequence_parallel=self.sequence_parallel,
            allreduce_dgrad=self.allreduce_dgrad
        )
        self.gather_from_mp_region = GatherFromModelParallelRegion()

    def construct(self, input_, word_embeddings_weight, parallel_output=True, bias=None):
        """LM logits using word embedding table"""
        if (
                self.sequence_parallel
                or self.allreduce_dgrad
        ):
            input_parallel = input_
        else:
            input_parallel = self.copy_to_mp_region(input_)

        origin_dtype = F.dtype(input_parallel)
        weight = ops.cast(word_embeddings_weight, self.compute_dtype)
        weight_param = None
        if self.gradient_accumulation_fusion and isinstance(word_embeddings_weight, Parameter):
            weight_param = word_embeddings_weight
        input_parallel = ops.cast(input_parallel, self.compute_dtype)

        bias = ops.cast(bias, self.compute_dtype) if bias else None

        # Matrix multiply.
        logits_parallel = self.forward_impl_(input_parallel, weight, bias, weight_param=weight_param)
        logits_parallel = ops.cast(logits_parallel, origin_dtype)

        # Gather if needed.
        if parallel_output:
            return logits_parallel

        return self.gather_from_mp_region(logits_parallel)
