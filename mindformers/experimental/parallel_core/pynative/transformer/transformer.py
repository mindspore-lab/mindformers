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

from mindformers.experimental.parallel_core.pynative.parallel_state import (
    get_pp_rank,
    get_tp_world_size,
    get_pp_world_size,
    get_cp_world_size,
    get_vpp_rank,
    get_vpp_world_size
)
from mindformers.experimental.parallel_core.pynative.tensor_parallel.random import get_rng_tracer
from mindformers.experimental.parallel_core.pynative.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    CopyToModelParallelRegion,
    GatherFromModelParallelRegion,
    LinearWithGradAccumulationAndAsyncCommunication
)
from mindformers.experimental.parallel_core.pynative.tensor_parallel.lora_layers import (
    ColumnParallelLoRA,
    RowParallelLoRA,
)
from mindformers.experimental.parallel_core.pynative.transformer.rotary_pos_embedding import (
    apply_rotary_pos_emb,
)
from mindformers.experimental.parallel_core.pynative.transformer.scale_mask_softmax import (
    ScaleMaskSoftmax,
)
from mindformers.experimental.parallel_core.pynative.transformer.moe.moe_layer import MoELayer
from mindformers.experimental.parallel_core.pynative.transformer.norm import get_norm
from mindformers.experimental.parallel_core.pynative.transformer.utils import get_attn_mask_func
from mindformers.experimental.parallel_core.pynative.recompute import CheckpointedRecomputeOrientedCell
from mindformers.experimental.parallel_core.pynative.utils import divide
from mindformers.experimental.parallel_core.pynative.context_parallel.ring_attention import RingAttention

from .module import Module
from .mlp import ParallelMLP

__all__ = [
    "BasePublicLayer",
    "PublicLayer",
    "ParallelAttention",
    "ParallelTransformerLayer",
    "ParallelTransformer",
    "ParallelLMLogits"
]


class BasePublicLayer(Module):
    r"""
    A base class for public layer.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_attr_for_public_layer()

    def add_attr_for_public_layer(self):
        """add attr for public layer"""
        self.is_public_layer = True


class PublicLayer(BasePublicLayer):
    r"""
    Public layer class for building pipeline parallel model.

    Args:
        config (dict): Configuration.

    Inputs:
        - **input_ids** (Tensor) - The tokenized inputs with datatype int32, shape :math:`(B, S)`.
        - **labels** (Tensor) - The tokenized labels with datatype int32, shape :math:`(B, S)`.
        - **input_mask** (Tensor) - The mask for input_ids, shape:math:`(B, S)`.
        - **attention_mask** (Tensor) - Attention mask, shape :math:`(B, S, S)` or math:`(B, 1, S, S)`.
        - **position_ids** (Tensor) - Position ids for position embedding, shape :math:`(B, S)`.

    Outputs:
        - **output_dict** (dict) - A public dict for each pipeline stage.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(self, config, **kwargs):
        super(PublicLayer, self).__init__(**kwargs)
        self.pad_token = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.seq_length = config.seq_length
        self.compute_type = config.compute_dtype
        self.flatten_labels_and_input_mask = config.flatten_labels_and_input_mask
        self.output_dict = {}

    def construct(self,
                  input_ids,
                  labels=None,
                  input_mask=None,
                  attention_mask=None,
                  position_ids=None,
                  ) -> dict:
        """public layer forward"""
        if labels is None and self.seq_length == len(input_ids) - 1:
            input_ids, labels = input_ids[:, : self.seq_length], input_ids[:, 1:]
        if attention_mask is None:
            if input_mask is None:
                input_mask = mint.ne(input_ids, self.vocab_size + 1).astype(
                    self.compute_type
                )
            attention_mask = self.get_attention_mask(input_mask)

        if self.flatten_labels_and_input_mask:
            labels = labels.reshape((-1,))
            input_mask = input_mask.reshape((-1,))

        self.output_dict["input_ids"] = input_ids
        self.output_dict["attention_mask"] = attention_mask
        self.output_dict["labels"] = labels
        self.output_dict["position_ids"] = position_ids
        return self.output_dict

    def get_attention_mask(self, input_mask):
        """get attention mask base on input_mask"""
        input_shape = input_mask.shape
        ones = mint.ones((self.seq_length, self.seq_length), dtype=input_mask.dtype)
        attention_mask_left = input_mask.reshape((input_shape[0], input_shape[1], 1))
        attention_mask_right = input_mask.reshape((input_shape[0], 1, input_shape[1]))
        attention_mask = mint.matmul(attention_mask_left, attention_mask_right)
        lower_triangle_mask = ops.tril(ones).unsqueeze(0)
        attention_mask = mint.mul(attention_mask, lower_triangle_mask)
        return attention_mask


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
        self.data_layout = self.config.dataset_config.data_layout

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
        score = score * ops.cast(self.inv_norm_factor, score.dtype)

        # attention scores and attention mask [B, N, S_q, S_k]
        attention_probs = self.scale_mask_softmax(score, attention_mask)

        if not self.sequence_parallel:
            with get_rng_tracer().rng_fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # [B, N, S, S] * [B, N, S, D] -> [B, N, S, D]
        weighted_values = ops.bmm(ops.cast(attention_probs, value_layer.dtype), value_layer)
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
        self.data_layout = self.config.dataset_config.data_layout

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
            if self.hidden_size != self.kv_hidden_size:
                raise ValueError(f"The hidden_size should equal to the kv_hidden_size, "
                                 f"but got hidden_size: {self.hidden_size} and kv_hidden_size: {self.kv_hidden_size}.")
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

        if get_cp_world_size() > 1:
            self.ring_attention = RingAttention(
                self.num_heads,
                input_layout="BNSD",
                scale_value=1 / self.norm_factor,
                sparse_mode=0,
            )

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
        if self.data_layout == "SBH":
            seq_len, _, _ = hidden_states.shape
        else:
            _, seq_len, _ = hidden_states.shape
        # apply query, key, value projection
        if self.attn_type == 'self_attn':
            if self.sequence_parallel:
                seq_len = seq_len * self.tp_group_size
            # [B, S, H] --> [B, S, H + 2 * kv_H]
            qkv, _ = self.qkv_proj(hidden_states)
            qkv = self.cast(qkv, self.compute_dtype)
            new_tensor_shape = qkv.shape[:-1] + (
                self.kv_num_heads_per_partition,
                (
                    (self.num_heads_per_partition //
                     self.kv_num_heads_per_partition + 2)
                    * self.head_dim
                ),
            )

            mixed_x_layer = qkv.view(*new_tensor_shape)

            (query,
             key,
             value) = mint.split(
                 mixed_x_layer,
                 [
                     (
                         self.num_heads_per_partition // self.kv_num_heads_per_partition
                         * self.head_dim
                     ),
                     self.head_dim,
                     self.head_dim
                 ],
                 dim=3
             )

            query = query.reshape(query.shape[0], query.shape[1], -1, self.head_dim)
        else:
            kv, _ = self.kv_proj(encoder_output)
            kv = self.cast(kv, self.compute_dtype)

            new_tensor_shape = kv.shape[:-1] + (self.num_heads_per_partition, 2 * self.head_dim)
            kv = kv.view(*new_tensor_shape)

            last_dim = kv.ndim - 1
            (key, value) = mint.split(kv, split_size_or_sections=kv.shape[last_dim] // 2, dim=last_dim)

            query, _ = self.q_proj(hidden_states)
            new_tensor_shape = query.shape[:-1] + (self.num_heads_per_partition, self.head_dim)
            query = query.view(*new_tensor_shape)

        if self.data_layout == "SBH":
            key = key.swapaxes(0, 1)
            value = value.swapaxes(0, 1)
            query = query.transpose(1, 2, 0, 3)
        else:
            query = query.transpose(0, 2, 1, 3)

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
            key = key.transpose(0, 2, 1, 3)
            value = value.transpose(0, 2, 1, 3)

        # apply rotary position embedding
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            query = apply_rotary_pos_emb(query, q_pos_emb)
            key = apply_rotary_pos_emb(key, k_pos_emb)

        if not self.use_flash_attention:
            context_layer = self.core_attention(query, key, value, attention_mask)
        elif get_cp_world_size() <= 1:
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
        else:
            if query.dtype == mstype.float32:
                query = query.astype(mstype.float16)
            if key.dtype == mstype.float32:
                key = key.astype(mstype.float16)
            if value.dtype == mstype.float32:
                value = value.astype(mstype.float16)

                output = self.ring_attention(query, key, value, None)

            context_layer = _merge_heads(output)

        if self.data_layout == "SBH":
            context_layer = context_layer.swapaxes(0, 1)

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
        if self.config.moe_config is not None and self.config.moe_config.num_experts > 1:
            moe_config = copy.deepcopy(config)
            self.mlp = MoELayer(moe_config)
        else:
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

        with get_rng_tracer().rng_fork():
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

        with get_rng_tracer().rng_fork():
            out = self.hidden_states_dropout(mlp_output)
        output = residual + out

        return output


def _get_num_layers(config, model_type, is_decoder=False):
    """get transformer layers nums for current rank"""
    if model_type is not None:
        raise NotImplementedError(
            "For _get_num_layers function, `model_type` is not supported for now."
        )
    if is_decoder:
        raise NotImplementedError(
            "For _get_num_layers function, `is_decoder` is not supported for now."
        )
    vpp = get_vpp_world_size() if get_vpp_world_size() is not None else 1
    pp_split_num = vpp * get_pp_world_size()
    if config.num_layers < pp_split_num:
        raise RuntimeError(f"The number of model layers is {config.num_layers}, "
                           f"but using pipeline parallel requires at least "
                           f"'pp({get_pp_world_size()}) * vpp({vpp}) = {pp_split_num}' layers for splitting")
    standalone_embedding_stage = config.parallel_config.standalone_embedding_stage
    if get_pp_world_size() > 1:
        if standalone_embedding_stage and get_pp_rank() == 0:
            num_layers = 0
            offset = 0
        else:
            def divide_layers(num_layers, stage, rank):
                num_layer_list = [num_layers // stage] * stage
                remain_layer_nums = num_layers - sum(num_layer_list)
                for i in range(remain_layer_nums):
                    num_layer_list[-i - 2] += 1
                num_layers = num_layer_list[rank]
                offset = sum(num_layer_list[:rank])
                return num_layers, offset

            num_layers = config.num_layers
            offset = 0

            vpp_stage = get_vpp_world_size()
            if vpp_stage is not None:
                vpp_rank = get_vpp_rank()
                num_layers, offset_vpp = divide_layers(num_layers, vpp_stage, vpp_rank)
                offset = offset + offset_vpp

            pp_stage = get_pp_world_size() - 1 if standalone_embedding_stage else get_pp_world_size()
            pp_rank = get_pp_rank() - 1 if standalone_embedding_stage else get_pp_rank()
            num_layers, offset_pp = divide_layers(num_layers, pp_stage, pp_rank)
            offset = offset + offset_pp
    else:
        num_layers = config.num_layers
        offset = get_pp_rank() * num_layers
    return num_layers, offset


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
        if drop_path_rate > 0.0:
            raise NotImplementedError("For ParallelTransformer, `drop_path_rate > 0` is not supported for now, "
                                      "but got `drop_path_rate={}`".format(drop_path_rate))
        self.config = config
        self.pre_process = pre_process
        self.post_process = post_process
        self.post_norm = post_norm
        # number of layers.
        self.num_layers, offset = _get_num_layers(
            config, model_type=None, is_decoder=False
        )
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
            layers_dict[str(i + offset)] = ParallelTransformerLayer(
                config=layers_index_config[i] if use_lora else layers_config,
                layer_number=i + 1 + offset,
            )
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

        if self.post_process and self.post_norm:
            # final layernorm before output.
            self.final_norm = get_norm(config)

        self.pipeline_parallel = get_pp_world_size() > 1
        if self.pipeline_parallel:
            batch_size = config.dataset_config.batch_size
            seq_length = config.seq_length
            if config.parallel_config.use_sequence_parallel:
                seq_length = seq_length // get_tp_world_size()
            hidden_size = config.hidden_size
            if config.dataset_config.data_layout == "BSH":
                hidden_states_shape = (batch_size, seq_length, hidden_size)
            else:
                hidden_states_shape = (seq_length, batch_size, hidden_size)
            self.set_hidden_states = Parameter(
                mint.zeros(
                    hidden_states_shape, dtype=config.compute_dtype
                ),
                requires_grad=False,
                name="set_hidden_states",
            )

    def _set_checkpointed_recompute(self, recompute_method, recompute_num_layers):
        """Set checkpointed recompute for transformer."""
        self.checkpointed_recompute = True
        self.checkpointed_layer_groups = nn.CellList()
        if recompute_method == "uniform":
            for idx in range(0, self.num_layers, recompute_num_layers):
                checkpointed_layer_group = CheckpointedRecomputeOrientedCell(
                    self.layers[idx: idx + recompute_num_layers]
                )
                checkpointed_layer_group.recompute()
                self.checkpointed_layer_groups.append(checkpointed_layer_group)
        elif recompute_method == "block":
            for idx in range(0, min(self.num_layers, recompute_num_layers)):
                self.layers[idx].recompute()
            self.checkpointed_layer_groups = self.layers
        else:
            raise NotImplementedError(
                f"recompute_method should be uniform or blocks, but got {recompute_method}"
            )

    def set_input_tensor(self, input_tensor):
        """
        In pipeline parallel, the receiving data from previous stage will be set into class.
        Construct function's input will be replace by self.set_hidden_states.
        """
        self.set_hidden_states.set_data(input_tensor, slice_shape=True)

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

        # self.hidden_states instead of input
        if not self.pre_process and self.pipeline_parallel:
            hidden_states = self.set_hidden_states.value()

        for layer in layers:
            hidden_states = layer(hidden_states=hidden_states,
                                  attention_mask=attention_mask,
                                  rotary_pos_emb=rotary_pos_emb)

        # final layernorm.
        if self.post_process and self.post_norm:
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
        self.sequence_parallel = config.parallel_config.use_sequence_parallel
        self.allreduce_dgrad = (
            get_tp_world_size() > 1 and not self.sequence_parallel
        )

        self.copy_to_mp_region = CopyToModelParallelRegion()
        self.gradient_accumulation_fusion = config.parallel_config.gradient_accumulation_fusion
        self.forward_impl_ = LinearWithGradAccumulationAndAsyncCommunication(
            bias=bias,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            sequence_parallel=self.sequence_parallel,
            allreduce_dgrad=self.allreduce_dgrad,
            data_layout=config.dataset_config.data_layout
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
