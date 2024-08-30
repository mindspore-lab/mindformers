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
"""LLaMA models' APIs."""
import math
from typing import Tuple

import mindspore as ms
import mindspore.common.dtype as mstype
import numpy as np
from mindspore import Parameter, Tensor, mint, nn, ops
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P

from mindformers.experimental.distri_cores.create_comm import (
    get_pp_rank, get_tp_rank, get_tp_world_size)
from mindformers.experimental.distri_cores.random import get_rng_tracer
from mindformers.experimental.distri_cores.tensor_parallel import (
    ReduceFromModelParallelRegion, ReduceScatterToSequenceParallelRegion)
from mindformers.experimental.distri_cores.tensor_parallel.layers import (
    ColumnParallelLinear, RowParallelLinear)
from mindformers.experimental.distri_cores.transformer import (
    Module, get_attn_mask_func)
from mindformers.experimental.distri_cores.transformer.scale_mask_softmax import \
    ScaleMaskSoftmax
from mindformers.experimental.distri_cores.transformer.transformer import \
    _merge_heads
from mindformers.experimental.distri_cores.utils import divide
from mindformers.models.llama.llama import LlamaPreTrainedModel
from mindformers.models.llama.llama_layer import LlamaSiLU
from mindformers.models.utils import lazy_inline
from mindformers.modules import PagedAttentionMgr
from mindformers.modules.infer_attention import InferRotaryEmbedding
from mindformers.modules.layers import FreqsMgr, RotaryEmbedding
from mindformers.modules.transformer import LowerTriangularMaskWithDynamic
from mindformers.tools.register.register import (MindFormerModuleType,
                                                 MindFormerRegister)
from mindformers.tools.utils import get_predict_run_mode
from mindformers.version_control import check_rmsnorm_big_kernel_valid

from .utils import convert_model_config

__all__ = ["ParallelLlamaModel", "ParallelLlamaForCausalLM"]


class LlamaRMSNorm(nn.Cell):
    r"""
    A self-defined RMSNorm operation using reduce mean.

        Args:
            dim (tuple): The shape of the input tensor
            eps (float): The epsilon value of the denominator. Default 1e-5.
            compute_type: The compute type.
        Inputs:
            - **x** (Tensor) - Tensor of shape :math:`(batch, seq\_length, hidden\_size)`.

        Outputs:
            Tensor of shape :math:`(batch, seq_length, hidden_size)`.
    """

    def __init__(self, dim, eps=1e-6, compute_type=mstype.float32):
        super(LlamaRMSNorm, self).__init__()
        self.eps = eps
        self.compute_type = compute_type
        self.weight = Parameter(initializer('ones', (dim,), dtype=self.compute_type), parallel_optimizer=False)

        if check_rmsnorm_big_kernel_valid():
            self.norm = P.RmsNorm(eps)
            self.rms_norm = self._rms_norm
            self.self_define = False
            self.cast = P.Cast()
            self.rcast = P.Cast()
        else:
            self.cast = P.Cast()
            self.mul = P.Mul()
            self.mul2 = P.Mul()
            self.square = P.Square()
            self.mean = P.ReduceMean(keep_dims=True)
            self.add = P.Add()
            self.rsqrt = P.Rsqrt()
            self.rms_norm = self._self_norm
            self.self_define = True

    def _self_norm(self, x):
        original_type = x.dtype
        norm_factor = self.square(self.cast(x, self.compute_type))
        norm_factor = self.mean(norm_factor, -1)
        norm_factor = self.add(norm_factor, self.eps)
        norm_factor = self.rsqrt(norm_factor)
        output = self.mul(x, self.cast(norm_factor, original_type))
        output = self.mul2(output, self.cast(self.weight, original_type))
        return output

    def _rms_norm(self, x):
        original_type = x.dtype
        output = self.norm(self.cast(x, self.compute_type), self.weight)[0]
        return self.rcast(output, original_type)

    def construct(self, x):
        """Forward of RMSNorm."""
        return self.rms_norm(x)


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
        self.scale_mask_softmax = ScaleMaskSoftmax(self.mask_func, softmax_compute_type=self.softmax_compute_dtype)

        self.attention_dropout = nn.Dropout(p=self.config.attention_dropout_rate)

    def construct(self, query_layer, key_layer, value_layer, attention_mask):
        """construct."""
        # score: [B, N, S, S]
        score = ops.bmm(query_layer, key_layer.transpose(0, 1, 3, 2))
        score = score * self.inv_norm_factor

        # attention scores and attention mask [B, N, S_q, S_k]
        attention_probs = self.scale_mask_softmax(score, attention_mask)

        # if not self.sequence_parallel:
        #     with get_rng_tracer().rng_fork():
        #         attention_probs = self.attention_dropout(attention_probs)
        # else:
        attention_probs = self.attention_dropout(attention_probs)

        # [B, N, S, S] * [B, N, S, D] -> [B, N, S, D]
        weighted_values = ops.bmm(attention_probs, value_layer)
        # [B, N, S, D] -> [B, S, N*D]
        attn_output = _merge_heads(weighted_values)

        return attn_output


class ParallelLlamaMLPWithGate(Module):
    def __init__(self, config, is_expert=False):
        super().__init__(config)
        self.config = config
        self.has_bias = self.config.mlp_has_bias
        self.hidden_size = self.config.hidden_size
        self.ffn_hidden_size = self.config.ffn_hidden_size
        self.mlp_has_gate = self.config.mlp_has_gate
        self.split = ms.ops.auto_generate.SplitWithSize()

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
        tp_group_size = get_tp_world_size()
        self.ffn_hidden_size_per_partition = divide(self.ffn_hidden_size, tp_group_size)
        # self.bias_gelu_fusion = False

        self.act_type = self.config.hidden_act
        # self.act_func = get_act_func(self.act_type)
        self.act_func = LlamaSiLU()

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

    def construct(self, hidden_states):
        """Construct function of mlp block."""
        gate_hidden_out = self.w_gate_hidden(hidden_states)
        gate, hidden = self.split(gate_hidden_out,
                                  (self.ffn_hidden_size_per_partition, self.ffn_hidden_size_per_partition), 2)
        gate = self.act_func(gate)
        hidden = self.mul(hidden, gate)
        output = self.w2(hidden)
        return output


class ParallelLlamaAttention(Module):
    def __init__(self, config, layer_number, attention_type="self_attn", attn_mask_type=None):
        super().__init__(config)
        if attn_mask_type:
            raise NotImplementedError("For ParallelAttention, `attn_mask_type` is not supported for now.")
        self.config = config
        self.layer_index = max(1, layer_number)
        self.param_init_dtype = self.config.param_init_dtype
        self.compute_dtype = self.config.compute_dtype
        self.is_first_iteration = True
        self.use_past = self.config.use_past

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
                    "Currently the kv_num_heads should be " "a multiple of the tensor parallel size"
                )
            self.kv_num_heads_per_partition = divide(self.kv_num_heads, tp_group_size)
        else:
            self.kv_num_heads_per_partition = self.num_heads_per_partition

        if self.attn_type == "self_attn":
            self.w_qkv = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size + 2 * self.kv_hidden_size,
                config=self.config.parallel_config,
                bias=self.config.qkv_has_bias,
                gather_output=False,
                transpose_b=True,
                param_init_type=self.config.param_init_dtype,
                compute_dtype=self.config.compute_dtype,
            )
            self.hidden_size_per_partition = divide(self.hidden_size, tp_group_size)
            self.kv_hidden_size_per_partition = divide(self.kv_hidden_size, tp_group_size)
            self.split_qkv = ms.ops.auto_generate.SplitWithSize()
        elif self.attn_type == "cross_attn":
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

        self.apply_rotary_emb = RotaryEmbedding(self.head_dim, config.rotary_dtype)

        self.core_attention = CoreAttention(self.layer_index, self.config)

        self.wo = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            input_is_parallel=True,
            config=self.config.parallel_config,
            bias=self.config.out_proj_has_bias,
            transpose_b=True,
            param_init_type=self.config.param_init_dtype,
            compute_dtype=self.config.compute_dtype,
        )
        if self.use_past:
            kv_shape = (self.config.num_blocks, self.config.block_size, self.kv_num_heads_per_partition, self.head_dim)
            self.paged_attention_mgr = PagedAttentionMgr(self.num_heads_per_partition,
                                                         self.head_dim,
                                                         self.kv_num_heads_per_partition,
                                                         kv_shape,
                                                         compute_dtype=self.compute_dtype)
            self.rotary_embedding = InferRotaryEmbedding(rotary_cos_format=2)

    def construct(self, x: Tensor, freqs_cis: Tuple[Tensor, Tensor], mask=None, batch_valid_length=None,
                  block_tables=None, slot_mapping=None, prefix_keys_values=None):
        """Construct function of attention block."""
        # hidden_states: [B, S, H]
        ori_dtype = x.dtype
        bs, seq_len, _ = x.shape
        # apply query, key, value projection
        if self.attn_type == "self_attn":
            if self.sequence_parallel:
                seq_len = seq_len * self.tp_group_size
            # [B, S, H] --> [B, S, H + 2 * kv_H]
            qkv = self.cast(self.w_qkv(x), self.compute_dtype)
            query, key, value = self.split_qkv(qkv, (
                self.hidden_size_per_partition, self.kv_hidden_size_per_partition, self.kv_hidden_size_per_partition),
                2)

        else:
            raise NotImplementedError("LLaMAAttention don't support CrossAttention")

        if self.use_past:
            query, key = self.rotary_embedding(query, key, freqs_cis, batch_valid_length)
            key_out = self.paged_attention_mgr(key, value, slot_mapping)
            query = ops.depend(query, key_out)

        if self.is_first_iteration:
            # [B, S, H] -> [B, N, S, D]
            query = query.reshape(bs, seq_len, -1, self.head_dim).transpose((0, 2, 1, 3))
            # [B, S, H] -> [B, S, N, D]
            key = key.reshape(bs, seq_len, -1, self.head_dim)
            value = value.reshape(bs, seq_len, -1, self.head_dim)
            # expand the key_layer and value_layer [B, S, kv_N_per_tp, D]
            # to [B, S, N_per_tp, D]
            if self.num_heads_per_partition // self.kv_num_heads_per_partition > 1:
                repeat_num = self.num_heads_per_partition - self.kv_num_heads_per_partition
                key = self._repeat_kv(key, repeat_num)
                value = self._repeat_kv(value, repeat_num)
            else:
                key = key.transpose((0, 2, 1, 3))
                value = value.transpose((0, 2, 1, 3))

            if not self.use_past:
                # apply rotary position embedding
                query, key = self.apply_rotary_emb(query, key, freqs_cis)

            if not self.use_flash_attention:
                context_layer = self.core_attention(query, key, value, mask)
            else:
                # if mask.ndim == 3:
                #     mask = mask.expand_dims(axis=1)
                # if query.dtype == mstype.float32:
                #     query = query.astype(mstype.float16)
                # if key.dtype == mstype.float32:
                #     key = key.astype(mstype.float16)
                # if value.dtype == mstype.float32:
                #     value = value.astype(mstype.float16)
                # mask = mask.astype(mstype.uint8)
                output = ops.flash_attention_score(
                    query,
                    key,
                    value,
                    self.num_heads_per_partition,
                    attn_mask=mask,
                    scalar_value=1.0 / self.norm_factor,
                    input_layout="BNSD",
                    sparse_mode=0,
                    pre_tokens=65536,
                    next_tokens=0,
                )
                context_layer = _merge_heads(output)
        else:
            context_layer = self.paged_attention_mgr.paged_attn(query, batch_valid_length, block_tables)

        # apply output projection
        output = self.wo(context_layer)
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


class ParallelLlamaTransformerLayer(Module):
    def __init__(
            self,
            config,
            layer_number,
            layer_type=None,
            self_attn_mask_type=None,
            drop_path_rate=0.0,
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
        self.hidden_size = self.config.hidden_size
        self.layer_index = layer_number

        self.apply_residual_connection_post_norm = self.config.apply_residual_connection_post_norm

        self.residual_connection_dtype = self.config.residual_connection_dtype

        # Normalize the input data.
        self.attention_norm = LlamaRMSNorm(self.hidden_size, config.layernorm_epsilon,
                                           compute_type=config.layernorm_compute_dtype)

        # Attention.
        self.attention = ParallelLlamaAttention(config, layer_number)

        # Normalize the attention output
        self.ffn_norm = LlamaRMSNorm(self.hidden_size, config.layernorm_epsilon,
                                     compute_type=config.layernorm_compute_dtype)

        # MLP
        self.feed_forward = ParallelLlamaMLPWithGate(config)

        # selective recompute
        if self.config.recompute_granularity == "selective":
            self._set_selective_recompute()

    def _set_selective_recompute(self):
        """Set selective recompute for transformer layer."""
        self.attention.core_attention.recompute()

    def construct(self, x, freqs_cis, mask=None, batch_valid_length=None, block_tables=None,
                  slot_mapping=None, prefix_keys_values=None):
        """Construct function of transformer layer."""
        # hidden_states: [B, S, H]
        # layernorm at the beginning of the transformer layer.
        norm_output = self.attention_norm(x)

        # attention.
        attention_output = self.attention(norm_output, freqs_cis, mask, batch_valid_length, block_tables,
                                          slot_mapping, prefix_keys_values)

        # residual-connection.
        if self.apply_residual_connection_post_norm:
            residual = norm_output
        else:
            residual = x

        norm_input = residual + attention_output

        # layernorm post attention.
        norm_output = self.ffn_norm(norm_input)

        # MLP.
        mlp_output = self.feed_forward(norm_output)

        # residual-connection.
        if self.apply_residual_connection_post_norm:
            residual = norm_output
        else:
            residual = norm_input

        output = residual + mlp_output

        return output


class VocabParallelLlamaEmbedding(nn.Cell):
    """
    Embedding parallelized in the vocabulary dimension.

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        parallel_config (Optional[Union[dict, ParallelContextConfig]]):
            Parallel Config For Running Environment. Default: None.
        init_method (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The values
            of str refer to the function `initializer`. Default: 'normal'.
        init_type (dtype.Number): The parameter initialization type. Default: mstype.float32.
    """

    def __init__(
            self,
            num_embeddings,
            embedding_dim,
            parallel_config,
            init_method="normal",
            init_type=mstype.float32,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.sequence_parallel = parallel_config.use_sequence_parallel

        self.tensor_model_parallel_size = get_tp_world_size()

        (
            self.vocab_start_index,
            self.vocab_end_index,
        ) = self._vocab_range_from_global_vocab_size(
            self.num_embeddings, get_tp_rank(), self.tensor_model_parallel_size
        )
        self.num_embeddings_per_partition = (
            self.vocab_end_index - self.vocab_start_index
        )

        with get_rng_tracer().rng_fork():
            self.embedding_weight = Parameter(
                initializer(
                    init=init_method,
                    shape=(self.num_embeddings_per_partition, self.embedding_dim),
                    dtype=init_type,
                ),
                name="embedding_weight",
            )
        self.reduce_from_mp_region = ReduceFromModelParallelRegion()
        self.reduce_scatter_to_sp_region = ReduceScatterToSequenceParallelRegion()
        self.max_index_per_partition = Tensor(self.num_embeddings_per_partition - 1, dtype=mstype.int32)
        self.reshape = ops.Reshape()
        self.gather = ops.Gather()

    def construct(self, x):
        """ construct. """
        if self.tensor_model_parallel_size > 1:
            displaced_x = mint.sub(x, self.vocab_start_index)
            down_truncated_x = mint.nn.functional.relu(displaced_x)
            truncated_x = mint.minimum(down_truncated_x, self.max_index_per_partition)
        else:
            truncated_x = x
        # Get the embeddings.
        output_parallel = self.gather(self.embedding_weight, truncated_x, 0)
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:
            input_mask = mint.eq(displaced_x, truncated_x)
            input_mask = self.reshape(input_mask, (input_mask.shape + (1,)))
            output_parallel = mint.mul(output_parallel, input_mask)

        if self.sequence_parallel:
            output_parallel = output_parallel.swapaxes(0, 1).contiguous()
            output = self.reduce_scatter_to_sp_region(output_parallel)
            output = output.swapaxes(0, 1).contiguous()
        else:
            # Reduce across all the model parallel devices.
            output = self.reduce_from_mp_region(output_parallel)
        return output

    # pylint: disable=W0613
    def _vocab_range_from_global_vocab_size(self, global_vocab_size, rank, world_size):
        if global_vocab_size % world_size != 0:
            raise ValueError(f"The vocabulary size is {global_vocab_size},"
                             f"which is not divisible by size of tensor parallel({world_size}).")
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    def sharded_state_dict(self):
        """provide the sharded state dict based on the config"""
        tp_size = get_tp_world_size()
        w_shard = (tp_size, 1)
        state_dict = {}
        state_dict[self.embedding_weight.name] = {'shape': self.embedding_weight.shape,
                                                  'shard': w_shard,
                                                  'opt_weight_shard_step': 0,
                                                  'opt_weight_shard_size': -1}

        return state_dict


class LlamaEmbedding(nn.Cell):
    def __init__(
            self,
            num_embeddings,
            embedding_dim,
            parallel_config=None,
            init_method="normal",
            init_type=mstype.float32,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding_weight = Parameter(
            initializer(
                init=init_method,
                shape=(self.num_embeddings, self.embedding_dim),
                dtype=init_type,
            ),
            name="embedding_weight",
        )
        self.gather = ops.Gather()

    def construct(self, input_ids):
        output = self.gather(self.weight, input_ids, 0)
        return output


class ParallelLlamaModel(LlamaPreTrainedModel):
    r"""
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]
    Args:
        config (): the config of network

    Returns:
        output (Tensor): the output of llama decoderlayer

    Examples:

    """

    def __init__(self, config):
        super().__init__(config)
        self.config = convert_model_config(config)
        self.use_past = config.use_past
        self.hidden_size = config.hidden_size
        self.is_first_iteration = True
        if config.parallel_config.vocab_emb_dp:
            self.tok_embeddings = LlamaEmbedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                parallel_config=config.parallel_config,
                init_method="normal",
                init_type=config.param_init_dtype,
            )
        else:
            self.tok_embeddings = VocabParallelLlamaEmbedding(num_embeddings=config.vocab_size,
                                                              embedding_dim=config.hidden_size,
                                                              parallel_config=config.parallel_config,
                                                              init_method="normal",
                                                              init_type=config.param_init_dtype)
        self.head_dim = config.hidden_size // config.num_heads
        self.num_layers = config.num_layers
        try:
            offset = get_pp_rank() * self.num_layers
        except:
            offset = 0
        self.layers = nn.SequentialCell(
            [ParallelLlamaTransformerLayer(config=config, layer_number=i + 1 + offset) for i in range(self.num_layers)]
        )
        self.freqs_mgr = FreqsMgr(head_dim=self.head_dim,
                                  seq_length=config.seq_length,
                                  max_position_embedding=config.max_position_embedding,
                                  rotary_dtype=config.rotary_dtype,
                                  theta=config.theta,
                                  scaling_factor=config.scaling_factor,
                                  extend_method=config.extend_method,
                                  parallel_config=config.parallel_config)
        self.casual_mask = LowerTriangularMaskWithDynamic(seq_length=config.seq_length,
                                                          compute_type=config.compute_dtype,
                                                          is_dynamic=True,
                                                          pad_token_id=config.pad_token_id,
                                                          use_flash_attention=config.use_flash_attention,
                                                          use_attn_mask_compression=config.use_attn_mask_compression)

        self.norm_out = LlamaRMSNorm(self.hidden_size, config.layernorm_epsilon,
                                     compute_type=config.layernorm_compute_dtype)
        self.post_norm = config.post_norm
        if self.post_norm:
            # final layernorm before output.
            self.final_norm = LlamaRMSNorm(self.hidden_size, config.layernorm_epsilon,
                                           compute_type=config.layernorm_compute_dtype)

    def construct(self, tokens: Tensor, batch_valid_length=None, batch_index=None, zactivate_len=None,
                  block_tables=None, slot_mapping=None, prefix_keys_values=None):
        """
        Forward of llama model.

        Args:
            input_ids (Tensor): the tokenized inputs with datatype int32
            attention_mask (Tensor):
        Returns:
            output (Tensor): the output of llama decoderlayer
        """
        mask = None
        bs, seq_len = tokens.shape
        if self.use_past:
            if self.is_first_iteration:
                freqs_cis = self.freqs_mgr.prefill(bs, seq_len)
                mask = self.casual_mask(tokens)  # mask: [bs, seq, seq]
            else:
                freqs_cis = self.freqs_mgr.increment(batch_valid_length)
        else:
            mask = self.casual_mask(tokens)
            freqs_cis = self.freqs_mgr(seq_len)
            if prefix_keys_values is not None:
                prefix_length = prefix_keys_values[0].shape[2]
                prefix_mask = ops.zeros((bs, 1, seq_len, prefix_length), dtype=mask.dtype)
                mask = ops.concat((prefix_mask, mask), -1)

        hidden_states = self.tok_embeddings(tokens)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        for layer in self.layers:
            prefix_kv = None  # No use_past
            hidden_states = layer(hidden_states, freqs_cis, mask, batch_valid_length=batch_valid_length,
                                  block_tables=block_tables,
                                  slot_mapping=slot_mapping, prefix_keys_values=prefix_kv)
        if self.post_norm:
            hidden_states = self.final_norm(hidden_states)

        output = self.norm_out(hidden_states)

        return output


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class ParallelLlamaForCausalLM(LlamaPreTrainedModel):
    r"""
    Provide llama training loss or logits through network.

    Args:
        config (LlamaConfig): The config of llama model.

    Returns:
        output: Tensor, the output of llama decoderlayer

    """

    @lazy_inline
    def __init__(self, config):
        super().__init__(config, auto_prefix=True)
        self.config = convert_model_config(config)
        self.ignore_token_id = config.ignore_token_id
        self.pad_token_id = config.pad_token_id
        self.use_past = config.use_past
        self.vocab_size = config.vocab_size
        self.is_first_iteration = True

        self.shape = ops.Shape()
        self.cast = ops.Cast()
        self.slice = ops.StridedSlice()
        self.not_equal = ops.NotEqual()
        self.mul = ops.Mul()
        self.add = ops.Add()
        self.ones = ops.Ones()
        self.gather = ops.Gather(1)
        self.sub_batch_valid_len = ops.Sub()
        self.model = ParallelLlamaModel(config=config)
        if config.parallel_config.vocab_emb_dp:
            self.lm_head = nn.Dense(
                in_channels=config.hidden_size,
                out_channels=config.vocab_size,
                weight_init="normal",
                has_bias=False,
                dtype=config.param_init_type,
            )
        else:
            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                output_size=config.vocab_size,
                config=config.parallel_config,
                bias=False,
                gather_output=True,
                param_init_type=config.param_init_dtype,
                compute_dtype=config.compute_dtype,
            )

        self.load_checkpoint(config)
        self.set_model_predict_config()
        self.predict_run_mode = get_predict_run_mode()

        self.use_past = config.use_past

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        if self.config.is_dynamic and "origin_inputs" in kwargs:
            input_ids = kwargs["origin_inputs"]
        return {
            "input_ids": Tensor(input_ids, mstype.int32)
        }

    # pylint: disable=W0613
    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        """Get Llama model input tuple for transform ckpt."""
        input_ids = Tensor(input_ids, mstype.int32)
        labels = Tensor(kwargs["labels"]) if "labels" in kwargs else None
        bs = input_ids.shape[0]
        slot_mapping = Tensor(np.ones(shape=tuple([bs])), mstype.int32)
        prefix_keys_values = Tensor(kwargs["prefix_keys_values"]) if "prefix_keys_values" in kwargs else None
        return input_ids, labels, None, None, None, None, None, None, None, None, None, slot_mapping, prefix_keys_values

    def set_dynamic_inputs(self, **kwargs):
        dynamic_input_ids = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_input_position = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_init_reset = Tensor([False], mstype.bool_)
        dynamic_batch_valid_length = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
        have_prefix_keys_values = getattr(kwargs, "have_prefix_keys_values", False)
        if have_prefix_keys_values:
            dynamic_prefix_keys_values = Tensor(shape=[2, None, None, None, None], dtype=mstype.float16)
            self.set_inputs(dynamic_input_ids, None, dynamic_input_position, None, None, None, dynamic_init_reset,
                            dynamic_batch_valid_length, None, None, dynamic_block_tables,
                            dynamic_slot_mapping, dynamic_prefix_keys_values)
        else:
            self.set_inputs(dynamic_input_ids, None, dynamic_input_position, None, None, None, dynamic_init_reset,
                            dynamic_batch_valid_length, None, None, dynamic_block_tables,
                            dynamic_slot_mapping, None)

    def add_flags_custom(self, is_first_iteration):
        """Add customized attributes for specific cells in the model."""
        self.add_flags(is_first_iteration=is_first_iteration)
        self.model.add_flags(is_first_iteration=is_first_iteration)
        for layer in self.model.layers:
            layer.add_flags(is_first_iteration=is_first_iteration)
            layer.attention.add_flags(is_first_iteration=is_first_iteration)

    # pylint: disable=W0613

    def construct(self, input_ids, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=True, batch_valid_length=None, batch_index=None, zactivate_len=None,
                  block_tables=None, slot_mapping=None, prefix_keys_values=None):
        """
        Forward of llama model.
        """
        # input_ids = input_ids[:, :-1].contiguous()
        batch_size, seq_len = input_ids.shape
        if self.training:
            tokens = input_ids[:, :seq_len - 1]
        else:
            tokens = input_ids
        if batch_valid_length is not None:
            batch_valid_length = batch_valid_length.reshape(-1, )
        output = self.model(tokens, batch_valid_length, batch_index, zactivate_len, block_tables,
                            slot_mapping, prefix_keys_values)
        pre_gather = (not self.use_past or self.is_first_iteration) and batch_valid_length is not None
        if pre_gather:
            output = self.gather(output, batch_valid_length - 1, 1)
        logits = self.lm_head(output)
        input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), mstype.float32)
        if labels is None:
            labels = self.slice(input_ids, (0, 1), (batch_size, seq_len), (1, 1))
        else:
            if labels.ndim > 1:
                if self.training:
                    labels = self.slice(labels, (0, 1), (batch_size, seq_len), (1, 1))
                label_mask = self.cast(self.not_equal(labels, self.ignore_token_id), mstype.float32)
                input_mask = self.mul(input_mask, label_mask)

        if not self.training:
            logits = self.cast(logits, mstype.float32)
            if self.predict_run_mode:
                return logits
            return logits, tokens, input_mask

        # DO NOT NEED LOSS

        # if logits.ndim > 2:
        #     logits = self.reshape(logits, (-1, logits.shape[-1]))
        # logits = self.cast(logits, mstype.float32)
        # labels = self.reshape(labels, (-1,))
        # input_mask = self.reshape(input_mask, (-1,))
        # loss = self.loss(logits, labels, input_mask)
        # return loss

    def kvcache(self, layer_idx):
        key_cache = self.model.layers[layer_idx].attention.infer_attention.paged_attention_mgr.key_cache
        value_cache = self.model.layers[layer_idx].attention.infer_attention.paged_attention_mgr.value_cache
        return key_cache, value_cache
