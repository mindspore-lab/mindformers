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
"""Attention."""
__all__ = [
    'SelfAttentionSubmodules',
    'Attention',
    'SelfAttention'
]

from abc import abstractmethod
from dataclasses import dataclass
import math
from typing import Union

from mindspore import mint, nn, ops

from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.inference.transformer.enums import AttnMaskType
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.inference.transformer.rotary_embedding import RotaryEmbedding
from mindformers.parallel_core.inference.utils import (
    get_tp_world_size,
    divide,
)


@dataclass
class SelfAttentionSubmodules:
    """Configuration class for specifying the submodules of a self-attention."""

    linear_qkv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None


class Attention(nn.Cell):
    """
    Attention block.

    Args:
        config (dict): Configuration for the transformer model.
        submodules (Union[SelfAttentionSubmodules]): submodules of Attention,
            currently only submodules of SelfAttention is implemented.
        layer_number (int): Number which indicates the index of this transformer layer in the whole transformer block.
        attn_mask_type (AttnMaskType): Type of attention mask used in the self-attention module, of type AttnMaskType.
        attention_type (str): Type of attention used in the self-attention module, default value is None.
        cp_comm_type (str): Type of communication used in the self-attention module, default value is None.


    Inputs:
        - **hidden_states** (Tensor) - inputs.
        - **attention_mask** (Union[Tensor, None]) - The attention mask tensor.
        - **key_value_states** (Union[Tensor, None]) - The attention mask tensor.
        - **rotary_pos_cos** (Tensor) - The precompute freqs cos for rotary position embedding
        - **rotary_pos_sin** (Tensor) - The precompute freqs sin for rotary position embedding
        - **position_ids** (Union[Tensor, None]) - The position tensor.
        - **batch_valid_length** (Tensor) -  In incremental inference, a tensor used for calculating the index
        - **block_tables** (Tensor) - The block mapping table with data type of int32.
        - **slot_mapping** (Tensor) - Store token cache physical slot index.
          of the previous step. It is of int32 type and has a shape of [batch_size].
        - **q_seq_lens** (Tensor) - Used by flash attention.
        - **actual_seq_qlen** (Union[List[int64], Tuple[int64], None]) - Size of query corresponding to each batch,
          array with increasing values and the last value equal to T1.
        - **actual_seq_kvlen** (Union[List[int64], Tuple[int64], None]) - Size of key and value corresponding to each
          batch, array with increasing values and the last value equal to T2.
        - **context_lens_tensor** (Tensor) - The context length of each sequence with data type of int32.
        - **key_cache** (Tensor, optional) - Key cache for incremental inference.
        - **value_cache** (Tensor, optional) - Value cache for incremental inference.

    Outputs:
        - **output** (Tensor) - Tensor of shape :math:`(B, S, H)`.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(
            self,
            config: TransformerConfig,
            submodules: Union[SelfAttentionSubmodules],
            layer_number: int,
            attn_mask_type: AttnMaskType = None,
            attention_type: str = None,
            cp_comm_type: str = None
    ):
        super().__init__(config)

        if attn_mask_type:
            raise NotImplementedError(
                "For Attention, 'attn_mask_type' is not supported for now."
            )
        if attention_type:
            raise NotImplementedError(
                "For Attention, 'attention_type' is not supported for now."
            )
        if cp_comm_type:
            raise NotImplementedError(
                "For Attention, 'cp_comm_type' is not supported for now."
            )

        self.config = config
        self.submodules = submodules
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type

        self.params_dtype = self.config.params_dtype
        self.compute_dtype = self.config.compute_dtype
        self.is_prefill = True

        self.num_heads = self.config.num_attention_heads
        self.num_query_groups = (self.num_heads
                                 if config.num_query_groups is None else
                                 config.num_query_groups)
        self.hidden_size_per_attention_head = getattr(config, 'kv_channels', divide(
            self.config.hidden_size, self.num_heads
        ))
        self.query_projection_size = self.hidden_size_per_attention_head * self.num_heads
        self.kv_projection_size = self.hidden_size_per_attention_head * self.num_query_groups
        self.n_rep = divide(self.num_heads, self.num_query_groups)

        self.use_flash_attention = self.config.use_flash_attention
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)

        self.tp_group_size = get_tp_world_size()

        self.cast = ops.Cast()

        self.num_attention_heads_per_partition = divide(self.num_heads,
                                                        self.tp_group_size)

        self.use_gqa = (self.num_heads != self.num_query_groups)

        if self.use_gqa:
            self._check_gqa_valid()
            self.num_query_groups_per_partition = divide(self.num_query_groups,
                                                         self.tp_group_size)
            self.repeat_num = divide(self.num_heads, self.num_query_groups)
        else:
            self.num_query_groups_per_partition = self.num_attention_heads_per_partition

        self.hidden_size_per_partition = divide(self.query_projection_size,
                                                self.tp_group_size)
        self.kv_hidden_size_per_partition = divide(self.kv_projection_size,
                                                   self.tp_group_size)

        if self.use_flash_attention:
            self.core_attention = build_module(
                submodules.core_attention,
                head_num=self.num_attention_heads_per_partition,
                head_dim=self.hidden_size_per_attention_head,
                kv_head_num=self.num_query_groups_per_partition,
                scale_value=1.0 / self.norm_factor,
                next_tokens=0,
            )
        else:
            self.core_attention = build_module(submodules.core_attention,
                                               config=self.config,
                                               layer_number=self.layer_number)

        self.rotary_embedding = RotaryEmbedding(kv_channels=self.hidden_size_per_attention_head,
                                                rotary_cos_format=2)

        self.linear_proj = build_module(
            submodules.linear_proj,
            self.query_projection_size,
            self.config.hidden_size,
            input_is_parallel=True,
            config=self.config,
            bias=self.config.add_bias_linear,
            transpose_b=True,
            compute_dtype=self.compute_dtype,
        )

    def _check_gqa_valid(self):
        """check whether the config is valid for grouped-query-attention"""
        if self.num_heads % self.num_query_groups != 0:
            raise ValueError(f"num_heads must be divisible by kv_num_heads, "
                             f"but got num_heads {self.num_heads} "
                             f"and kv_num_heads {self.num_query_groups}")
        if self.num_query_groups % self.tp_group_size != 0:
            raise ValueError(
                f"kv_num_heads must be divisible by tp_group_size, "
                f"but got kv_num_heads {self.num_query_groups} "
                f"and tp_group_size {self.tp_group_size}")

    def construct(self,
                  hidden_states,
                  attention_mask,
                  key_valus_states=None,
                  rotary_pos_cos=None,
                  rotary_pos_sin=None,
                  position_ids=None,
                  batch_valid_length=None,
                  block_tables=None,
                  slot_mapping=None,
                  q_seq_lens=None,
                  actual_seq_qlen=None,
                  actual_seq_kvlen=None,
                  context_lens_tensor=None,
                  key_cache=None,
                  value_cache=None):
        """Forward process of the SelfAttention."""
        ori_dtype = hidden_states.dtype

        if key_valus_states and position_ids:
            pass

        # apply query, key, value projection
        query, key, value = self.get_query_key_value_tensors(hidden_states)

        if rotary_pos_cos is not None and rotary_pos_sin is not None:
            query, key = self.rotary_embedding(query, key, rotary_pos_cos,
                                               rotary_pos_sin,
                                               batch_valid_length)

        if self.use_flash_attention:
            core_attn_out = self.core_attention(
                query=query,
                key=key,
                value=value,
                slot_mapping=slot_mapping,
                block_tables=block_tables,
                batch_valid_length=batch_valid_length,
                context_lens_tensor=context_lens_tensor,
                q_seq_lens=q_seq_lens,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
                attn_mask=attention_mask,
                key_cache=key_cache,
                value_cache=value_cache)
        else:
            core_attn_out = self.core_attention(query, key, value, attention_mask)

        # apply output projection
        output = self.linear_proj(core_attn_out)
        output = self.cast(output, ori_dtype)
        return output


    @abstractmethod
    def get_query_key_value_tensors(self, hidden_states):
        """
        This method needs to be implemented based on whether the derived class
        is "self-attn" or "cross-attn".
        """


class SelfAttention(Attention):
    """
    SelfAttention block.

    Args:
        config (dict): Configuration for the transformer model.
        submodules (SelfAttentionSubmodules): submodules of SelfAttention
        layer_number (int): Number which indicates the index of this transformer layer in the whole transformer block.
        attn_mask_type (AttnMaskType): Type of attention mask used in the self-attention module, of type AttnMaskType.
        cp_comm_type (str): Type of communication used in the self-attention module, default value is None.

    """

    def __init__(self,
                 config: TransformerConfig,
                 submodules: SelfAttentionSubmodules,
                 layer_number: int,
                 attn_mask_type: AttnMaskType = None,
                 cp_comm_type: str = None):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=None,
            cp_comm_type=cp_comm_type
        )

        self.linear_qkv = build_module(
            self.submodules.linear_qkv,
            self.config.hidden_size,
            self.hidden_size_per_attention_head,
            self.num_heads,
            self.num_query_groups,
            config=self.config,
            bias=self.config.add_qkv_bias,
            gather_output=False,
            transpose_b=True,
            compute_dtype=self.compute_dtype,
        )

        if submodules.q_layernorm is not None:
            self.q_layernorm = build_module(
                self.submodules.q_layernorm,
                config=config,
                hidden_size=self.hidden_size_per_attention_head,
                eps=self.config.layernorm_epsilon
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                self.submodules.k_layernorm,
                config=config,
                hidden_size=self.hidden_size_per_attention_head,
                eps=self.config.layernorm_epsilon
            )
        else:
            self.k_layernorm = None

        self.cast = ops.Cast()

    def get_query_key_value_tensors(self, hidden_states):
        qkv = self.cast(self.linear_qkv(hidden_states), self.compute_dtype)
        query, key, value = mint.split(qkv,
                                       (self.hidden_size_per_partition,
                                        self.kv_hidden_size_per_partition,
                                        self.kv_hidden_size_per_partition), -1)

        if self.q_layernorm is not None:
            orig_query_shape = query.shape
            query = self.q_layernorm(query.reshape(hidden_states.shape[:-1] +
                                                   (-1, self.hidden_size_per_attention_head,)))
            query = query.reshape(orig_query_shape)


        if self.k_layernorm is not None:
            orig_query_shape = key.shape
            key = self.k_layernorm(key.reshape(hidden_states.shape[:-1] +
                                               (-1, self.hidden_size_per_attention_head,)))
            key = key.reshape(orig_query_shape)

        return query, key, value
