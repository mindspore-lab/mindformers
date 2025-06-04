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
"""DotProduct Attention Layer"""
__all__ = ['DotProductAttention']

import math
from mindspore import Tensor, mint, nn

from mindformers.parallel_core.inference.utils import get_attn_mask_func
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.inference.utils import get_tp_world_size
from mindformers.parallel_core.inference.transformer.fused_softmax import FusedScaleMaskSoftmax
from mindformers.parallel_core.inference.utils import divide


class DotProductAttention(nn.Cell):
    """
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

    def __init__(self, config: TransformerConfig, layer_number, attn_mask_type=None):
        super().__init__()
        if attn_mask_type:
            raise NotImplementedError(
                "For DotProductAttention, `attn_mask_type` is not supported for now."
            )
        if config.context_parallel_size > 1:
            raise NotImplementedError(
                "For DotProductAttention, 'context_parallel_size' is not supported for now."
            )

        self.config = config
        self.layer_number = max(1, layer_number)
        self.compute_dtype = self.config.compute_dtype
        self.softmax_compute_dtype = self.config.softmax_compute_dtype

        self.apply_query_key_layer_scaling = self.config.apply_query_key_layer_scaling
        self.num_heads = self.config.num_attention_heads
        self.query_projection_size = self.config.hidden_size
        self.hidden_size_per_attention_head = getattr(config, 'kv_channels', divide(
            self.query_projection_size, self.num_heads
        ))
        self.num_query_groups = (self.num_heads
                                 if config.num_query_groups is None else
                                 config.num_query_groups)
        self.use_gqa = (self.num_heads != self.num_query_groups)
        if self.use_gqa:
            self.repeat_num = divide(self.num_heads, self.num_query_groups)

        self.tp_group_size = get_tp_world_size()
        self.hidden_size_per_partition = divide(self.query_projection_size,
                                                self.tp_group_size)

        coeff = None
        self.softmax_scale = Tensor(1.0 / math.sqrt(self.hidden_size_per_attention_head), dtype=self.compute_dtype)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.softmax_scale /= coeff

        self.mask_func = get_attn_mask_func(self.config.mask_func_type)
        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.mask_func, softmax_compute_type=self.softmax_compute_dtype)

    def construct(self, query_layer, key_layer, value_layer, attention_mask):
        """Forward process of the CoreAttention."""
        bs, seq_len, _ = query_layer.shape
        # [B, S, H] --> [B, S, N, D]
        query_layer = query_layer.reshape(bs, seq_len, -1, self.hidden_size_per_attention_head)
        key_layer = key_layer.reshape(bs, seq_len, -1, self.hidden_size_per_attention_head)
        value_layer = value_layer.reshape(bs, seq_len, -1, self.hidden_size_per_attention_head)
        # [B, S, N_kv, D] --> [B, S, N, D]
        if self.use_gqa:
            key_layer = mint.repeat_interleave(key_layer,
                                               repeats=self.repeat_num,
                                               dim=2)
            value_layer = mint.repeat_interleave(value_layer,
                                                 repeats=self.repeat_num,
                                                 dim=2)
        # [B, S, N, D] --> [B, N, S, D]
        query_layer = mint.transpose(query_layer, -3, -2)
        key_layer = mint.transpose(key_layer, -3, -2)
        value_layer = mint.transpose(value_layer, -3, -2)
        # [B, N, S, D] --> [B * N, S, D]
        query_layer = query_layer.reshape(-1, seq_len, self.hidden_size_per_attention_head)
        key_layer = key_layer.reshape(-1, seq_len, self.hidden_size_per_attention_head)
        value_layer = value_layer.reshape(-1, seq_len, self.hidden_size_per_attention_head)

        # score shape: [B * N, S_q, S_k]
        score = mint.bmm(query_layer, mint.transpose(key_layer, -2, -1))
        score = mint.mul(score, self.softmax_scale)

        # attention scores and attention mask [B * N, S_q, S_k]
        attention_probs = self.scale_mask_softmax(score, attention_mask)

        # [B * N, S_q, S_k] * [B * N, S_v, D] -> [B * N, S_q, D]
        core_attn_out = mint.bmm(attention_probs, value_layer)
        # [B * N, S_q, D] -> [B, N, S_q, D]
        core_attn_out = core_attn_out.reshape(bs, -1, seq_len, self.hidden_size_per_attention_head)

        core_attn_out = mint.transpose(core_attn_out, -3, -2).reshape(
            bs, seq_len, self.hidden_size_per_partition)

        return core_attn_out
