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
"""ShardKeyValue Adapter"""
from mindspore import nn
from mindspore.ops import auto_generate as aclnn_ops
from mindformers.parallel_core.training_graph.base_models.common.embeddings.rope_utils import ApplyRotaryPosEmb
from mindformers.parallel_core.training_graph.tensor_parallel.layers import ColumnParallelLinear
from mindformers.parallel_core.training_graph.transformer.norm import RMSNorm
from mindformers.parallel_core.transformer_config import TransformerConfig


class SharedKVAdapter(nn.Cell):
    """
    Class for sharded key and sharded value for the sharded cross attention.

    This class defines the configuration for preparing the sharded cross attention.

    Args:
        config (dict): Configuration.

    Supported Platforms:
    ``Ascend``
    """
    def __init__(
            self,
            config: TransformerConfig
    ):
        super().__init__()
        self.shape = aclnn_ops.Shape()
        self.reshape = aclnn_ops.Reshape()
        self.seq_length_in_cfg = config.seq_length
        self.init_method = config.init_method
        self.head_dim = config.kv_channels
        self.num_heads = config.num_attention_heads
        self.use_gqa = config.num_query_groups < config.num_attention_heads
        self.kv_num_heads = config.num_query_groups if self.use_gqa else self.num_heads
        self.kv_hidden_size = self.head_dim * self.kv_num_heads
        self.parallel_config = config
        self.apply_rotary_pos_emb = ApplyRotaryPosEmb(self.parallel_config)


        self.k_proj = ColumnParallelLinear(config.hidden_size,
                                           self.kv_hidden_size,
                                           config=config,
                                           init_method=self.init_method,
                                           bias=config.add_bias_linear or config.add_qkv_bias)
        self.v_proj = ColumnParallelLinear(config.hidden_size,
                                           self.kv_hidden_size,
                                           config=config,
                                           init_method=self.init_method,
                                           bias=config.add_bias_linear or config.add_qkv_bias)
        self.kv_layer_norm = RMSNorm(config=config,
                                     dim=config.hidden_size,
                                     eps=config.layernorm_epsilon)

    def construct(self, hidden_states, rotary_pos_emb):
        """
        Perform a forward pass through the transformer layer.

        This method implements the core computation of a transformer layer, including
        self-attention, cross-attention (if applicable), and feed-forward operations.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
            b is batch size, and h is hidden size.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
            key (Tensor): A tensor of shape [s, b, n, d].
            value (Tensor): A tensor of shape [s, b, n, d],
            otherwise None.
        """
        seq_len, bs, _ = self.shape(hidden_states)
        x_norm = self.kv_layer_norm(hidden_states)
        key, _ = self.k_proj(x_norm)
        value, _ = self.v_proj(x_norm)
        key = self.reshape(
            key, (seq_len, bs, self.kv_num_heads, self.head_dim))
        value = self.reshape(
            value, (seq_len, bs, self.kv_num_heads, self.head_dim))
        if rotary_pos_emb is not None:
            key = self.apply_rotary_pos_emb(key, rotary_pos_emb)
        return key, value
