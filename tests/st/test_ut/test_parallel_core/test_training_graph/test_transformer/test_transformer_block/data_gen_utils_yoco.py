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
"""Data for SharedKVCrossAttention UT."""

import numpy as np
import mindspore as ms
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.training_graph.base_models.common.embeddings.rotary_pos_embedding import RotaryEmbedding


def get_init_params(config: TransformerConfig, seq_length=2, batch_size=2, hidden_size=8, q_hidden_size=8):
    """Generate initial parameters for TransformerBlock with SharedKVCrossAttention"""
    np.random.seed(1)
    rotary_pos_emb = RotaryEmbedding(
        kv_channels=4,
        use_eod_reset=config.use_eod_reset
    )
    params = {
        "hidden_states": ms.Tensor(0.01 * np.random.randn(seq_length, batch_size, config.hidden_size), ms.bfloat16),
        "attention_mask": ms.Tensor(np.triu(np.ones((2, 2, 2, 2), dtype=np.int8), k=1), dtype=ms.uint8),
        "rotary_pos_emb": rotary_pos_emb(seq_length)
    }
    linear_x0_weight = 0.01 * np.random.randn(32, 8)
    linear_x1_weight = 0.01 * np.random.randn(32, 8)
    weight_shape = (hidden_size, q_hidden_size)
    linear_q_weight = 0.01 * np.random.randn(*weight_shape)
    linear_k_weight = 0.01 * np.random.randn(*weight_shape)
    linear_v_weight = 0.01 * np.random.randn(*weight_shape)
    linear_proj_weight = ms.Tensor(0.01 * np.random.randn(*weight_shape), ms.bfloat16)
    cross_linear_q_weight = 0.01 * np.random.randn(*weight_shape)
    cross_linear_k_weight = 0.01 * np.random.randn(*weight_shape)
    cross_linear_v_weight = 0.01 * np.random.randn(*weight_shape)
    linear_fc2_weight = ms.Tensor(0.01 * np.random.randn(8, 32), ms.bfloat16)
    kv_layer_norm_weight = ms.Tensor(0.01 * np.random.randn(8), ms.bfloat16)
    input_layernorm_weight = ms.Tensor(0.01 * np.random.randn(8), ms.bfloat16)
    pre_mlp_layernorm_weight= ms.Tensor(0.01 * np.random.randn(8), ms.bfloat16)
    weight_dict = {
        "layers.0.self_attention.linear_qkv.weight": ms.Tensor(np.concatenate(
            (linear_q_weight, linear_k_weight, linear_v_weight), axis=0),
            ms.bfloat16),
        "layers.0.self_attention.linear_proj.weight": linear_proj_weight,
        "layers.1.cross_attention.linear_q.weight": ms.Tensor(cross_linear_q_weight, ms.bfloat16),
        "layers.1.cross_attention.linear_proj.weight": linear_proj_weight,
        "layers.0.pre_mlp_layernorm.weight": pre_mlp_layernorm_weight,
        "layers.1.pre_mlp_layernorm.weight": pre_mlp_layernorm_weight,
        "adapter.k_proj.weight": ms.Tensor(cross_linear_k_weight, ms.bfloat16),
        "adapter.v_proj.weight": ms.Tensor(cross_linear_v_weight, ms.bfloat16),
        "layers.0.mlp.linear_fc1.weight": ms.Tensor(np.concatenate(
            (linear_x0_weight, linear_x1_weight), axis=0), ms.bfloat16
        ),
        "layers.1.mlp.linear_fc1.weight": ms.Tensor(np.concatenate(
            (linear_x0_weight, linear_x1_weight), axis=0), ms.bfloat16
        ),
        "layers.0.mlp.linear_fc2.weight": linear_fc2_weight,
        "layers.1.mlp.linear_fc2.weight": linear_fc2_weight,
        "adapter.kv_layer_norm.weight": kv_layer_norm_weight,
        "layers.0.input_layernorm.weight": input_layernorm_weight,
        "layers.1.pre_cross_attn_layernorm.weight": input_layernorm_weight,
        "final_layernorm.weight": ms.Tensor(0.01 * np.random.randn(8), ms.bfloat16)
    }
    return params, weight_dict



GOLDEN_DATA = {
    "output": np.array(
        [[[-0.003723, -0.000454, 0.004303, 0.009277,
           0.010315, -0.019531, 0.011353, -0.010132],
          [-0.000889, -0.000224, -0.014465, 0.021729,
           -0.004639, -0.003937, 0.008972, -0.017822]],
         [[0.000641, -0.001076, -0.000565, -0.008301,
           -0.021484, 0.015869, 0.009705, 0.010986],
          [-0.004028, -0.000984, 0.001953, 0.015747,
           -0.006165, 0.008728, -0.008789, -0.010254]]],
        dtype=np.float32)
}

GPU_DATA = {
    "output": np.array(
        [[[-0.0037, -0.0005, 0.0043, 0.0093,
           0.0103, -0.0195, 0.0114, -0.0102],
          [-0.0009, -0.0002, -0.0145, 0.0218,
           -0.0046, -0.0040, 0.0090, -0.0179]],
         [[0.0006, -0.0011, -0.0006, -0.0083,
           -0.0215, 0.0159, 0.0096, 0.0110],
          [-0.0040, -0.0010, 0.0020, 0.0158,
           -0.0062, 0.0087, -0.0087, -0.0103]]],
        dtype=np.float32)
}
