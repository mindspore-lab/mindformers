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
    """Generate initial parameters for SharedKVCrossAttention"""
    np.random.seed(1)
    shape = (seq_length, batch_size, config.num_attention_heads, config.kv_channels)
    rotary_pos_emb = RotaryEmbedding(
        kv_channels=4,
        use_eod_reset=config.use_eod_reset
    )
    params = {
        "hidden_states": ms.Tensor(0.01 * np.random.randn(seq_length, batch_size, config.hidden_size), ms.bfloat16),
        "attention_mask": ms.Tensor(np.triu(np.ones((2, 2, 2, 2), dtype=np.int8), k=1), dtype=ms.uint8),
        "rotary_pos_emb": rotary_pos_emb(seq_length),
        "sharded_key": ms.Tensor(0.01 * np.random.randn(*shape), ms.bfloat16),
        "sharded_value": ms.Tensor(0.01 * np.random.randn(*shape), ms.bfloat16)
    }
    weight_shape = (hidden_size, q_hidden_size)
    weight_dict = {
        "linear_q.weight": ms.Tensor(0.01 * np.random.randn(*weight_shape), ms.bfloat16),
        "linear_proj.weight": ms.Tensor(0.01 * np.random.randn(*weight_shape), ms.bfloat16)
    }
    return params, weight_dict

def get_init_block_params(config: TransformerConfig, seq_length=2, batch_size=2, hidden_size=8, q_hidden_size=8):
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
    weight_dict = {
        "layers.0.cross_attention.linear_q.weight": ms.Tensor(0.01 * np.random.randn(*weight_shape), ms.bfloat16),
        "layers.0.cross_attention.linear_proj.weight": ms.Tensor(0.01 * np.random.randn(*weight_shape), ms.bfloat16),
        "adapter.k_proj.weight": ms.Tensor(0.01 * np.random.randn(*weight_shape), ms.bfloat16),
        "adapter.v_proj.weight": ms.Tensor(0.01 * np.random.randn(*weight_shape), ms.bfloat16),
        "layers.0.mlp.linear_fc1.weight": ms.Tensor(np.concatenate(
            (linear_x0_weight, linear_x1_weight), axis=0), ms.bfloat16
        ),
        "layers.0.mlp.linear_fc2.weight": ms.Tensor(0.01 * np.random.randn(8, 32), ms.bfloat16),
        "adapter.kv_layer_norm.weight": ms.Tensor(0.01 * np.random.randn(8), ms.bfloat16),
        "layers.0.input_layernorm.weight": ms.Tensor(0.01 * np.random.randn(8), ms.bfloat16),
        "final_layernorm.weight": ms.Tensor(0.01 * np.random.randn(8), ms.bfloat16)
    }
    return params, weight_dict



GOLDEN_DATA = {
    "output_attn": np.array(
        [[[-5.602836608886719e-05, 0.0007781982421875,
           -0.0003643035888671875, -1.5497207641601562e-05,
           -0.0001850128173828125, 0.000446319580078125,
           0.0001506805419921875, 0.000263214111328125],
          [-1.8477439880371094e-05, -0.0004482269287109375,
           0.0002574920654296875, -4.696846008300781e-05,
           0.000591278076171875, -0.00055694580078125,
           0.000370025634765625, 2.658367156982422e-05]],
         [[-4.1484832763671875e-05, 0.0004138946533203125,
           -0.00019741058349609375, -2.002716064453125e-05,
           -0.00015163421630859375, 0.0002689361572265625,
           0.00010347366333007812, 0.00011301040649414062],
          [-3.8623809814453125e-05, -0.00038909912109375,
           0.00018787384033203125, -6.938353180885315e-08,
           0.000392913818359375, -0.0002079010009765625,
           0.000232696533203125, 6.4849853515625e-05]]],
        dtype=np.float32),
    "output_block": np.array(
        [[[0.016235, - 0.006104, - 0.005249, - 0.010742,
           0.008667, - 0.023071, 0.017456, - 0.007599],
          [0.003204, - 0.002502, 0.014709, - 0.020630,
           - 0.003220, - 0.003845, 0.011353, - 0.010986]],
         [[-0.001724, - 0.008728, 0.000395, 0.005829,
           - 0.010925, 0.011475, 0.009033, 0.005035],
          [0.009033, - 0.006836, - 0.001244, - 0.009338,
           - 0.002640, 0.005280, - 0.006897, - 0.003967]]],
        dtype=np.float32)
}

GPU_DATA = {
    "output_attn": np.array(
        [[[-5.5512e-05, 7.7754e-04, -3.6442e-04, -1.5505e-05, -1.8534e-04, 4.4649e-04, 1.4995e-04, 2.6433e-04],
          [-4.1193e-05, 4.1335e-04, -1.9753e-04, -2.0061e-05, -1.5163e-04, 2.6995e-04, 1.0301e-04, 1.1354e-04]],
         [[-1.8618e-05, -4.4912e-04,  2.5768e-04, -4.6789e-05,  5.9355e-04, -5.5917e-04,  3.7121e-04,  2.6458e-05],
          [-3.8943e-05, -3.8902e-04,  1.8821e-04, -1.7298e-07,  3.9256e-04, -2.0821e-04,  2.3300e-04,  6.4698e-05]]],
        dtype=np.float32),
    "output_block": np.array(
        [[[0.0162, -0.0061, -0.0053, -0.0107,
           0.0087, -0.0230, 0.0174, -0.0076],
          [-0.0017, -0.0088, 0.0004, 0.0058,
           -0.0110, 0.0114, 0.0090, 0.0050]],
         [[0.0032, -0.0025, 0.0146, -0.0206,
           -0.0032, -0.0038, 0.0113, -0.0110],
          [0.0090, -0.0068, -0.0012, -0.0094,
           -0.0027, 0.0053, -0.0069, -0.0040]]],
        dtype=np.float32)
}
