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
"""Generate data for test."""
import numpy as np
import mindspore as ms
from mindspore.ops import auto_generate as aclnn_ops
from mindformers.parallel_core.training_graph.base_models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from mindformers.parallel_core.transformer_config import MLATransformerConfig

reshape = aclnn_ops.Reshape()

def get_init_params(config: MLATransformerConfig, seq_length=8, batch_size=2):
    """Generate SBND-format input tensors for FlashAttention."""
    np.random.seed(1)
    shape = (seq_length, batch_size, config.num_attention_heads, config.kv_channels)
    attn_mask = ms.Tensor(np.triu(np.ones((2048, 2048), dtype=np.int8), k=1), dtype=ms.uint8)

    return {
        "query": ms.tensor(0.01 * np.random.randn(*shape), ms.bfloat16),
        "key": ms.tensor(0.01 * np.random.randn(*shape), ms.bfloat16),
        "value": ms.tensor(0.01 * np.random.randn(*shape), ms.bfloat16),
        "attention_mask": ms.tensor(attn_mask)
    }

def get_init_tnd_params(config: MLATransformerConfig, seq_length=8, batch_size=2):
    """Generate TND-format input tensors for FlashAttention."""
    np.random.seed(1)
    shape = (seq_length, batch_size, config.num_attention_heads, config.kv_channels)
    new_shape = (batch_size * seq_length, config.num_attention_heads, config.kv_channels)
    attn_mask = ms.Tensor(np.triu(np.ones((2048, 2048), dtype=np.int8), k=1), dtype=ms.uint8)
    query = ms.tensor(0.01 * np.random.randn(*shape), ms.bfloat16)
    key = ms.tensor(0.01 * np.random.randn(*shape), ms.bfloat16)
    value = ms.tensor(0.01 * np.random.randn(*shape), ms.bfloat16)
    return {
        "query": reshape(query.transpose(1, 0, 2, 3), new_shape),
        "key": reshape(key.transpose(1, 0, 2, 3), new_shape),
        "value": reshape(value.transpose(1, 0, 2, 3), new_shape),
        "attention_mask": ms.tensor(attn_mask),
        "actual_seq_qlen": ms.Tensor([8, 16], ms.int),
        "actual_seq_kvlen": ms.Tensor([8, 16], ms.int)
    }


def get_init_attn_params(config: MLATransformerConfig, seq_length=2, batch_size=2, hidden_size=8, q_hidden_size=8):
    """Generate initial parameters for SlidingWindowAttention"""
    np.random.seed(1)

    rotary_pos_emb = RotaryEmbedding(
        kv_channels=4,
        use_eod_reset=config.use_eod_reset
    )
    attn_mask = ms.Tensor(np.triu(np.ones((2048, 2048), dtype=np.int8), k=1), dtype=ms.uint8)
    params = {
        "hidden_states": ms.Tensor(0.01 * np.random.randn(seq_length, batch_size, config.hidden_size), ms.bfloat16),
        "attention_mask": ms.tensor(attn_mask),
        "rotary_pos_emb": rotary_pos_emb(seq_length)
    }
    weight_shape = (hidden_size, q_hidden_size)
    linear_q_weight = 0.01 * np.random.randn(*weight_shape)
    linear_k_weight = 0.01 * np.random.randn(*weight_shape)
    linear_v_weight = 0.01 * np.random.randn(*weight_shape)
    weight_dict = {
        "linear_qkv.weight": ms.Tensor(np.concatenate(
            (linear_q_weight, linear_k_weight, linear_v_weight), axis=0),
            ms.bfloat16),
        "linear_proj.weight": ms.Tensor(0.01 * np.random.randn(*weight_shape), ms.bfloat16)
    }
    return params, weight_dict

def get_init_block_params(config: MLATransformerConfig, seq_length=2, batch_size=2, hidden_size=8, q_hidden_size=8):
    """Generate initial parameters for TransformerBlock with SlidingWindowAttention"""
    np.random.seed(1)

    rotary_pos_emb = RotaryEmbedding(
        kv_channels=4,
        use_eod_reset=config.use_eod_reset
    )
    attn_mask = ms.Tensor(np.triu(np.ones((2048, 2048), dtype=np.int8), k=1), dtype=ms.uint8)
    params = {
        "hidden_states": ms.Tensor(0.01 * np.random.randn(seq_length, batch_size, config.hidden_size), ms.bfloat16),
        "attention_mask": ms.tensor(attn_mask),
        "rotary_pos_emb": rotary_pos_emb(seq_length)
    }
    weight_shape = (hidden_size, q_hidden_size)
    linear_q_weight = 0.01 * np.random.randn(*weight_shape)
    linear_k_weight = 0.01 * np.random.randn(*weight_shape)
    linear_v_weight = 0.01 * np.random.randn(*weight_shape)
    linear_proj_weight = ms.Tensor(0.01 * np.random.randn(*weight_shape), ms.bfloat16)
    linear_x0_weight = 0.01 * np.random.randn(32, 8)
    linear_x1_weight = 0.01 * np.random.randn(32, 8)
    weight_dict = {
        "layers.0.self_attention.linear_qkv.weight": ms.Tensor(np.concatenate(
            (linear_q_weight, linear_k_weight, linear_v_weight), axis=0),
            ms.bfloat16),
        "layers.0.self_attention.linear_proj.weight": linear_proj_weight,
        "layers.0.mlp.linear_fc1.weight": ms.Tensor(np.concatenate(
            (linear_x0_weight, linear_x1_weight), axis=0),
            ms.bfloat16),
        "layers.0.mlp.linear_fc2.weight": ms.Tensor(0.01 * np.random.randn(8, 32), ms.bfloat16),
        "layers.0.input_layernorm.weight": ms.Tensor(0.01 * np.random.randn(8), ms.bfloat16),
        "final_layernorm.weight": ms.Tensor(0.01 * np.random.randn(8), ms.bfloat16)

    }
    return params, weight_dict


def get_golden() -> dict[str, np.ndarray]:
    """Generate golden data for test."""
    output_bnsd = np.array(
        [[[-0.011963, 0.008606, - 0.001808, - 0.006042],
          [-0.012329, 0.005493, 0.007935, - 0.006226]],
         [[-0.003372, - 0.001404, 0.003098, - 0.002792],
          [-0.007111, 0.002243, 0.008301, 0.000641]],
         [[-0.000488, - 0.000479, 0.002319, 0.000206],
          [-0.003967, 0.003769, 0.004517, - 0.007660]],
         [[0.002228, 0.005096, 0.002853, - 0.000095],
          [-0.003311, 0.002533, 0.003418, - 0.008545]],
         [[0.000751, 0.002090, 0.002777, - 0.000668],
          [-0.001663, 0.001678, 0.004700, - 0.006409]],
         [[0.004272, - 0.001411, 0.001236, 0.000950],
          [0.002823, 0.000984, 0.003998, - 0.005737]],
         [[0.005554, - 0.001617, 0.002029, 0.000355],
          [0.000610, 0.001289, 0.004150, - 0.003067]],
         [[0.004730, - 0.002182, 0.002472, 0.000614],
          [0.000885, 0.001038, 0.005066, - 0.002213]]], dtype=np.float32)
    output_tnd = np.array(
        [[[-0.011963, 0.008606],
          [-0.001808, -0.006042]],
         [[-0.003372, -0.001404],
          [0.003098, -0.002792]],
         [[-0.000488, -0.000479],
          [0.002319, 0.000206]],
         [[0.002228, 0.005096],
          [0.002853, -0.000095]],
         [[0.000751, 0.002090],
          [0.002777, -0.000668]],
         [[0.004272, -0.001411],
          [0.001236, 0.000950]],
         [[0.005554, -0.001617],
          [0.002029, 0.000355]],
         [[0.004730, -0.002182],
          [0.002472, 0.000614]],
         [[-0.012329, 0.005493],
          [0.007935, -0.006226]],
         [[-0.007111, 0.002243],
          [0.008301, 0.000641]],
         [[-0.003967, 0.003769],
          [0.004517, -0.007660]],
         [[-0.003311, 0.002533],
          [0.003418, -0.008545]],
         [[-0.001663, 0.001678],
          [0.004700, -0.006409]],
         [[0.002823, 0.000984],
          [0.003998, -0.005737]],
         [[0.000610, 0.001289],
          [0.004150, -0.003067]],
         [[0.000885, 0.001038],
          [0.005066, -0.002213]]]
        , dtype=np.float32)
    output_attn = np.array(
        [[[0.000012, 0.000015, - 0.000013, - 0.000025,
           0.000009, 0.000016, 0.000003, - 0.000003],
          [-0.000010, - 0.000001, - 0.000011, - 0.000019,
           0.000009, 0.000007, - 0.000010, 0.000000]],
         [[0.000003, 0.000011, - 0.000010, - 0.000013,
           0.000005, 0.000007, - 0.000000, - 0.000001],
          [-0.000003, 0.000002, - 0.000006, - 0.000015,
           0.000006, 0.000004, - 0.000007, - 0.000000]]], dtype=np.float32)
    output_block = np.array(
        [[[0.016235, - 0.006042, - 0.005280, - 0.010742,
           0.008667, - 0.023071, 0.017456, - 0.007599],
          [0.003159, - 0.002487, 0.014648, - 0.020630,
           - 0.003220, - 0.003830, 0.011353, - 0.010986]],
        [[-0.001701, - 0.008789, 0.000431, 0.005829,
          - 0.010986, 0.011475, 0.009033, 0.005035],
        [0.009033, - 0.006805, - 0.001228, - 0.009338,
         - 0.002701, 0.005310, - 0.006958, - 0.003967]]], dtype=np.float32)
    grad_query = np.array(
        [[[[0.000000, 0.000000],
           [0.000000, 0.000000]],
          [[0.000000, 0.000000],
           [0.000000, 0.000000]]],
         [[[0.000002, - 0.000005],
           [-0.000023, - 0.000103]],
          [[-0.000017, 0.000015],
           [0.000043, 0.000032]]],
         [[[-0.000009, - 0.000014],
           [-0.000021, - 0.000062]],
          [[-0.000028, 0.000019],
           [0.000032, 0.000008]]],
         [[[-0.000007, 0.000030],
           [-0.000015, - 0.000046]],
          [[-0.000018, 0.000016],
           [0.000013, 0.000007]]],
         [[[0.000006, 0.000034],
           [-0.000007, - 0.000039]],
          [[-0.000018, 0.000020],
           [0.000025, 0.000014]]],
         [[[0.000005, 0.000028],
           [-0.000006, - 0.000033]],
          [[-0.000027, 0.000016],
           [0.000021, 0.000011]]],
         [[[0.000011, 0.000016],
           [-0.000005, - 0.000030]],
          [[-0.000021, 0.000033],
           [0.000044, 0.000012]]],
         [[[0.000010, 0.000023],
           [-0.000001, - 0.000017]],
          [[-0.000018, 0.000029],
           [0.000059, 0.000011]]]], dtype=np.float32)
    grad_key = np.array(
        [[[[-0.000012, 0.000021],
           [-0.000029, 0.000038]],
          [[0.000048, - 0.000022],
           [-0.000019, 0.000042]]],
         [[[-0.000018, 0.000035],
           [0.000033, - 0.000046]],
          [[0.000019, - 0.000012],
           [0.000093, 0.000048]]],
         [[[-0.000005, - 0.000018],
           [-0.000007, 0.000010]],
          [[-0.000056, 0.000039],
           [-0.000098, - 0.000112]]],
         [[[0.000030, - 0.000054],
           [-0.000002, 0.000001]],
          [[0.000013, - 0.000001],
           [-0.000021, - 0.000032]]],
         [[[0.000006, 0.000014],
           [0.000003, - 0.000007]],
          [[-0.000007, 0.000002],
           [0.000037, 0.000052]]],
         [[[0.000000, - 0.000000],
           [-0.000000, 0.000000]],
          [[-0.000037, 0.000001],
           [-0.000002, - 0.000001]]],
         [[[0.000007, 0.000011],
           [0.000000, - 0.000000]],
          [[0.000021, - 0.000009],
           [0.000003, 0.000006]]],
         [[[-0.000007, - 0.000008],
           [0.000001, 0.000004]],
          [[-0.000000, 0.000000],
           [0.000006, - 0.000003]]]], dtype=np.float32)
    grad_value = np.array(
        [[[[2.718750, 2.718750],
           [2.718750, 2.718750]],
          [[2.718750, 2.718750],
           [2.718750, 2.718750]]],
         [[[1.718750, 1.718750],
           [1.718750, 1.718750]],
          [[1.718750, 1.718750],
           [1.718750, 1.718750]]],
         [[[1.218750, 1.218750],
           [1.218750, 1.218750]],
          [[1.218750, 1.218750],
           [1.218750, 1.218750]]],
         [[[0.886719, 0.886719],
           [0.886719, 0.886719]],
          [[0.886719, 0.886719],
           [0.886719, 0.886719]]],
         [[[0.636719, 0.636719],
           [0.636719, 0.636719]],
          [[0.636719, 0.636719],
           [0.636719, 0.636719]]],
         [[[0.435547, 0.435547],
           [0.435547, 0.435547]],
          [[0.435547, 0.435547],
           [0.435547, 0.435547]]],
         [[[0.267578, 0.267578],
           [0.267578, 0.267578]],
          [[0.267578, 0.267578],
           [0.267578, 0.267578]]],
         [[[0.125000, 0.125000],
           [0.125000, 0.125000]],
          [[0.125000, 0.125000],
           [0.125000, 0.125000]]]], dtype=np.float32)
    return {
        'bnsd': output_bnsd,
        'tnd': output_tnd,
        "attn": output_attn,
        "query": grad_query,
        "key": grad_key,
        "value": grad_value,
        "block": output_block
    }


def get_gpu_datas() -> dict[str, np.ndarray]:
    """Generate gpu data for test."""
    output_bnsd = np.array(
        [[[-1.1963e-02, 8.6060e-03, -1.8082e-03, -6.0425e-03],
          [-1.2329e-02, 5.4932e-03, 7.9346e-03, -6.2256e-03]],
         [[-3.3722e-03, -1.4038e-03, 3.0975e-03, -2.7924e-03],
          [-7.1106e-03, 2.2430e-03, 8.3008e-03, 6.4087e-04]],
         [[-4.8828e-04, -4.7874e-04, 2.3193e-03, 2.0599e-04],
          [-3.9673e-03, 3.7689e-03, 4.5166e-03, -7.6599e-03]],
         [[2.2278e-03, 5.0964e-03, 2.8534e-03, -9.5367e-05],
          [-3.3112e-03, 2.5330e-03, 3.4180e-03, -8.5449e-03]],
         [[7.5150e-04, 2.0905e-03, 2.7771e-03, -6.6757e-04],
          [-1.6632e-03, 1.6785e-03, 4.6997e-03, -6.4087e-03]],
         [[4.2725e-03, -1.4114e-03, 1.2360e-03, 9.4986e-04],
          [2.8229e-03, 9.8419e-04, 3.9978e-03, -5.7373e-03]],
         [[5.5542e-03, -1.6174e-03, 2.0294e-03, 3.5477e-04],
          [6.1035e-04, 1.2894e-03, 4.1504e-03, -3.0670e-03]],
         [[4.7302e-03, -2.1820e-03, 2.4719e-03, 6.1417e-04],
          [8.8501e-04, 1.0376e-03, 5.0659e-03, -2.2125e-03]]], dtype=np.float16)
    output_tnd = np.array(
        [[[-1.1963e-02, 8.6060e-03],
          [-1.8082e-03, -6.0425e-03]],
         [[-3.3722e-03, -1.4038e-03],
          [3.0975e-03, -2.7924e-03]],
         [[-4.8828e-04, -4.7874e-04],
          [2.3193e-03, 2.0599e-04]],
         [[2.2278e-03, 5.0964e-03],
          [2.8534e-03, -9.5367e-05]],
         [[7.5150e-04, 2.0905e-03],
          [2.7771e-03, -6.6757e-04]],
         [[4.2725e-03, -1.4114e-03],
          [1.2360e-03, 9.4986e-04]],
         [[5.5542e-03, -1.6174e-03],
          [2.0294e-03, 3.5477e-04]],
         [[4.7302e-03, -2.1820e-03],
          [2.4719e-03, 6.1417e-04]],
         [[-1.2329e-02, 5.4932e-03],
          [7.9346e-03, -6.2256e-03]],
         [[-7.1106e-03, 2.2430e-03],
          [8.3008e-03, 6.4087e-04]],
         [[-3.9673e-03, 3.7689e-03],
          [4.5166e-03, -7.6599e-03]],
         [[-3.3112e-03, 2.5330e-03],
          [3.4180e-03, -8.5449e-03]],
         [[-1.6632e-03, 1.6785e-03],
          [4.6997e-03, -6.4087e-03]],
         [[2.8229e-03, 9.8419e-04],
          [3.9978e-03, -5.7373e-03]],
         [[6.1035e-04, 1.2894e-03],
          [4.1504e-03, -3.0670e-03]],
         [[8.8501e-04, 1.0376e-03],
          [5.0659e-03, -2.2125e-03]]], dtype=np.float16)
    output_attn = np.array(
        [[[1.1813e-05, 1.4700e-05, -1.3472e-05, -2.5206e-05,
           9.1753e-06, 1.5654e-05, 2.7061e-06, -2.9343e-06],
          [-9.8812e-06, -1.4772e-06, -1.0940e-05, -1.9328e-05,
           8.5110e-06, 7.1337e-06, -1.0376e-05, 5.1637e-07]],
         [[2.8103e-06, 1.1080e-05, -9.7287e-06, -1.3028e-05,
           4.7935e-06, 6.7433e-06, -4.5341e-08, -9.6254e-07],
          [-3.2683e-06, 1.5758e-06, -6.3239e-06, -1.5027e-05,
           5.9035e-06, 4.4412e-06, -7.2456e-06, -5.4673e-07]]], dtype=np.float32)
    output_block = np.array(
        [[[0.0162, -0.0061, -0.0053, -0.0107,
           0.0087, -0.0230, 0.0174, -0.0076],
          [0.0032, -0.0025, 0.0146, -0.0206,
           -0.0032, -0.0038, 0.0113, -0.0110]],
         [[-0.0017, -0.0088, 0.0004, 0.0058,
           -0.0110, 0.0114, 0.0090, 0.0050],
          [0.0090, -0.0068, -0.0012, -0.0094,
           -0.0027, 0.0053, -0.0069, -0.0040]]], dtype=np.float32)
    grad_query = np.array(
        [[[[0.0000e+00, 0.0000e+00],
           [0.0000e+00, 0.0000e+00]],
          [[0.0000e+00, 0.0000e+00],
           [0.0000e+00, 0.0000e+00]]],
         [[[1.6466e-06, -4.7982e-06],
           [-2.3484e-05, -1.0252e-04]],
          [[-1.7405e-05, 1.5497e-05],
           [4.2915e-05, 3.2425e-05]]],
         [[[-8.7023e-06, -1.4007e-05],
           [-2.1219e-05, -6.1512e-05]],
          [[-2.7895e-05, 1.9193e-05],
           [3.2187e-05, 7.5102e-06]]],
         [[[-7.3910e-06, 3.0398e-05],
           [-1.5199e-05, -4.5776e-05]],
          [[-1.7524e-05, 1.5974e-05],
           [1.3113e-05, 6.9439e-06]]],
         [[[6.2585e-06, 3.4332e-05],
           [-7.3314e-06, -3.9101e-05]],
          [[-1.8001e-05, 1.9550e-05],
           [2.5392e-05, 1.3709e-05]]],
         [[[5.0962e-06, 2.8372e-05],
           [-5.9307e-06, -3.2902e-05]],
          [[-2.6584e-05, 1.6451e-05],
           [2.1219e-05, 1.1444e-05]]],
         [[[1.0908e-05, 1.6451e-05],
           [-5.2750e-06, -2.9922e-05]],
          [[-2.0504e-05, 3.3140e-05],
           [4.4346e-05, 1.2159e-05]]],
         [[[9.7156e-06, 2.2650e-05],
           [-1.0654e-06, -1.6570e-05]],
          [[-1.8001e-05, 2.9087e-05],
           [5.8651e-05, 1.1146e-05]]]], dtype=np.float32)
    grad_key = np.array(
        [[[[-1.2279e-05, 2.1100e-05],
           [-2.8610e-05, 3.8147e-05]],
          [[4.8399e-05, -2.1815e-05],
           [-1.8954e-05, 4.1962e-05]]],
         [[[-1.7881e-05, 3.5048e-05],
           [3.3379e-05, -4.6492e-05]],
          [[1.8597e-05, -1.1504e-05],
           [9.2983e-05, 4.8399e-05]]],
         [[[-5.3644e-06, -1.8358e-05],
           [-6.9439e-06, 1.0014e-05]],
          [[-5.6267e-05, 3.9101e-05],
           [-9.7752e-05, -1.1206e-04]]],
         [[[2.9922e-05, -5.3644e-05],
           [-2.0862e-06, 1.2517e-06]],
          [[1.2636e-05, -1.1325e-06],
           [-2.0504e-05, -3.2187e-05]]],
         [[[6.0797e-06, 1.3649e-05],
           [3.0249e-06, -7.0333e-06]],
          [[-7.4208e-06, 2.4736e-06],
           [3.7193e-05, 5.2452e-05]]],
         [[[1.4435e-07, -3.8370e-07],
           [-4.0978e-07, 2.8312e-07]],
          [[-3.6955e-05, 1.4529e-06],
           [-1.5199e-06, -6.9663e-07]]],
         [[[6.6161e-06, 1.0550e-05],
           [4.9919e-07, -2.9802e-08]],
          [[2.0862e-05, -8.6427e-06],
           [3.2932e-06, 6.0201e-06]]],
         [[[-7.2718e-06, -8.1062e-06],
           [1.2442e-06, 3.8743e-06]],
          [[-1.0151e-07, 1.6857e-07],
           [5.6326e-06, -3.2783e-06]]]], dtype=np.float32)
    grad_value = np.array(
        [[[[2.7188, 2.7188],
           [2.7188, 2.7188]],
          [[2.7188, 2.7188],
           [2.7188, 2.7188]]],
         [[[1.7188, 1.7188],
           [1.7188, 1.7188]],
          [[1.7188, 1.7188],
           [1.7188, 1.7188]]],
         [[[1.2188, 1.2188],
           [1.2188, 1.2188]],
          [[1.2188, 1.2188],
           [1.2188, 1.2188]]],
         [[[0.8828, 0.8828],
           [0.8828, 0.8828]],
          [[0.8828, 0.8828],
           [0.8828, 0.8828]]],
         [[[0.6328, 0.6328],
           [0.6328, 0.6328]],
          [[0.6328, 0.6328],
           [0.6328, 0.6328]]],
         [[[0.4336, 0.4336],
           [0.4336, 0.4336]],
          [[0.4336, 0.4336],
           [0.4336, 0.4336]]],
         [[[0.2676, 0.2676],
           [0.2676, 0.2676]],
          [[0.2676, 0.2676],
           [0.2676, 0.2676]]],
         [[[0.1250, 0.1250],
           [0.1250, 0.1250]],
          [[0.1250, 0.1250],
           [0.1250, 0.1250]]]], dtype=np.float32)
    return {
        'bnsd': output_bnsd,
        'tnd': output_tnd,
        "attn": output_attn,
        "query": grad_query,
        "key": grad_key,
        "value": grad_value,
        "block":output_block
    }
