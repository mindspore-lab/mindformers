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
""" Test ParallelAttention. """
import argparse
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore import Tensor
import mindspore.common.dtype as mstype
from mindspore.nn import AdamWeightDecay
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.communication.management import init

from mindformers.experimental.distri_cores.config import ModelParallelConfig, TransformerConfig
from mindformers.experimental.distri_cores.create_comm import initialize_model_parallel
from mindformers.experimental.distri_cores.transformer import ParallelAttention
from mindformers.experimental.distri_cores.transformer.rotary_pos_embedding import RotaryEmbedding

from tests.st.test_distri_core.utils import TestData, train, transform_transformerlayer_params, generate_ckpt


class ParallelAttentionNet(nn.Cell):
    """ ParallelAttentionNet. """
    def __init__(self, config, with_rope=False):
        super(ParallelAttentionNet, self).__init__()
        self.with_rope = with_rope
        if with_rope:
            self.rope = RotaryEmbedding(kv_channels=config.hidden_size//config.num_heads,
                                        rotary_percent=1.0)
        self.attention = ParallelAttention(layer_number=1, config=config)
        self.loss = SoftmaxCrossEntropyWithLogits()

    def construct(self, x, attention_mask, labels):
        """ construct."""
        if self.with_rope:
            emb = self.rope(max_seq_len=x.shape[1])
            output, _ = self.attention(x, attention_mask, rotary_pos_emb=emb)
        else:
            output, _ = self.attention(x, attention_mask)
        output = ops.sum(output, dim=-1, keepdim=False)
        loss = self.loss(output, labels)
        return loss


def run_parallel_attention_with_rope(use_fa=False, use_gqa=False):
    """ Test ParallelAttention. """
    batch_size = 1
    dataset_size = 3
    seq_length = 8
    num_heads = 8
    kv_num_heads = 4
    hidden_size = 16
    tensor_parallel = 2

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')
    init()
    initialize_model_parallel(tensor_model_parallel_size=tensor_parallel)

    ms.set_seed(2024)
    input_data = np.random.random((dataset_size, seq_length, hidden_size)).astype(np.float32)
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    for i in range(label_data.shape[0]):
        label_data[i][0] = 1
    if use_fa:
        attn_mask = (1-np.tril(np.ones(shape=(1, seq_length, seq_length)))).astype(np.uint8)
    else:
        attn_mask = ((1-np.tril(np.ones(shape=(1, seq_length, seq_length)))) * -10000).astype(np.float16)
    dataset = TestData(input_data=input_data, label_data=label_data, attn_mask=attn_mask)
    dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels', "attention_mask"])
    dataset = dataset.batch(batch_size)

    parallel_config = ModelParallelConfig(tensor_parallel=tensor_parallel)
    config = TransformerConfig(vocab_size=1,
                               num_layers=1,
                               seq_length=seq_length,
                               num_heads=num_heads,
                               use_gqa=use_gqa,
                               kv_num_heads=kv_num_heads,
                               hidden_size=hidden_size,
                               ffn_hidden_size=hidden_size,
                               attn_type='self_attn',
                               qkv_has_bias=True,
                               out_proj_has_bias=False,
                               parallel_config=parallel_config,
                               param_init_dtype='float32',
                               compute_dtype='float32',
                               softmax_compute_dtype='float32',
                               attention_dropout_rate=0.0,
                               mask_func_type="attn_mask_add",
                               use_flash_attention=use_fa,
                               fa_config={
                                   'pre_tokens': 65536,
                                   'next_tokens': 0,
                                   'sparse_mode': 0,
                                   'input_layout': 'BNSD',
                               })
    network = ParallelAttentionNet(config=config, with_rope=True)
    kv_hidden_size = hidden_size // num_heads * kv_num_heads if use_gqa else None

    param_dict = generate_ckpt(hidden_size=hidden_size, module_type='attention', kv_hidden_size=kv_hidden_size)
    pynative_params = transform_transformerlayer_params(param_dict,
                                                        hidden_size=hidden_size,
                                                        kv_hidden_size=kv_hidden_size)
    ms.load_param_into_net(network, pynative_params)

    input_ids = Tensor(shape=(None, None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    attn_mask = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(input_ids, labels, attn_mask)

    optimizer = AdamWeightDecay(params=network.get_parameters())

    losses = train(1, dataset, network, optimizer, None, with_attn_input=True)
    losses = list(map(lambda x: x[0], losses))
    golden_losses = [61.793457, 137.65796, 122.2657]
    if use_gqa:
        golden_losses = [35.357117, 96.4671, 88.65936]
    if use_fa:
        golden_losses = [61.765625, 137.57812, 122.34394]

    assert np.allclose(losses, golden_losses, atol=1.e-3, rtol=1.e-3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--use_fa', action='store_true', help="Whether use flashattention."
    )
    parser.add_argument(
        '--use_gqa', action='store_true', help="Whether use group attention."
    )

    args, rest_args = parser.parse_known_args()
    run_parallel_attention_with_rope(args.use_fa, args.use_gqa)
