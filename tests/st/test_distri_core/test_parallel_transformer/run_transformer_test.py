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
""" Test ParallelTransformer. """
import os
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
from mindspore.common.initializer import Normal, Zero

from mindformers.experimental.parallel_core.pynative.config import ModelParallelConfig, TransformerConfig
from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel
from mindformers.experimental.parallel_core.pynative.transformer import ParallelTransformer
from mindformers.experimental.parallel_core.pynative.transformer.rotary_pos_embedding import RotaryEmbedding

from tests.st.test_distri_core.utils import TestData, train, transform_transformerlayer_params, generate_ckpt


class ParallelTransformerNet(nn.Cell):
    """ ParallelTransformerNet. """
    def __init__(self, config, with_rope=False):
        super(ParallelTransformerNet, self).__init__()
        self.with_rope = with_rope
        if with_rope:
            self.rope = RotaryEmbedding(kv_channels=config.hidden_size // config.num_heads,
                                        rotary_percent=1.0)
        self.transformer = ParallelTransformer(config=config, post_norm=False)
        self.loss = SoftmaxCrossEntropyWithLogits()

    def construct(self, x, attention_mask, labels):
        """ construct. """
        if self.with_rope:
            emb = self.rope(max_seq_len=x.shape[1])
            output = self.transformer(x, attention_mask, rotary_pos_emb=emb)
        else:
            output = self.transformer(x, attention_mask)
        output = ops.sum(output, dim=-1, keepdim=False)
        loss = self.loss(output, labels)
        return loss


def run_parallel_transformer():
    """ Test ParallelTransformer. """
    batch_size = 1
    dataset_size = 3
    num_layers = 2
    seq_length = 16
    num_heads = 8
    hidden_size = 64
    tensor_parallel = 2

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')

    init()
    initialize_model_parallel(tensor_model_parallel_size=tensor_parallel)

    ms.set_seed(2024)
    input_data = np.random.random((dataset_size, seq_length, hidden_size)).astype(np.float32)
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    for i in range(label_data.shape[0]):
        label_data[i][0] = 1
    attn_mask = ((1 - np.tril(np.ones(shape=(1, seq_length, seq_length)))) * -10000).astype(np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data, attn_mask=attn_mask)
    dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels', "attention_mask"])
    dataset = dataset.batch(batch_size)

    parallel_config = ModelParallelConfig(tensor_parallel=tensor_parallel)
    config = TransformerConfig(seq_length=seq_length,
                               vocab_size=1,
                               num_layers=num_layers,
                               num_heads=num_heads,
                               hidden_size=hidden_size,
                               attention_type='self_attn',
                               qkv_has_bias=True,
                               out_proj_has_bias=False,
                               parallel_config=parallel_config,
                               param_init_dtype='float32',
                               compute_dtype='float32',
                               softmax_compute_dtype='float32',
                               init_method=Normal(),
                               bias_init=Zero(),
                               hidden_dropout_rate=0.0,
                               attention_dropout_rate=0.0,
                               mask_func_type="attn_mask_add",
                               mlp_has_bias=True,
                               ffn_hidden_size=4 * hidden_size,
                               hidden_act='gelu',
                               apply_residual_connection_post_norm=False,
                               normalization='FusedRMSNorm',
                               layernorm_epsilon=1.e-5)
    network = ParallelTransformerNet(config=config, with_rope=True)
    save_golden = False
    if save_golden:
        golden_ckpt_path = "transformer_golden.ckpt"
        assert os.path.exists(golden_ckpt_path), \
            "'golden.ckpt' did not exits, please run generate_golden() to generate one by running below command: \n" + \
            "`pytest -sv test_transformer.py::TestParallelTransformer::generate_golden`"
        golden_params = ms.load_checkpoint(golden_ckpt_path)
        pynative_params = transform_transformerlayer_params(golden_params,
                                                            hidden_size=hidden_size,
                                                            kv_hidden_size=None,
                                                            prefix="transformer.layers.")
        param_not_load, _ = ms.load_param_into_net(network, pynative_params)
        assert not param_not_load, f"{param_not_load} was not loaded in this net, test failed."
    else:
        param_dict = generate_ckpt(hidden_size=hidden_size,
                                   module_type='transformer',
                                   num_layers=num_layers)
        pynative_params = transform_transformerlayer_params(param_dict,
                                                            hidden_size=hidden_size,
                                                            kv_hidden_size=None,
                                                            prefix='transformer.layers.')
        ms.load_param_into_net(network, pynative_params)

    input_ids = Tensor(shape=(None, None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    attn_mask = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(input_ids, labels, attn_mask)

    optimizer = AdamWeightDecay(params=network.get_parameters())

    losses = train(1, dataset, network, optimizer, None, with_attn_input=True)
    losses = list(map(lambda x: x[0], losses))
    golden_losses = [1487.875, 2019.565, 932.0647]

    assert np.allclose(losses, golden_losses, atol=1.e-3, rtol=1.e-3)


if __name__ == '__main__':
    run_parallel_transformer()
