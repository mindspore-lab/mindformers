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
""" Test ParamAndGradBuffer """
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
from mindspore.communication.management import init
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.mint.optim import AdamW

from mindformers.experimental.distri_cores.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mindformers.experimental.distri_cores.create_comm import initialize_model_parallel
from mindformers.experimental.distri_cores.config import OptimizerConfig, ModelParallelConfig, TransformerConfig
from mindformers.experimental.distri_cores.distributed import DistributedDataParallel, \
    DistributedDataParallelConfig
from mindformers.experimental.distri_cores.optimizer.distributed_optimizer import DistributedOptimizer
from mindformers.experimental.distri_cores.create_comm import get_dp_group

from tests.st.test_distri_core.utils import TestData, train


class TestNet(nn.Cell):
    """ test class. """
    def __init__(self, config):
        super(TestNet, self).__init__()
        hidden_size = config.hidden_size
        self.columnlinear = ColumnParallelLinear(input_size=hidden_size,
                                                 output_size=hidden_size,
                                                 config=config,
                                                 init_method=config.init_method,
                                                 bias=config.mlp_has_bias,
                                                 gather_output=False,
                                                 skip_bias_add=False,
                                                 bias_init=config.bias_init)
        self.rowlinear = RowParallelLinear(input_size=hidden_size,
                                           output_size=hidden_size,
                                           config=config,
                                           init_method=config.init_method,
                                           bias=config.mlp_has_bias,
                                           input_is_parallel=True,
                                           skip_bias_add=False,
                                           bias_init=config.bias_init)
        self.loss = SoftmaxCrossEntropyWithLogits()

    def construct(self, input_, label_):
        output, _ = self.columnlinear(input_)
        output, _ = self.rowlinear(output)
        output = ops.sum(output, dim=-1, keepdim=False)
        output = ops.cast(output, mstype.float32)
        loss = self.loss(output, label_)

        return loss


def run_distributed_optimizer():
    """
    Feature: test DDP with DistributedOptimizer
    Description: test DDP with DistributedOptimizer
    Expectation: test success
    """
    batch_size = 1
    dataset_size = 5
    seq_length = 8
    hidden_size = 16
    tensor_parallel = 1
    bucket_size = 10

    ms.set_context(device_target='Ascend', mode=ms.PYNATIVE_MODE, deterministic='ON')
    ms.set_seed(2024)
    init()
    initialize_model_parallel(tensor_model_parallel_size=tensor_parallel)
    input_data = np.random.random((dataset_size, seq_length, hidden_size)).astype(np.float32)
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    for i in range(label_data.shape[0]):
        label_data[i][0] = 1
    attn_mask = ((1-np.tril(np.ones(shape=(1, seq_length, seq_length)))) * -10000).astype(np.float16)
    dataset = TestData(input_data=input_data, label_data=label_data, attn_mask=attn_mask)
    dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels', 'attention_mask'])
    dataset = dataset.batch(batch_size)

    parallel_config = ModelParallelConfig()
    optimizer_config = OptimizerConfig(parallel_config=parallel_config)
    model_config = TransformerConfig(vocab_size=40000,
                                     num_layers=1,
                                     num_heads=1,
                                     mlp_has_bias=True,
                                     mlp_has_gate=False,
                                     hidden_size=hidden_size,
                                     ffn_hidden_size=4*hidden_size,
                                     hidden_act='gelu',
                                     parallel_config=parallel_config,
                                     param_init_dtype='float32',
                                     compute_dtype='float32')
    ddp_config = DistributedDataParallelConfig(
        overlap_grad_reduce=True,
        use_distributed_optimizer=True,
        bucket_size=bucket_size,
    )
    network = TestNet(config=model_config)
    network = DistributedDataParallel(config=model_config,
                                      ddp_config=ddp_config,
                                      module=network)

    optimizer = AdamW(params=network.get_parameters(), lr=1.e-1, weight_decay=0.0)
    optimizer = DistributedOptimizer(optimizer=optimizer,
                                     config=optimizer_config,
                                     per_model_buffers=network.buffers,
                                     data_parallel_group=get_dp_group(with_context_parallel=True))

    losses = train(epoch_num=1,
                   dataset=dataset,
                   network=network,
                   optimizer=optimizer,
                   reduce_grad=False)
    losses = list(map(lambda x: x[0], losses))
    golden_losses = [2.0789604, 1.834939, 1.2082496, 7.185639, 6.9287825]

    assert np.allclose(losses, golden_losses, atol=1.e-3, rtol=1.e-3)


if __name__ == '__main__':
    run_distributed_optimizer()
