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
"""
Test module for testing the Optimizer classes under distri_core path.
"""
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from mindspore.communication.management import init, get_rank, get_group_size, GlobalComm
from mindspore import context, Tensor, Parameter
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.nn.wrap.cell_wrapper import WithLossCell
from mindspore.common import ParameterTuple
from mindformers import AdamWeightDecayZeRO2


ms.set_context(mode=ms.PYNATIVE_MODE)
context.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
init()


class Net(Cell):
    """
    Define a network contains three layers: mul, add and mul.
    """
    def __init__(self, size, fp16_init=False):
        """Init network."""
        super().__init__()
        if fp16_init:
            init_dtype = np.float16
        else:
            init_dtype = np.float32
        self.weight_1 = Parameter(Tensor(np.full(size, 0.1, dtype=init_dtype)), name='weight_1')
        self.mul_1 = P.Mul()
        self.weight_2 = Parameter(Tensor(np.full(size, 0.5, dtype=init_dtype)), name='weight_2')
        self.mul_2 = P.Mul()
        self.weight_3 = Parameter(Tensor(np.full(size, 1, dtype=init_dtype)), name='weight_3')
        self.add = P.Add()

    def construct(self, x):
        """construct method."""
        output = self.mul_1(x, self.weight_1)
        output = self.mul_2(output, self.weight_2)
        output = self.add(output, self.weight_3)
        return output


class TestData():
    """
    Define dataset iterator.
    """
    def __init__(self, input_data, label_data):
        """Init dataset."""
        super().__init__()
        self.input_data = input_data
        self.data_num = self.input_data.shape[0]
        self.seq_length = self.input_data[0].shape[0]
        self.label_data = label_data

        for i in range(self.data_num):
            self.label_data[i][i] = 1

    def __getitem__(self, index):
        """get item given index."""
        return (Tensor(self.input_data[index]), Tensor(self.label_data[index]))

    def __len__(self):
        """get dataset size."""
        return self.input_data.shape[0]


def train(epoch_size, dataset, net, loss, opt):
    """train method."""
    net_with_criterion = WithLossCell(net, loss)
    net_with_criterion.set_grad()
    net_with_criterion.set_train()
    grad_func = F.grad(net_with_criterion, grad_position=None,
                       weights=ParameterTuple(net_with_criterion.trainable_params()))

    for epoch in range(epoch_size):
        step = 0
        for input_ids, labels in dataset:
            fw_output = net(input_ids)
            loss_output = loss(fw_output, labels)
            grads = grad_func(input_ids, labels)
            opt(grads)
            print("Epoch: {} | Step: {} | Loss: {}".format(epoch, step, loss_output))
            step += 1


def run_adamwzero2_optimizer():
    """test adamwzero2 optimizer."""
    rank_id = get_rank()
    rank_size = get_group_size()

    data_num = 8
    seq_length = 8
    input_data = np.random.random((data_num, seq_length)).astype(np.float32)
    label_data = np.zeros((data_num, seq_length)).astype(np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data)

    parallel_dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels'],
                                           num_shards=rank_size, shard_id=rank_id)
    parallel_dataset = parallel_dataset.batch(data_num // rank_size)

    network = Net(size=(seq_length))
    loss = SoftmaxCrossEntropyWithLogits(reduction='none')
    optimizer = AdamWeightDecayZeRO2(params=network.get_parameters(), use_parallel=True,
                                     opt_parallel_group=GlobalComm.WORLD_COMM_GROUP, cpu_offload=False)

    train(epoch_size=1, dataset=parallel_dataset, net=network, loss=loss, opt=optimizer)


def run_adamwzero2_optimizer_offload():
    """test adamwzero2 optimizer with offload."""
    rank_id = get_rank()
    rank_size = get_group_size()

    data_num = 8
    seq_length = 8
    input_data = np.random.random((data_num, seq_length)).astype(np.float32)
    label_data = np.zeros((data_num, seq_length)).astype(np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data)

    parallel_dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels'],
                                           num_shards=rank_size, shard_id=rank_id)
    parallel_dataset = parallel_dataset.batch(data_num // rank_size)

    network = Net(size=(seq_length))
    loss = SoftmaxCrossEntropyWithLogits(reduction='none')
    optimizer = AdamWeightDecayZeRO2(params=network.get_parameters(), use_parallel=True,
                                     opt_parallel_group=GlobalComm.WORLD_COMM_GROUP, cpu_offload=True)

    train(epoch_size=1, dataset=parallel_dataset, net=network, loss=loss, opt=optimizer)


run_adamwzero2_optimizer()
run_adamwzero2_optimizer_offload()
