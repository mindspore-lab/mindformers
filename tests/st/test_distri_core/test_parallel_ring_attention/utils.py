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
"""some utility functions"""
import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.nn import DistributedGradReducer

from mindformers.experimental.distri_cores.create_comm import (
    get_dp_group,
    get_dp_world_size,
)
from mindformers.experimental.distri_cores.distributed.distributed_data_parallel import DistributedDataParallel
from mindformers.experimental.distri_cores.optimizer.distributed_optimizer import DistributedOptimizer


class TestData:
    """
    generate a test dataset
    """

    def __init__(self, data_size=None, input_data=None, label_data=None, with_attn_mask=False):
        super().__init__()
        self.with_attn_mask = with_attn_mask
        if input_data is not None:
            assert label_data is not None
            self.input_data = input_data
            self.data_size = self.input_data.shape
        else:
            self.input_data = np.random.random(data_size).astype(np.float32)
            self.data_size = self.input_data.shape
        if label_data is not None:
            assert input_data is not None
            self.label_data = label_data
        else:
            self.label_data = np.zeros(self.data_size[:2]).astype(np.float32)
        for i in range(self.data_size[0]):
            self.label_data[i][0] = 1
        seq_length = self.data_size[1]
        if self.with_attn_mask:
            # self.attention_mask = np.tril(np.ones(shape=(1, seq_length, seq_length))).astype(np.uint8)
            self.attention_mask = ops.ones(shape=(seq_length, seq_length), dtype=mstype.uint8)
            self.attention_mask = ops.triu(self.attention_mask, diagonal=1)
            self.attention_mask = ops.expand_dims(self.attention_mask, 0)
    def __getitem__(self, index):
        if self.with_attn_mask:
            return (
                Tensor(self.input_data[index]),
                Tensor(self.label_data[index]),
                Tensor(self.attention_mask),
            )
        return (Tensor(self.input_data[index]), Tensor(self.label_data[index]))

    def __len__(self):
        return self.input_data.shape[0]


def train(epoch_num,
          dataset,
          network,
          optimizer,
          save_ckpt_path=None,
          with_attn_input=False,
          reduce_grad=True,
          zero_level=-1,
          ):
    """
    define a train process
    """
    network.set_train()
    grad_func = ops.value_and_grad(
        network, grad_position=None, weights=list(network.get_parameters())
    )
    if (reduce_grad and ms.get_context("mode") == ms.PYNATIVE_MODE
            and get_dp_world_size(with_context_parallel=True) > 1
            and not isinstance(network, DistributedDataParallel)):
        grad_reducer = DistributedGradReducer(
            network.get_parameters(),
            group=get_dp_group(with_context_parallel=True),
            mean=True,
            degree=get_dp_world_size(with_context_parallel=True),
        )
    all_loss = []
    for epoch in range(epoch_num):
        step = 0
        for data in dataset:
            if isinstance(network, DistributedDataParallel):
                network.zero_grad_buffer()
            if with_attn_input:
                input_ids, labels, attn_mask = data
                loss, grads = grad_func(input_ids, attn_mask, labels)
            else:
                input_ids, labels = data
                loss, grads = grad_func(input_ids, labels)
            if isinstance(network, DistributedDataParallel):
                network.final_grad_reduce()
            if (reduce_grad and ms.get_context("mode") == ms.PYNATIVE_MODE
                    and get_dp_world_size(with_context_parallel=True) > 1
                    and not isinstance(network, DistributedDataParallel)):
                if zero_level < 0:
                    print(
                        "reduce gradients on group {}".format(
                            get_dp_group(with_context_parallel=True)
                        )
                    )
                    grads = grad_reducer(grads)
            if isinstance(optimizer, DistributedOptimizer):
                optimizer()
            else:
                optimizer(grads)
            print("Epoch {}, step {}, loss {}".format(epoch, step, loss))
            step += 1
            all_loss.append(loss.asnumpy())

    if save_ckpt_path is not None:
        ms.save_checkpoint(network, save_ckpt_path)
    return all_loss
