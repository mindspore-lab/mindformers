# Copyright 2023 Huawei Technologies Co., Ltd
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
""" test transformer apis."""
import numpy as np

import mindspore
from mindspore import Tensor
from mindspore.common import dtype
from mindspore.ops import operations as ops
from mindspore.common.api import _cell_graph_executor

from mindformers.core import CrossEntropyLoss
from mindformers.modules import FeedForward, FixedSparseAttention, LowerTriangularMaskWithDynamic


class MyActivation(mindspore.nn.Cell):
    """An example of custom activation"""

    def __init__(self):
        super(MyActivation, self).__init__()
        self.add = ops.Add()

    def construct(self, x):
        return self.add(x, 0.1)

    def activation_shard(self, parallel_config):
        self.add.shard(((parallel_config.data_parallel, parallel_config.model_parallel), ()))


class MyActivationNoShard(mindspore.nn.Cell):
    """An example of custom activation without shard"""

    def __init__(self):
        super(MyActivationNoShard, self).__init__()
        self.add = ops.Add()

    def construct(self, x):
        return self.add(x, 0.1)


def test_feedforward():
    """
    Feature: Feedforward
    Description: Test Feedforward module
    Expectation: No exception
    """
    model = FeedForward(hidden_size=15,
                        ffn_hidden_size=30,
                        dropout_rate=0.1,
                        hidden_act='relu')
    tensor = Tensor(np.ones((2, 20, 15)), dtype.float32)
    _cell_graph_executor.compile(model, tensor)


def test_cross_entropy_loss():
    """
    Feature: CrossEntropyLoss
    Description: Test CrossEntropyLoss with fake data
    Expectation: No exception
    """
    model = CrossEntropyLoss()
    logits = Tensor(np.array([[3, 5, 6, 9, 12, 33, 42, 12, 32, 72]]), dtype.float32)
    labels_np = np.array([1]).astype(np.int32)
    input_mask = Tensor(np.ones(1).astype(np.float32))
    labels = Tensor(labels_np)
    _cell_graph_executor.compile(model, logits, labels, input_mask)


def test_lower_triangular_mask_with_dynamic():
    """
    Feature: LowerTriangularMaskWithDynamic
    Description: Test LowerTriangularMaskWithDynamic module
    Expectation: No exception
    """
    model = LowerTriangularMaskWithDynamic(seq_length=19)
    inputs = Tensor(np.ones((2, 19)), dtype.float32)
    _cell_graph_executor.compile(model, inputs)


def test_fixed_sparse_attention():
    """
    Feature: FixedSparseAttention
    Description: Test FixedSparseAttention module
    Expectation: No exception
    """
    model = FixedSparseAttention(batch_size=2,
                                 seq_length=1024,
                                 size_per_head=64,
                                 num_heads=8,
                                 block_size=64)
    q = Tensor(np.ones((2, 1024, 512)), dtype.float16)
    k = Tensor(np.ones((2, 1024, 512)), dtype.float16)
    v = Tensor(np.ones((2, 1024, 512)), dtype.float16)
    mask = Tensor(np.ones((2, 1024, 1024)), dtype.float32)
    _cell_graph_executor.compile(model, q, k, v, mask)
