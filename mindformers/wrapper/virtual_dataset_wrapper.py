# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Virtual dataset wrapper module for MindSpore.

This module provides a virtual dataset wrapper class that converts data parallel
layout to model parallel layout in semi auto parallel and auto parallel modes.
"""
from mindspore.nn import Cell
from mindspore.ops import PrimitiveWithInfer, prim_attr_register


class _VirtualDatasetCell(Cell):
    """
    Wrap the network with virtual dataset to convert data parallel layout to model parallel layout.

    _VirtualDataset is a virtual Primitive, it does not exist in the final executing graph. Inputs and outputs
    of _VirtualDataset are distributed in data parallel pattern, tensor redistribution Primitives is inserted
    dynamically during the graph compile process.

    Note:
        Only used in semi auto parallel and auto parallel mode.

    Args:
        backbone (Cell): The target network to wrap.

    Examples:
        >>> net = Net()
        >>> net = _VirtualDatasetCell(net)
    """

    def __init__(self, backbone):
        super().__init__(auto_prefix=False)
        self._backbone = backbone
        self._virtual_dataset = _VirtualDataset()
        self._get_attr_from_cell(backbone)

    def construct(self, *inputs):
        output = self._virtual_dataset(*inputs)
        return self._backbone(*output)


class _VirtualDataset(PrimitiveWithInfer):
    """
    Auto parallel virtual dataset operator.

    It would insert VirtualDataset operator in forward computation and be deleted before backward computation.
    """

    @prim_attr_register
    def __init__(self):
        """Initialize _VirtualDataset."""
        self.add_prim_attr('order_enforce_skip', True)

    def infer_shape(self, *args):
        return args

    def infer_dtype(self, *args):
        return args
