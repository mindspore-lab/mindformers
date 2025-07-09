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
# ======================
"""mapping"""
from mindspore import Tensor, mint, nn, ops

from mindformers.parallel_core.inference.parallel_state import ProcessGroup


class GatherFromModelParallelRegion(nn.Cell):
    """ Gather the input tensor across the specified group and concatinate along the specified dimension. """

    def __init__(self, process_group: ProcessGroup, dim: int = -1) -> None:
        super().__init__()
        self.group_size = process_group.size
        if self.group_size > 1:
            self.dim = dim
            self.group = process_group.group
            self.all_gather_into_tensor = ops.AllGather(group=self.group)

    def construct(self, input_: Tensor) -> Tensor:
        if self.group_size == 1:
            return input_
        if self.dim == 0:
            return self.all_gather_into_tensor(input_)
        input_ = mint.transpose(input_, 0, self.dim)
        output = self.all_gather_into_tensor(input_)
        output = mint.transpose(output, 0, self.dim)
        return output


class ReduceFromModelParallelRegion(nn.Cell):
    """ Allruduce the input tensor across the specified group. """

    def __init__(self, process_group: ProcessGroup) -> None:
        super().__init__()
        self.group_size = process_group.size
        if self.group_size > 1:
            self.group = process_group.group
            self.all_reduce = ops.AllReduce(group=self.group)

    def construct(self, input_: Tensor) -> Tensor:
        if self.group_size == 1:
            return input_
        output = self.all_reduce(input_)
        return output


class ReduceScatterToModelParallelRegion(nn.Cell):
    """ Reducescatter the input tensor across the specified group. """

    def __init__(self, process_group: ProcessGroup) -> None:
        super().__init__()
        self.group_size = process_group.size
        if self.group_size > 1:
            self.group = process_group.group
            self.reduce_scatter = ops.ReduceScatter(group=self.group)

    def construct(self, input_: Tensor) -> Tensor:
        if self.group_size == 1:
            return input_
        output = self.reduce_scatter(input_)
        return output


class ScatterToModelParallelRegion(nn.Cell):
    "Split the input and keep only the corresponding chuck to the rank across the specified group."

    def __init__(self, process_group: ProcessGroup, dim: int = -1) -> None:
        super().__init__()
        self.group_size = process_group.size
        if self.group_size > 1:
            self.rank = process_group.rank
            self.split = ops.Split(axis=dim, output_num=self.group_size)

    def construct(self, input_: Tensor) -> Tensor:
        if self.group_size == 1:
            return input_
        tensor_tuple = self.split(input_)
        output = tensor_tuple[self.rank]
        return output


def gather_from_model_parallel_region(input_: Tensor, process_group: ProcessGroup, dim: int = -1) -> Tensor:
    """
    Gather the input tensor across the specified model parallel group and concatenate along the specified dimension.

    Args:
        input_ (Tensor): The input tensor to gather.
        process_group (ProcessGroup): The group information for model parallel.
        dim (int): The dimension along which to concatenate the gathered tensors.

    Returns:
        Tensor: The gathered tensor.
    """
    gather_op = GatherFromModelParallelRegion(process_group, dim)
    return gather_op(input_)


def reduce_from_model_parallel_region(input_: Tensor, process_group: ProcessGroup) -> Tensor:
    """
    Reduce the input tensor across the specified model parallel group.

    Args:
        input_ (Tensor): The input tensor to reduce.
        process_group (ProcessGroup): The group information for model parallel.

    Returns:
        Tensor: The reduced tensor.
    """
    reduce_op = ReduceFromModelParallelRegion(process_group)
    return reduce_op(input_)


def reduce_scatter_to_model_parallel_region(input_: Tensor, process_group: ProcessGroup) -> Tensor:
    """
    Reduce scatter the input tensor across the specified model parallel group.

    Args:
        input_ (Tensor): The input tensor to reduce scatter.
        process_group (ProcessGroup): The group information for model parallel.

    Returns:
        Tensor: The reduced scattered tensor.
    """
    reduce_scatter_op = ReduceScatterToModelParallelRegion(process_group)
    return reduce_scatter_op(input_)


def scatter_to_model_parallel_region(input_: Tensor, process_group: ProcessGroup, dim: int = -1) -> Tensor:
    """
    Split the input tensor and keep only the corresponding chunk to the rank in model parallel region.

    Args:
        input_ (Tensor): The input tensor to scatter.
        process_group (ProcessGroup): The group information for model parallel.
        axis (int): The axis along which to split the input tensor.

    Returns:
        Tensor: The scattered tensor.
    """
    scatter_op = ScatterToModelParallelRegion(process_group, dim)
    return scatter_op(input_)
