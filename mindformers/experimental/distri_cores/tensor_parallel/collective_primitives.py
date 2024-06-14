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

from mindspore.communication import get_group_size
from mindspore import ops, nn

from mindformers.experimental.distri_cores.utils import divide
from ..create_comm import (
    get_tp_group,
    get_tp_world_size,
    get_tp_rank,
)


# pylint: disable=W0622
def all_to_all_sp2hp(input):
    world_size = get_tp_world_size()
    tp_group = get_tp_group()
    all_to_all = AllToAll(tp_group, world_size, 0, 0)
    input = input.reshape(-1, input.shape[-1])
    split_tensors = ops.split(input, input.shape[-1] // world_size, axis=1)
    concat_tensor = ops.cat(split_tensors, axis=0)
    output = all_to_all(concat_tensor)
    return output

# pylint: disable=W0622
def all_to_all_hp2sp(input):
    world_size = get_tp_world_size()
    tp_group = get_tp_group()
    all_to_all = AllToAll(tp_group, world_size, 0, 0)
    input_exchanged = all_to_all(input)
    input_reshaped = input_exchanged.reshape(-1, input_exchanged.shape[-1])
    split_tensors = ops.split(input_reshaped, input_reshaped.shape[0] // world_size, axis=0)
    output = ops.cat(split_tensors, axis=-1)
    return output


class CopyToModelParallelRegion(nn.Cell):
    "Pass the input to the model parallel region."

    def __init__(self):
        super(CopyToModelParallelRegion, self).__init__()
        self.world_size = get_tp_world_size()
        self.all_reduce = ops.AllReduce(group=get_tp_group())

    # pylint: disable=C0303
    def construct(self, input_):
        return ops.stop_gradient(input_)
    
    # pylint: disable=W0613, C0111
    def bprop(self, x, out, dout):
        if self.world_size == 1:
            return (dout,)
        output = self.all_reduce(dout)
        return (output,)


class ScatterToModelParallelRegion(nn.Cell):
    "Split the input and keep only the corresponding chuck to the rank."

    def __init__(self):
        super(ScatterToModelParallelRegion, self).__init__()
        self.world_size = get_tp_world_size()
        self.all_gather = ops.AllGather(group=get_tp_group())
        self.rank = get_tp_rank()

    def construct(self, input_):
        if self.world_size == 1:
            return ops.stop_gradient(input_)

        last_dim = input_.ndim - 1
        last_dim_size = input_.shape[last_dim] // self.world_size
        tensor_tuple = ops.split(input_, last_dim_size, axis=last_dim)
        output = tensor_tuple[self.rank]

        return output

    # pylint: disable=W0613, C0111
    def bprop(self, x, out, dout):
        if self.world_size == 1:
            return (dout,)

        # Size and dimension.
        last_dim = dout.ndim - 1
        output = self.all_gather(dout)
        tensor_list = ops.split(output, dout.shape[0], axis=0)
        output = ops.cat(tensor_list, axis=last_dim).contiguous()

        return (output,)


class GatherFromModelParallelRegion(nn.Cell):
    "Gather the input from model parallel region and concatinate."

    def __init__(self):
        super(GatherFromModelParallelRegion, self).__init__()
        self.all_gather = ops.AllGather(group=get_tp_group())
        self.world_size = get_tp_world_size()
        self.rank = get_tp_rank()

    def construct(self, input_):
        # Size and dimension.
        last_dim = input_.ndim - 1
        output = self.all_gather(input_)
        tensor_list = ops.split(output, input_.shape[0], axis=0)
        output = ops.cat(tensor_list, axis=last_dim).contiguous()

        return output

    # pylint: disable=W0613, C0111
    def bprop(self, x, out, dout):
        if self.world_size == 1:
            return (dout,)
        last_dim = dout.ndim -1
        last_dim_size = divide(dout.shape[last_dim], self.world_size)
        # 对按第零维allgather的结果重新按最后一维排列
        tensor_tuple = ops.split(dout, last_dim_size, axis=last_dim)

        rank = get_tp_rank()
        output = tensor_tuple[rank].contiguous()
        return (output,)


class ReduceFromModelParallelRegion(nn.Cell):
    "All reduce the input from the model parallel region."

    def __init__(self):
        super(ReduceFromModelParallelRegion, self).__init__()
        self.world_size = get_tp_world_size()
        self.all_reduce = ops.AllReduce(group=get_tp_group())

    def construct(self, input_):
        if self.world_size == 1:
            return ops.stop_gradient(input_)
        output = self.all_reduce(input_)
        return output

    # pylint: disable=W0613, C0111
    def bprop(self, x, out, dout):
        return (dout,)


class ReduceScatterToSequenceParallelRegion(nn.Cell):
    "Reduce scatter the input from the model parallel region."

    def __init__(self):
        super(ReduceScatterToSequenceParallelRegion, self).__init__()
        self.reduce_scatter = ops.ReduceScatter(group=get_tp_group())
        self.all_gather = ops.AllGather(group=get_tp_group())
        self.world_size = get_tp_world_size()

    def construct(self, input_):
        if self.world_size == 1:
            return ops.stop_gradient(input_)
        output = self.reduce_scatter(input_.contiguous())
        return output

    # pylint: disable=W0613, C0111
    def bprop(self, x, out, dout):
        if self.world_size == 1:
            return dout
        output = self.all_gather(dout.contiguous())

        return (output,)


class ReduceScatterToTensorParallelRegion(nn.Cell):
    "Reduce scatter the input from the model parallel region."

    def __init__(self):
        super(ReduceScatterToTensorParallelRegion, self).__init__()
        self.reduce_scatter = ops.ReduceScatter(group=get_tp_group())
        self.all_gather = ops.AllGather(group=get_tp_group())
        self.world_size = get_tp_world_size()

    def construct(self, input_):
        num_dims = input_.ndim
        permute_order = (num_dims - 1,) + tuple(range(num_dims - 1))
        input_ = ops.transpose(input_, permute_order).contiguous()
        output = self.reduce_scatter(input_)

        permute_order = tuple(range(1, num_dims)) + (0,)
        output = ops.transpose(output, permute_order).contiguous()
        return output

    # pylint: disable=W0613, C0111
    def bprop(self, x, out, dout):
        if self.world_size == 1:
            return (dout,)

        # Size and dimension.
        last_dim = dout.ndim - 1
        output = self.all_gather(dout)
        tensor_list = ops.split(output, dout.shape[0], axis=0)
        output = ops.cat(tensor_list, axis=last_dim).contiguous()

        return (output,)


class ScatterToSequenceParallelRegion(nn.Cell):
    """Scatter To Sequence Paralle lRegion"""
    def __init__(self):
        super(ScatterToSequenceParallelRegion, self).__init__()
        self.all_gather = ops.AllGather(group=get_tp_group())
        self.world_size = get_tp_world_size()
        self.rank = get_tp_rank()

    # pylint: disable=C0111
    def construct(self, input_):
        if self.world_size == 1:
            return ops.stop_gradient(input_)

        dim_size = input_.shape[0]
        assert (
            dim_size % self.world_size == 0
        ), "First dimension of the tensor should be divisible by tensor parallel size"
        local_dim_size = dim_size // self.world_size

        dim_offset = self.rank * local_dim_size
        output = input_[dim_offset : dim_offset + local_dim_size].contiguous()

        return output

    # pylint: disable=W0613, C0111
    def bprop(self, x, out, dout):
        if self.world_size == 1:
            return (dout,)
        output = self.all_gather(dout.contiguous())
        return (output,)

class GatherFromSequenceParallelRegion(nn.Cell):
    """Gather From Sequence Parallel Region"""
    def __init__(self, tensor_parallel_output_grad=True):
        super(GatherFromSequenceParallelRegion, self).__init__()
        self.all_gather = ops.AllGather(group=get_tp_group())
        self.reduce_scatter = ops.ReduceScatter(group=get_tp_group())
        self.world_size = get_tp_world_size()
        self.rank = get_tp_rank()
        self.tensor_parallel_output_grad = tensor_parallel_output_grad

    def construct(self, input_):
        if self.world_size == 1:
            return ops.stop_gradient(input_)
        return self.all_gather(input_.contiguous())

    # pylint: disable=W0613, C0111
    def bprop(self, x, out, dout):
        if self.world_size == 1:
            return (dout,)

        if self.tensor_parallel_output_grad:
            return self.reduce_scatter(dout.contiguous())
        dim_size = dout.shape[0]
        assert (
            dim_size % self.world_size == 0
        ), "First dimension of the tensor should be divisible by tensor parallel size"
        local_dim_size = dim_size // self.world_size

        dim_offset = self.rank * local_dim_size
        output = dout[dim_offset : dim_offset + local_dim_size].contiguous()

        return (output,)


class AllGatherFromTensorParallelRegion(nn.Cell):
    """AllGather From Tensor Parallel Region"""
    def __init__(self):
        super(AllGatherFromTensorParallelRegion, self).__init__()
        self.all_gather = ops.AllGather(group=get_tp_group())
        self.reduce_scatter = ops.ReduceScatter(group=get_tp_group())
        self.world_size = get_tp_world_size()

    def construct(self, input_):
        if self.world_size == 1:
            return ops.stop_gradient(input_)

        # Size and dimension.
        last_dim = input_.ndim - 1
        output = self.all_gather(input_)
        tensor_list = ops.split(output, input_.shape[0], axis=0)
        output = ops.cat(tensor_list, axis=last_dim).contiguous()
        return output

    # pylint: disable=W0613
    def bprop(self, x, out, dout):
        num_dims = dout.ndim
        permute_order = (num_dims - 1,) + tuple(range(num_dims - 1))
        dout = ops.transpose(dout, permute_order).contiguous()
        output = self.reduce_scatter(dout)

        permute_order = tuple(range(1, num_dims)) + (0,)
        output = ops.transpose(output, permute_order).contiguous()
        return (output,)


class AllToAll(nn.Cell):
    """All to All"""
    def __init__(self, group, split_count, split_dim, concat_dim):
        super(AllToAll, self).__init__()
        self.all_to_all = ops.AlltoAll(split_count, split_dim, concat_dim, group=group)
        self.all_to_all_grad = ops.AlltoAll(split_count, concat_dim, split_dim, group=group)
        self.world_size = get_group_size(group=group)

    def construct(self, input_):
        if self.world_size == 1:
            return ops.stop_gradient(input_)

        return self.all_to_all(input_)

    # pylint: disable=W0613
    def bprop(self, x, out, dout):
        if self.world_size == 1:
            return (dout,)

        output = self.all_to_all_grad(dout)
        return (output,)
