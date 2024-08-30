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

from mindspore import nn, ops

from mindformers.experimental.distri_cores.create_comm import get_tp_group, get_tp_rank, get_tp_world_size


class GatherFromModelParallelRegion(nn.Cell):
    "Gather the input from model parallel region and concatinate."

    def __init__(self):
        super().__init__()
        self.all_gather = ops.AllGather(group=get_tp_group())
        self.world_size = get_tp_world_size()
        self.split = ops.Split(axis=0, output_num=self.world_size)

    def construct(self, input_):
        # Size and dimension.
        if self.world_size == 1:
            return input_
        output = self.all_gather(input_)
        tensor_list = self.split(output)
        output = ops.cat(tensor_list, axis=-1)

        return output


class GatherFromSequenceParallelRegion(nn.Cell):
    """
    Class for gathering sequences in a parallel region across multiple devices.

    If the world size is greater than 1, it gathers input from all devices; otherwise,
    it returns the input as is.
    """

    def __init__(self):
        super().__init__()
        self.all_gather = ops.AllGather(group=get_tp_group())
        self.world_size = get_tp_world_size()

    def construct(self, input_):
        if self.world_size == 1:
            return input_
        return self.all_gather(input_)


class ReduceFromModelParallelRegion(nn.Cell):
    "All reduce the input from the model parallel region."

    def __init__(self):
        super().__init__()
        self.all_reduce = ops.AllReduce(group=get_tp_group())
        self.world_size = get_tp_world_size()

    def construct(self, input_):
        if self.world_size == 1:
            return input_
        output = self.all_reduce(input_)
        return output


class ReduceScatterToSequenceParallelRegion(nn.Cell):
    "Reduce scatter the input from the model parallel region."

    def __init__(self):
        super().__init__()
        self.reduce_scatter = ops.ReduceScatter(group=get_tp_group())
        self.world_size = get_tp_world_size()

    def construct(self, input_):
        if self.world_size == 1:
            return input_
        output = self.reduce_scatter(input_)
        return output


class ScatterToModelParallelRegion(nn.Cell):
    "Split the input and keep only the corresponding chuck to the rank."

    def __init__(self):
        super().__init__()
        self.world_size = get_tp_world_size()
        self.rank = get_tp_rank()
        self.split = ops.Split(axis=-1, output_num=self.world_size)

    def construct(self, input_):
        if self.world_size == 1:
            return input_
        tensor_tuple = self.split(input_)
        output = tensor_tuple[self.rank]
        return output
