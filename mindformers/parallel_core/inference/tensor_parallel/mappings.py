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
from mindspore.communication import get_group_size, GlobalComm, get_rank
from mindformers.parallel_core.inference.utils import get_tp_world_size, get_dp_world_size, get_moe_tp_world_size
from mindformers.parallel_core.inference.parallel_state import (get_tensor_model_parallel_group,
                                                                get_tensor_model_parallel_rank,
                                                                get_moe_tensor_parallel_group,
                                                                get_moe_tensor_parallel_rank,
                                                                get_data_parallel_rank)


class GatherFromWorldParallelRegionV1(nn.Cell):
    """
    Gather the input from world parallel region and concatinate, simultaneously perform
    transpose operation on input.
    """

    def __init__(self):
        super().__init__()
        self.world_size = get_group_size()
        if self.world_size > 1:
            self.world_group = GlobalComm.WORLD_COMM_GROUP
            self.all_gather_into_tensor = ops.AllGather(group=self.world_group)

    def construct(self, input_):
        # Size and dimension.
        if self.world_size == 1:
            return input_
        input_ = ops.swapaxes(input_, 0, -1)
        output = self.all_gather_into_tensor(input_)
        output = ops.swapaxes(output, 0, -1)
        return output


class GatherFromModelParallelRegion(GatherFromWorldParallelRegionV1):
    "Gather the input from model parallel region and concatinate."

    def __init__(self):
        super().__init__()
        self.world_size = get_tp_world_size()
        if self.world_size > 1:
            self.tp_group = get_tensor_model_parallel_group()
            self.all_gather_into_tensor = ops.AllGather(group=self.tp_group)


class GatherFromMoeTensorParallelRegion(GatherFromWorldParallelRegionV1):
    """
    Gather the input from moe tensor parallel region and concatinate, simultaneously perform
    transpose operation on input.
    """

    def __init__(self):
        super().__init__()
        self.world_size = get_moe_tp_world_size()
        if self.world_size > 1:
            self.moe_tp_group = get_moe_tensor_parallel_group()
            self.all_gather_into_tensor = ops.AllGather(group=self.moe_tp_group)


class GatherFromWorldParallelRegionV2(nn.Cell):
    "Gather the input from world parallel region and concatinate."

    def __init__(self):
        super().__init__()
        self.world_size = get_group_size()
        if self.world_size > 1:
            self.world_group = GlobalComm.WORLD_COMM_GROUP
            self.all_gather_into_tensor = ops.AllGather(group=self.world_group)

    def construct(self, input_):
        if self.world_size == 1:
            return input_
        return self.all_gather_into_tensor(input_)


class GatherFromSequenceParallelRegion(GatherFromWorldParallelRegionV2):
    """
    Class for gathering sequences in a parallel region across multiple devices.

    If the world size is greater than 1, it gathers input from all devices; otherwise,
    it returns the input as is.
    """

    def __init__(self):
        super().__init__()
        self.world_size = get_tp_world_size()
        if self.world_size > 1:
            self.tp_group = get_tensor_model_parallel_group()
            self.all_gather_into_tensor = ops.AllGather(group=self.tp_group)


class GatherFromMoeTensorParallelRegionV2(GatherFromWorldParallelRegionV2):
    """
    Class for gathering sequences in moe tensor parallel region across multiple devices.
    """

    def __init__(self):
        super().__init__()
        self.world_size = get_moe_tp_world_size()
        if self.world_size > 1:
            self.moe_tp_group = get_moe_tensor_parallel_group()
            self.all_gather_into_tensor = ops.AllGather(group=self.moe_tp_group)


class ReduceFromWorldParallelRegion(nn.Cell):
    "All reduce the input from the world parallel region."

    def __init__(self):
        super().__init__()
        self.world_size = get_group_size()
        if self.world_size > 1:
            self.world_group = GlobalComm.WORLD_COMM_GROUP
            self.all_reduce = ops.AllReduce(group=self.world_group)

    def construct(self, input_):
        if self.world_size == 1:
            return input_
        output = self.all_reduce(input_)
        return output


class ReduceFromModelParallelRegion(ReduceFromWorldParallelRegion):
    "All reduce the input from the model parallel region."

    def __init__(self):
        super().__init__()
        self.world_size = get_tp_world_size()
        if self.world_size > 1:
            self.tp_group = get_tensor_model_parallel_group()
            self.all_reduce = ops.AllReduce(group=self.tp_group)


class ReduceFromMoeTensorParallelRegion(ReduceFromWorldParallelRegion):
    "All reduce the input from the moe tensor parallel region."

    def __init__(self):
        super().__init__()
        self.world_size = get_moe_tp_world_size()
        if self.world_size > 1:
            self.moe_tp_group = get_moe_tensor_parallel_group()
            self.all_reduce = ops.AllReduce(group=self.moe_tp_group)


class ReduceScatterToWorldParallelRegion(nn.Cell):
    "Reduce scatter the input from the world parallel region."

    def __init__(self):
        super().__init__()
        self.world_size = get_group_size()
        if self.world_size > 1:
            self.world_group = GlobalComm.WORLD_COMM_GROUP
            self.reduce_scatter_tensor = ops.ReduceScatter(group=self.world_group)

    def construct(self, input_):
        if self.world_size == 1:
            return input_
        output = self.reduce_scatter_tensor(input_)
        return output


class ReduceScatterToSequenceParallelRegion(ReduceScatterToWorldParallelRegion):
    "Reduce scatter the input from the model parallel region."

    def __init__(self):
        super().__init__()
        self.world_size = get_tp_world_size()
        if self.world_size > 1:
            self.tp_group = get_tensor_model_parallel_group()
            self.reduce_scatter_tensor = ops.ReduceScatter(group=self.tp_group)


class ReduceScatterToMoeTensorParallelRegion(ReduceScatterToWorldParallelRegion):
    "Reduce scatter the input from the moe tensor parallel region."
    def __init__(self):
        super().__init__()
        self.world_size = get_moe_tp_world_size()
        if self.world_size > 1:
            self.moe_tp_group = get_moe_tensor_parallel_group()
            self.reduce_scatter_tensor = ops.ReduceScatter(group=self.moe_tp_group)


class ScatterToWorldParallelRegion(nn.Cell):
    "Split the input and keep only the corresponding chuck to the rank in world parallel region."

    def __init__(self, axis=-1):
        super().__init__()
        self.world_size = get_group_size()
        if self.world_size > 1:
            self.rank = get_rank()
            self.split = ops.Split(axis=axis, output_num=self.world_size)

    def construct(self, input_):
        if self.world_size == 1:
            return input_
        tensor_tuple = self.split(input_)
        output = tensor_tuple[self.rank]
        return output


class ScatterToModelParallelRegion(ScatterToWorldParallelRegion):
    "Split the input and keep only the corresponding chuck to the rank in model parallel region."

    def __init__(self, axis=-1):
        super().__init__(axis)
        self.world_size = get_tp_world_size()
        if self.world_size > 1:
            self.rank = get_tensor_model_parallel_rank()
            self.split = ops.Split(axis=axis, output_num=self.world_size)


class ScatterToMoeTensorParallelRegion(ScatterToWorldParallelRegion):
    "Split the input and keep only the corresponding chuck to the rank in moe tensor parallel region."

    def __init__(self, axis=-1):
        super().__init__(axis)
        self.world_size = get_moe_tp_world_size()
        if self.world_size > 1:
            self.rank = get_moe_tensor_parallel_rank()
            self.split = ops.Split(axis=axis, output_num=self.world_size)


class ScatterToDataParallelRegion(ScatterToWorldParallelRegion):
    "Split the input and keep only the corresponding chuck to the rank in data parallel region."

    def __init__(self, axis=-1):
        super().__init__()
        self.world_size = get_dp_world_size()
        if self.world_size > 1:
            self.rank = get_data_parallel_rank()
            self.split = ops.Split(axis=axis, output_num=self.world_size)
