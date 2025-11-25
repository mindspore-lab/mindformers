# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Copyright 2025 Huawei Technologies Co., Ltd
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
""" utils """

import hashlib
from mindspore import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode
from mindspore.communication.management import get_rank, get_group_size, create_group

GROUP_NAME = {}


def get_group(rank_list):
    """check whether a group has been created."""
    rank_list_str = "-".join([str(i) for i in rank_list])
    if rank_list_str in GROUP_NAME:
        return GROUP_NAME[rank_list_str]

    hashed = hashlib.sha256(rank_list_str.encode()).hexdigest()[:48]
    group_name = str(hashed)
    create_group(group_name, rank_list)
    GROUP_NAME[rank_list_str] = group_name
    return group_name


def get_dp_mod_ep_group_name(data_parallel_size, expert_model_parallel_size):
    """Create MoE data parallel group across DP."""
    rank_id = get_rank()
    world_size = get_group_size()
    dp_group_id = rank_id // data_parallel_size

    start_rank = dp_group_id * data_parallel_size
    end_rank = min(start_rank + data_parallel_size, world_size)

    rank_list = []
    ep_group_id_in_dp = (rank_id % data_parallel_size) % expert_model_parallel_size
    for r in range(start_rank, end_rank):
        if r % expert_model_parallel_size == ep_group_id_in_dp:
            rank_list.append(r)
    return get_group(rank_list)


def get_ep_group_name(rank_id, expert_model_parallel_size):
    """Get expert model parallel group."""
    if _get_parallel_mode() == ParallelMode.STAND_ALONE:
        return None

    rank_start = rank_id // expert_model_parallel_size * expert_model_parallel_size
    rand_end = rank_id // expert_model_parallel_size * expert_model_parallel_size + expert_model_parallel_size
    rank_list = list(range(rank_start, rand_end))
    return get_group(rank_list)


def get_oep_group_name(rank_id, expert_model_parallel_size, npu_nums_per_device):
    """
    Generates a unique group name for a set of ranks involved in outer expert partitioning (oep)
    and creates a communication group with this name.
    This method calculates a range of ranks based on the current rank id
    and the expert partition size, hashes this range to create a unique
    identifier, and then establishes a new communication group using this identifier.
    """
    rank_start = rank_id // expert_model_parallel_size * expert_model_parallel_size
    rank_start = rank_start + rank_id % npu_nums_per_device
    rand_end = rank_start + expert_model_parallel_size
    rank_list = list(range(rank_start, rand_end, npu_nums_per_device))
    return get_group(rank_list)


def get_iep_group_name(rank_id, npu_nums_per_device):
    """
    Generates a unique group name for a set of ranks involved in inner expert partitioning (iep)
    and creates a communication group with this name.
    This method calculates a range of ranks based on the current rank id
    and the expert partition size, hashes this range to create a unique
    identifier, and then establishes a new communication group using this identifier.
    """
    rank_start = rank_id // npu_nums_per_device * npu_nums_per_device
    rand_end = rank_start + npu_nums_per_device
    rank_list = list(range(rank_start, rand_end))
    return get_group(rank_list)
