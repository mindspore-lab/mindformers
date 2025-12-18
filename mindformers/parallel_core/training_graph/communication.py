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
"""Communication utilities for parallel training."""
import hashlib
from typing import Tuple, List

from mindspore.communication import create_group

from mindformers.checkpoint.sharded_tensor import ShardedTensor
from mindformers.tools.logger import logger


def compute_repeat_num_and_model_parallel_size(sharded_info: ShardedTensor, world_size: int, pp: int, op: int):
    """Compute real op size."""
    axis_fragmentations = sharded_info.axis_fragmentations
    flag = False
    weight_sharded_size = 1
    for axis in axis_fragmentations:
        if axis == 1:
            continue
        if flag:
            raise ValueError("Only one axis can be fragmented in Muon optimizer.")
        flag = True
        weight_sharded_size *= axis
    repeat_num = world_size // pp // weight_sharded_size
    real_op_size = min(op, repeat_num)
    if sharded_info.local_shape[0] % real_op_size != 0:
        real_op_size = 1
    return real_op_size, weight_sharded_size


def create_communication_group(rank_list):
    """
    Create a communication group with a hashed name.

    Args:
        rank_list: List of ranks in the communication group

    Returns:
        str: The created group name
    """
    rank_list_str = "-".join([str(i) for i in rank_list])
    hashed = hashlib.md5(rank_list_str.encode()).hexdigest()[:48]
    group_name = str(hashed)
    create_group(group_name, rank_list)
    return group_name


OP_GROUP_NAME = {}
CP_GROUP_NAME = {}
DP_GROUP_NAME = {}


def get_cp_group_name(rank_id: int, dp: int, tp: int, cp: int) -> Tuple[str, List[int]]:
    """
    Get the CP (Context Parallel) communication group name and rank list.

    Under the rank encoding where DP is the highest bit, CP is the second bit,
    and TP is the lowest bit, return the CP communication domain rank_list for the current rank.

    Args:
        rank_id (int): Current rank ID.
        dp (int): Data parallel size.
        tp (int): Tensor parallel size.
        cp (int): Context parallel size.

    Returns:
        Tuple[str, List[int]]: Communication group name and rank list.
    """
    cache_key = (rank_id, dp, tp, cp)
    if cache_key in CP_GROUP_NAME:
        return CP_GROUP_NAME[cache_key]

    pp_block = dp * cp * tp
    inner = cp * tp

    pp_base = (rank_id // pp_block) * pp_block
    local = rank_id % pp_block

    dp_id = local // inner
    tp_id = local % tp

    base = pp_base + dp_id * inner + tp_id
    rank_list = [base + c * tp for c in range(cp)]
    logger.info(f"Get cp rank list: {rank_list}")
    result = (create_communication_group(rank_list), rank_list)
    CP_GROUP_NAME[cache_key] = result
    return result


def get_dp_group_name(rank_id: int, dp: int, tp: int, cp: int) -> Tuple[str, List[int]]:
    """
    Get the DP (Data Parallel) communication group name and rank list.

    Under the rank encoding where DP is the highest bit, CP is the second bit,
    and TP is the lowest bit, return the DP communication domain rank_list for the current rank.

    Args:
        rank_id (int): Current rank ID.
        dp (int): Data parallel size.
        tp (int): Tensor parallel size.
        cp (int): Context parallel size.

    Returns:
        Tuple[str, List[int]]: Communication group name and rank list.
    """
    cache_key = (rank_id, dp, tp, cp)
    if cache_key in DP_GROUP_NAME:
        return DP_GROUP_NAME[cache_key]

    pp_block = dp * cp * tp
    inner = cp * tp

    pp_base = (rank_id // pp_block) * pp_block
    local = rank_id % pp_block
    base_low = local % inner
    rank_list = [pp_base + base_low + d * inner for d in range(dp)]
    logger.info(f"Get dp rank list: {rank_list}")
    result = (create_communication_group(rank_list), rank_list)
    DP_GROUP_NAME[cache_key] = result
    return result


def get_op_group_name(rank_id: int, real_op_size: int, model_parallel_size: int):
    """Get op group name."""
    if (rank_id, real_op_size, model_parallel_size) in OP_GROUP_NAME:
        return OP_GROUP_NAME[(rank_id, real_op_size, model_parallel_size)]
    dp_range = model_parallel_size
    op_range = model_parallel_size * real_op_size
    rank_start = rank_id % dp_range + rank_id // op_range * op_range
    rank_end = rank_start + op_range
    rank_list = list(range(rank_start, rank_end, dp_range))
    op_group_name = create_communication_group(rank_list)
    OP_GROUP_NAME[(rank_id, real_op_size, model_parallel_size)] = (op_group_name, rank_list)
    return op_group_name, rank_list
