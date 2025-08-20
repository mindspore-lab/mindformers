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
"""Sharded Tensor"""

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union, Callable

import mindspore as ms
from mindspore.parallel.shard import _DistributedTensorInfo

ReplicaId = Union[int, Tuple[int, ...]]


@dataclass
class ShardedTensor:
    """
    Represents a mapping between a local tensor and a global tensor.

    Global tensor is assumed to consist of many local tensors distributed
    between different processes.
    """

    key: str
    """Unique identifier of a global tensor."""

    dtype: ms.dtype
    """Tensor dtype."""

    local_shape: Tuple[int, ...]
    """Local tensor shape."""

    global_shape: Tuple[int, ...]
    """Global tensor shape."""

    global_offset: Tuple[int, ...]
    """Offset of a local tensor in a global tensor, specified in number of tensor elements."""

    axis_fragmentations: Optional[Tuple[int, ...]]
    """Global tensor fragmentation of each axis."""

    replica_id: ReplicaId = 0
    """
    Indicates given local tensor's replication wrt.
    Local tensors in different processes.
    """

    allow_shape_mismatch: bool = False
    """
    If True, during loading, the global shape of a stored tensor does not have to match the expected global shape.
    Useful for representing tensors with flexible shape, e.g. padded.
    """

    allow_to_save: bool = True
    """If True, during saving, the tensor has to be saved in files."""

    layout: ms.Layout = None
    """Mindspore parallel layout describes the detailed sharding information."""


def is_main_replica(replica_id: ReplicaId):
    """Checks if given `replica_id` is considered as main.

    "Main" replica is:
    - integer 0
    - or an iterable with all 0 elements

    It is the application responsibility to set correct replicas for sharded tensors.

    Args:
        replica_id (Union[int, Tuple[int, ...]]): replica id

    Returns:
        (bool): True for a "main" replica
    """
    if isinstance(replica_id, int):
        return replica_id == 0
    return all(r == 0 for r in replica_id)


def _alias_name_with_rank_id(cur_dev_matrix, cur_alias_name, cur_rank_list) -> Dict:
    """Alias_name vs Rank_list, fill with 0 or 1, from right to left."""
    alias_slice_rank_dict = {}
    stride = [1] * len(cur_dev_matrix)

    for dev_dim in range(len(cur_dev_matrix) - 1, -1, -1):
        axis_alias_name = cur_alias_name[dev_dim]
        axis_dev_num = cur_dev_matrix[dev_dim]
        if dev_dim == len(cur_dev_matrix) - 1:
            stride[dev_dim] = 1
        else:
            stride[dev_dim] = stride[dev_dim + 1] * cur_dev_matrix[dev_dim + 1]
        cur_stride = stride[dev_dim]

        if axis_dev_num == 1:
            axis_rank_table = [0] * len(cur_rank_list)
            alias_slice_rank_dict[axis_alias_name] = [axis_dev_num, axis_rank_table]
            continue

        axis_rank_table = []
        j = 0
        while j < len(cur_rank_list):
            for i in range(axis_dev_num):
                axis_rank_table.extend([i] * cur_stride)
            j += cur_stride * axis_dev_num
        alias_slice_rank_dict[axis_alias_name] = [axis_dev_num, axis_rank_table]

    return alias_slice_rank_dict


def _flatten_tensor_map(cur_tensor_map) -> List:
    """Flatten tensor_map."""
    if len(cur_tensor_map) == 1 and isinstance(cur_tensor_map[0], (list, tuple)):
        cur_tensor_map = cur_tensor_map[0]
    flat_tensor_map = []

    for item in cur_tensor_map:
        if isinstance(item, (list, tuple)):
            flat_tensor_map.extend(_flatten_tensor_map(item))  # Recursive call
        else:
            flat_tensor_map.append(item)

    return flat_tensor_map


def _tensor_map_with_rank_id(cur_dev_matrix, flat_tensor_map, cur_alias_name, dev_arrange) -> List:
    """Tensor_map vs Rank_list, from right to left."""
    alias_rank_stride = [None] * len(flat_tensor_map)

    if not dev_arrange.values():
        return alias_rank_stride  # Handle empty case

    rank_num = len(next(iter(dev_arrange.values()))[1])
    dev_arrange[None] = [1, [0] * rank_num]
    axis_dev_num = [1] * len(flat_tensor_map)
    stride = [1] * len(flat_tensor_map)
    axis_rank_table = [None] * len(flat_tensor_map)

    for pos in range(len(flat_tensor_map) - 1, -1, -1):
        tensor_map_dim = flat_tensor_map[pos]
        if tensor_map_dim == -1:
            alias = None
        else:
            # Assuming tensor_map_dim maps to device dim index.
            # This mapping logic might need adjustment based on actual layout spec.
            dev_dim = len(cur_dev_matrix) - 1 - tensor_map_dim
            if 0 <= dev_dim < len(cur_alias_name):
                alias = cur_alias_name[dev_dim]
            else:
                alias = None

        axis_dev_num[pos] = dev_arrange.get(alias, [1])[0]
        axis_rank_table[pos] = dev_arrange.get(alias, [1, [0] * rank_num])[1]

        if pos == len(flat_tensor_map) - 1:
            stride[pos] = 1
        else:
            stride[pos] = stride[pos + 1] * axis_dev_num[pos + 1]
        alias_rank_stride[pos] = [axis_rank_table[pos], stride[pos]]

    return alias_rank_stride


def _rank_id_with_slice_id(alias_rank_stride) -> List:
    """Rank_id vs Slice_id, rank_id ranges from 0 to dev_num."""
    cur_global_offset: Tuple[int, ...] = ()
    rank_num = len(alias_rank_stride[0][0])
    rank_slice_table = [None] * rank_num
    for i in range(rank_num):
        slice_num = 0
        for elem in alias_rank_stride:
            if elem and elem[0] and len(elem[0]) > i:
                stride = elem[1]
                cur_rank = elem[0][i]
                slice_num += stride * cur_rank
        rank_slice_table[i] = slice_num
    cur_global_offset += tuple(int(x) for x in rank_slice_table)
    return rank_slice_table, cur_global_offset


def get_param_name_from_layout(param_infos: List[Dict]) -> List[str]:
    """Extract parameter names."""
    names = []

    for param_dict in param_infos:
        for param_name, _ in param_dict.items():
            names.append(param_name)

    return names


def get_value_type_from_layout(param_infos: List[Dict]) -> List[type]:
    """Extract parameter types."""
    types = []

    for param_dict in param_infos:
        for _, (_, param_type, _) in param_dict.items():
            types.append(param_type)

    return types


def get_local_shape_from_layout(param_infos: List[Dict]) -> List[Tuple[int, ...]]:
    """Compute local (sharded) shape on current device."""
    shapes = []

    for param_dict in param_infos:
        for _, (cur_layout, _, cur_shape) in param_dict.items():
            distributed_info = _DistributedTensorInfo(cur_layout)
            cur_stra = distributed_info.sharding_strategy
            shapes.append(tuple(int(s // c) for s, c in zip(cur_shape, cur_stra)))

    return shapes


def get_global_shape_from_layout(param_infos: List[Dict]) -> List[Tuple[int, ...]]:
    """Extract global shapes."""
    shapes = []

    for param_dict in param_infos:
        for _, (_, _, cur_shape) in param_dict.items():
            shapes.append(cur_shape)

    return shapes


def get_axis_fragmentations_from_layout(param_infos: List[Dict]) -> List[Tuple[int, ...]]:
    """Extract axis fragmentations (effective sharding strategies per tensor axis)."""
    fragmentations = []

    for param_dict in param_infos:
        for _, (cur_layout, _, _) in param_dict.items():
            distributed_info = _DistributedTensorInfo(cur_layout)
            cur_stra = distributed_info.sharding_strategy
            fragmentations.append(tuple(int(x) for x in cur_stra))

    return fragmentations


def get_global_offset_from_layout(param_infos: List[Dict]) -> List[Tuple[int, ...]]:
    """Calculate global offsets (starting index in full tensor) for current device."""
    offsets = []

    for param_dict in param_infos:
        for _, (cur_layout, _, _) in param_dict.items():
            cur_layout_dict = cur_layout.to_dict()
            cur_dev_matrix = cur_layout_dict.get("device_matrix")
            cur_alias_name = cur_layout_dict.get("alias_name")
            cur_tensor_map = cur_layout_dict.get("tensor_map")
            cur_rank_list = cur_layout_dict.get("rank_list")

            dev_arrange = _alias_name_with_rank_id(cur_dev_matrix, cur_alias_name, cur_rank_list)
            flat_tensor_map = _flatten_tensor_map(cur_tensor_map)
            alias_rank_stride = _tensor_map_with_rank_id(
                cur_dev_matrix, flat_tensor_map, cur_alias_name, dev_arrange
            )

            _, cur_global_offset = _rank_id_with_slice_id(alias_rank_stride)
            offsets.append(cur_global_offset)

    return offsets


def get_replica_id_from_layout(param_infos: List[Dict]) -> List[List[int]]:
    """Determine replica ID for each device (0 for primary, 1 for duplicate, etc.)."""
    replica_ids = []

    for param_dict in param_infos:
        for _, (cur_layout, _, _) in param_dict.items():
            cur_layout_dict = cur_layout.to_dict()
            cur_dev_matrix = cur_layout_dict.get("device_matrix")
            cur_alias_name = cur_layout_dict.get("alias_name")
            cur_tensor_map = cur_layout_dict.get("tensor_map")
            cur_rank_list = cur_layout_dict.get("rank_list")

            dev_arrange = _alias_name_with_rank_id(cur_dev_matrix, cur_alias_name, cur_rank_list)
            flat_tensor_map = _flatten_tensor_map(cur_tensor_map)
            alias_rank_stride = _tensor_map_with_rank_id(
                cur_dev_matrix, flat_tensor_map, cur_alias_name, dev_arrange
            )

            rank_slice_table, _ = _rank_id_with_slice_id(alias_rank_stride)

            slice_cnt: Dict[int, int] = defaultdict(int)
            cur_replica_id: List[int] = []
            for _, slice_id in enumerate(rank_slice_table):
                replica_id = slice_cnt[slice_id]
                slice_cnt[slice_id] += 1
                cur_replica_id.append(replica_id)
            replica_ids.append(cur_replica_id)

    return replica_ids


def get_sharded_tensor_list_from_strategy_metadata(param_infos: List[Dict], cur_npu_rank: int,
                                                   filter_func: Callable[[str], bool]) -> Optional[List[ShardedTensor]]:
    """
    Transform distributed strategy of a network to a list of ShardedTensor.

    Args:
        param_infos (List[Dict]): The distributed strategy of a rank of network.
        cur_npu_rank (int): Current Rank ID of NPUs.
        filter_func (Callable[[str], bool]): A filter function
            that decide whether to save metadata info of optimizer weight.

    Returns:
        A list of ShardedTensor or None: A list containing sharded tensor metadata, or None if no param_infos.
    """
    if not param_infos:
        return None

    cur_rank_sharded_tensor_list = list()

    cur_param_name_list = get_param_name_from_layout(param_infos)
    cur_value_type_list = get_value_type_from_layout(param_infos)
    cur_local_shape_list = get_local_shape_from_layout(param_infos)
    cur_global_shape_list = get_global_shape_from_layout(param_infos)
    cur_axis_fragmentations_list = get_axis_fragmentations_from_layout(param_infos)
    cur_global_offset_list = get_global_offset_from_layout(param_infos)
    cur_replica_id_list = get_replica_id_from_layout(param_infos)

    for idx, param_name in enumerate(cur_param_name_list):
        # If not save optimizer weight, the metadata will also not save the optimizer part.
        if filter_func and not filter_func(param_name):
            continue

        org_global_offset = cur_global_offset_list[idx]
        npu_nums_per_pp = len(org_global_offset)

        # The situation where different strategies need to be adapted later
        global_offset = (org_global_offset[cur_npu_rank % npu_nums_per_pp],)

        cur_sharded_tensor = ShardedTensor(
            key=param_name,
            dtype=cur_value_type_list[idx],
            local_shape=cur_local_shape_list[idx],
            global_shape=cur_global_shape_list[idx],
            global_offset=global_offset,
            axis_fragmentations=cur_axis_fragmentations_list[idx],
            replica_id=cur_replica_id_list[idx],
            allow_shape_mismatch=False,
            allow_to_save=True,
            layout=param_infos[idx][param_name][0]
        )
        cur_rank_sharded_tensor_list.append(cur_sharded_tensor)

    return cur_rank_sharded_tensor_list
