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
from mindspore.nn import Cell
from mindspore.parallel.shard import _DistributedTensorInfo
from mindspore.parallel.strategy import get_current_strategy_metadata, get_strategy_metadata

from mindformers.tools.utils import get_real_rank, get_real_group_size
from mindformers.tools.logger import logger


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

    org_key: str
    """
    Record the original weight key name.
    Mostly used in load Hugging Face weight with online resharding.
    """

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


def build_sharded_tensor(
        param_name: str, param_dtype: ms.dtype, local_shape: Tuple[int, ...], global_shape: Tuple[int, ...],
        axis_fragmentations: Tuple[int, ...], global_offset: Tuple[int, ...], replica_id: ReplicaId = 0,
        allow_shape_mismatch: bool = False, allow_to_save: bool = True, layout: Optional[ms.Layout] = None
) -> ShardedTensor:
    """Creates and returns a ShardedTensor instance with the specified parameters."""
    return ShardedTensor(
        key=param_name, org_key=param_name, dtype=param_dtype, local_shape=tuple(local_shape),
        global_shape=tuple(global_shape), global_offset=tuple(global_offset),
        axis_fragmentations=tuple(axis_fragmentations), replica_id=replica_id,
        allow_shape_mismatch=allow_shape_mismatch, allow_to_save=allow_to_save, layout=layout
    )


def get_strategy_info_from_sharded_tensor(sharded_tensor: ShardedTensor):
    """get strategy info from sharded tensor"""
    return sharded_tensor.global_shape, sharded_tensor.axis_fragmentations, sharded_tensor.global_offset


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


def _rank_id_with_slice_id(alias_rank_stride):
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


def _get_global_offset_and_replica_id_from_layout(layout):
    """
    Extracts rank-specific global offset and replica IDs from the distributed tensor layout.

    Parses the layout's device matrix, alias name, tensor map, and rank list to compute the global offset
    (starting index of the local slice in the global tensor) and replica IDs (identifies the replica of the
    local slice across sharded dimensions).

    Args:
        layout: Distributed tensor layout object (must implement `to_dict()` method) containing sharding
            configuration details.

    Returns:
        Tuple[List[int], List[int]]: A tuple with two elements:
            1. global_offset: List of integers representing the base global offset for the tensor's sharding
            2. cur_replica_id: List of integers where each element is the replica index for the corresponding
                sharded dimension
    """
    layout_dict = layout.to_dict()
    dev_matrix = layout_dict.get("device_matrix")
    alias_name = layout_dict.get("alias_name")
    tensor_map = layout_dict.get("tensor_map")
    rank_list = layout_dict.get("rank_list")

    dev_arrange = _alias_name_with_rank_id(dev_matrix, alias_name, rank_list)
    flat_tensor_map = _flatten_tensor_map(tensor_map)
    alias_rank_stride = _tensor_map_with_rank_id(
        dev_matrix, flat_tensor_map, alias_name, dev_arrange
    )

    rank_slice_table, global_offset = _rank_id_with_slice_id(alias_rank_stride)
    slice_cnt: Dict[int, int] = defaultdict(int)
    cur_replica_id: List[int] = []
    for _, slice_id in enumerate(rank_slice_table):
        replica_id = slice_cnt[slice_id]
        slice_cnt[slice_id] += 1
        cur_replica_id.append(replica_id)
    return global_offset, cur_replica_id


def get_sharded_tensor_from_strategy_metadata(
        param_infos: Dict[str, List],
        cur_npu_rank: int,
        filter_func: Callable[[str], bool] = None
) -> Optional[Dict[str, ShardedTensor]]:
    """
    Creates ShardedTensor instances for the current NPU rank based on distributed strategy metadata.

    Processes parameter metadata (layout, dtype, global shape) to construct sharded tensors tailored to the current
    NPU rank. Applies an optional filter to select specific parameters, computes rank-specific sharding details
    (local shape, global offset, replica ID), and builds ShardedTensor objects using the provided metadata.

    Args:
        param_infos: A dictionary mapping parameter names (str) to their distributed metadata. Each value is a list
            containing three elements in order:
                1. layout: Distributed tensor layout object (supports `to_dict()` method) containing sharding
                   configuration (device matrix, alias name, tensor map, rank list)
                2. dtype: Data type of the parameter (e.g., torch.float32, numpy.float64)
                3. global_shape: Tuple of integers representing the full global shape of the unsharded tensor
        cur_npu_rank: Integer indicating the current NPU rank index (used to compute rank-specific global offset)
        filter_func: Optional callable that takes a parameter name (str) and returns a boolean. If provided, only
            parameters for which the function returns True are included in the output. Defaults to None (all parameters
            included).

    Returns:
        Optional[Dict[str, ShardedTensor]]: A dictionary where keys are parameter names (filtered if `filter_func` is
        provided) and values are corresponding ShardedTensor instances for the current NPU rank. Returns None if the
        input `param_infos` is empty.
    """
    if not param_infos:
        return None

    cur_rank_sharded_tensor_dict = {}
    for param_name, param_info in param_infos.items():
        if filter_func and not filter_func(param_name):
            continue

        layout, dtype, global_shape = param_info
        distributed_info = _DistributedTensorInfo(layout)
        strategy = distributed_info.sharding_strategy
        axis_fragmentations = tuple(int(x) for x in strategy)
        local_shape = tuple(int(s // c) for s, c in zip(global_shape, axis_fragmentations))
        global_offset, replica_id = _get_global_offset_and_replica_id_from_layout(layout)
        npu_nums_per_pp = len(global_offset)
        global_offset = (global_offset[cur_npu_rank % npu_nums_per_pp],)

        cur_sharded_tensor = build_sharded_tensor(
            param_name=param_name,
            param_dtype=dtype,
            local_shape=local_shape,
            global_shape=global_shape,
            global_offset=global_offset,
            axis_fragmentations=axis_fragmentations,
            replica_id=replica_id,
            allow_shape_mismatch=False,
            allow_to_save=True,
            layout=layout
        )
        cur_rank_sharded_tensor_dict[param_name] = cur_sharded_tensor

    return cur_rank_sharded_tensor_dict


def get_sharded_tensor_from_cell(
        network: Cell,
        optimizer: Optional[Cell] = None,
) -> Dict[str, ShardedTensor]:
    """
    Extracts sharded tensor metadata from a network cell and optional optimizer cell.

    Collects parameter information from the network and optimizer (if provided)
    to create ShardedTensor objects containing metadata like dtype, shape, and
    fragmentation information. Parameters from the optimizer that already exist
    in the network are ignored.

    Args:
        network: The main network Cell containing parameters
        optimizer: Optional optimizer Cell containing additional parameters

    Returns:
        Dict of ShardedTensor objects with metadata from network and optimizer parameters
    """
    logger.info(".........Get Current Strategy Metadata from Cell.........")
    sharded_tensor_dict: Dict[str, ShardedTensor] = {}

    def _get_sharded_tensors_from_cell(
            cell: Cell, ignore_params_list: Optional[List[str]] = None
    ) -> Dict[str, ShardedTensor]:
        """
        Helper function to extract sharded tensors from a single Cell.

        Creates ShardedTensor objects for each parameter in the cell, skipping
        any parameters in the ignore list.

        Args:
            cell: The Cell to extract parameters from
            ignore_params_list: Optional list of parameter names to skip

        Returns:
            Dict of ShardedTensor objects for the cell's parameters
        """
        cur_cell_sharded_tensor_dict = {}
        for param in cell.get_parameters():
            param_name = param.name

            # Skip parameters in the ignore list if provided
            if ignore_params_list and param_name in ignore_params_list:
                continue

            # Extract parameter properties
            param_dtype = param.data.dtype
            param_shape = param.data.shape
            global_offset = (0,)
            axis_fragmentations = (1,) * len(param_shape)

            # Create and add sharded tensor metadata
            sharded_tensor = build_sharded_tensor(
                param_name=param_name,
                param_dtype=param_dtype,
                local_shape=param_shape,
                global_shape=param_shape,
                global_offset=global_offset,
                axis_fragmentations=axis_fragmentations
            )
            cur_cell_sharded_tensor_dict[param_name] = sharded_tensor

        return cur_cell_sharded_tensor_dict

    # Get sharded tensors from the main network
    sharded_tensor_dict.update(_get_sharded_tensors_from_cell(network))

    # Add sharded tensors from optimizer if provided, ignoring network parameters
    if optimizer:
        # Create list of parameter names already collected from network
        ignore_params_list = list(sharded_tensor_dict.keys())
        # Get optimizer parameters, skipping those already in network
        sharded_tensor_dict.update(
            _get_sharded_tensors_from_cell(optimizer, ignore_params_list)
        )

    return sharded_tensor_dict


def get_all_sharded_tensor(
        network: Cell,
        filter_func: Callable[[str], bool] = None
) -> Dict[int, Dict[str, ShardedTensor]]:
    """
    Collects sharded tensor metadata for all ranks in the parallel group from the MindSpore network.

    Retrieves global distributed strategy metadata from the input network, then generates rank-specific
    ShardedTensor instances for every rank in the parallel group. Applies an optional filter to select
    target parameters (e.g., exclude non-trainable weights) during ShardedTensor creation.

    Args:
        network (Cell): A MindSpore Network Cell containing distributed parameters and their sharding strategy.
        filter_func (Optional[Callable[[str], bool]]): An optional filtering function that takes a parameter name (str)
            and returns a boolean. Only parameters for which the function returns `True` are included in the
            ShardedTensor collection. Defaults to `None` (all eligible parameters are included).

    Returns:
        Dict[int, Dict[str, ShardedTensor]]: A nested dictionary where:
        - Outer keys: Rank IDs (int) in the parallel group (range: `[0, total_ranks - 1]`).
        - Outer values: Dictionaries mapping parameter names (str) to their corresponding `ShardedTensor` instances,
            containing rank-specific metadata (local shape, global offset, dtype, etc.).

    Raises:
        RuntimeError: If `get_strategy_metadata` returns `None`, indicating no distributed strategy metadata is
            associated with the network.
    """
    logger.info(".........Get All Ranks' Strategy Metadata.........")
    global_strategy_info = get_strategy_metadata(network)
    if not global_strategy_info:
        raise RuntimeError('`get_strategy_metadata` returns `None`, which indicates there is no strategy info. '
                           'Please check whether this is a distributed job.')

    npu_nums = get_real_group_size()
    sharded_tensor_metas: Dict[int, Dict[str, ShardedTensor]] = {}
    for cur_npu_rank in range(0, npu_nums):
        cur_rank_strategy_layout = global_strategy_info[cur_npu_rank]

        # Get Sharded tensors from strategy metadata of current rank.
        cur_rank_sharded_tensors = get_sharded_tensor_from_strategy_metadata(
            param_infos=cur_rank_strategy_layout,
            cur_npu_rank=cur_npu_rank,
            filter_func=filter_func
        )

        sharded_tensor_metas[cur_npu_rank] = cur_rank_sharded_tensors
    return sharded_tensor_metas


def get_cur_sharded_tensor(
        network: Cell,
        filter_func: Callable[[str], bool] = None
) -> Dict[str, ShardedTensor]:
    """
    Retrieves rank-specific ShardedTensor instances for the current NPU rank from the MindSpore network.

    Args:
        network (Cell): A MindSpore Network Cell containing distributed parameters and their sharding strategy.
        filter_func (Optional[Callable[[str], bool]]): An optional filtering function that takes a parameter name (str)
            and returns a boolean. Only parameters for which the function returns `True` are included in the
            output. Defaults to `None` (all eligible parameters assigned to the current rank are included).

    Returns:
        Dict[str, ShardedTensor]: A dictionary where keys are parameter names (str) and values are `ShardedTensor`
        instances.
    """
    logger.info(".........Get Current Strategy Metadata.........")
    strategy_info = get_current_strategy_metadata(network)[0]
    # Get sharded tensors from strategy metadata
    cur_rank_sharded_tensors = get_sharded_tensor_from_strategy_metadata(
        param_infos=strategy_info, cur_npu_rank=get_real_rank(), filter_func=filter_func
    )
    return cur_rank_sharded_tensors


def get_cur_sharded_tensor_after_balanced(
        rank_id_to_sharded_tensors: Dict[int, Dict[str, Tuple]]
) -> Dict[str, ShardedTensor]:
    """
    Retrieves the load-balanced ShardedTensor instances assigned to the current rank.

    Args:
        rank_id_to_sharded_tensors: A nested dictionary representing the load-balanced shard distribution across ranks:
        - Outer keys: Rank IDs (int) in the parallel group.
        - Outer values: Dictionaries mapping unique shard IDs (str) to tuples containing:
        1. Target `ShardedTensor` instance (with rank-specific metadata like local shape, global offset, etc.).
        2. Rank group (Tuple[int, ...]): Redundant ranks with copies of the shard (ignored in this function).

    Returns:
        Dict[str, ShardedTensor]: A dictionary where keys are parameter names (str) and values are `ShardedTensor`
        instances.
    """
    cur_rank_sharded_tensors = {}
    local_rank = get_real_rank()
    sharded_tensors = rank_id_to_sharded_tensors[local_rank]
    for _, shard_id_info in sharded_tensors.items():
        sharded_tensor, _ = shard_id_info
        param_name = sharded_tensor.key
        cur_rank_sharded_tensors[param_name] = sharded_tensor
    return cur_rank_sharded_tensors


def get_param_redundancy_after_balanced(
        rank_id_to_sharded_tensors: Dict[int, Dict[str, Tuple]]
) -> Dict[Tuple, List]:
    """
    Identifies redundant parameter shards for the current rank from a load-balanced shard distribution.

    Args:
        rank_id_to_sharded_tensors: A nested dictionary representing the load-balanced shard distribution across ranks:
            - Outer keys: Rank IDs (int) in the parallel group.
            - Outer values: Dictionaries mapping unique shard IDs (str) to tuples containing:
                1. `ShardedTensor` instance (with `key` attribute specifying the original parameter name).
                2. Rank group (Tuple[int, ...]): Sorted tuple of ranks that store redundant copies of the shard.

    Returns:
        Dict[Tuple[int, ...], List[str]]: A dictionary where:
            - Keys: Rank groups (Tuple[int, ...]) – sets of ranks that share redundant copies of parameters.
              Only groups containing the current rank are included.
            - Values: Lists of parameter names that are redundantly stored across the corresponding rank group.
    """
    param_redundancy = {}
    local_rank = get_real_rank()
    for _, sharded_tensors in rank_id_to_sharded_tensors.items():
        for _, shard_id_info in sharded_tensors.items():
            sharded_tensor, rank_group = shard_id_info
            param_name = sharded_tensor.key
            if len(rank_group) == 1:
                continue
            if local_rank in rank_group:
                param_redundancy.setdefault(tuple(rank_group), []).append(param_name)
    return param_redundancy
