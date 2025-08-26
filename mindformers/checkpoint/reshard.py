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
"""resharding tensor"""
import operator
from typing import Dict, List, Tuple, Optional, Any, Union
from functools import reduce
import numpy as np

from mindspore import Tensor


def check_layout(layout: Optional[Any], name: str) -> None:
    """
    Validates that a layout contains required attributes with correct types.

    Args:
        layout: Layout object to validate
        name: Name of the layout (for error messages)

    Raises:
        ValueError: If layout missing required attributes or has size mismatches
        TypeError: If layout components are not tuples/lists
    """
    if not layout:
        return

    # Check for required attributes
    required_attrs = ['_device_shape', '_tensor_map', '_rank_list']
    for attr in required_attrs:
        if not hasattr(layout, attr):
            raise ValueError(
                f"Layout {name} must contain attribute {attr}"
            )

    # Validate component types
    def check_type_is_sequence(obj: Any, obj_name: str) -> None:
        if not isinstance(obj, (tuple, list)):
            raise TypeError(
                f"Layout {name} {obj_name} must be tuple or list, "
                f"but got {type(obj).__name__}"
            )

    layout_dict = layout.to_dict()
    check_type_is_sequence(layout_dict['device_matrix'], 'device_matrix')
    check_type_is_sequence(layout_dict['tensor_map'], 'tensor_map')
    check_type_is_sequence(layout_dict['rank_list'], 'rank_list')

    # Validate rank list size matches device count
    dev_num = reduce(operator.mul, layout_dict['device_matrix'])
    if len(layout_dict['rank_list']) != dev_num:
        raise ValueError(
            f"Layout {name} rank_list size ({len(layout_dict['rank_list'])}) "
            f"must match device count ({dev_num})"
        )


def rank_id_to_dev_id_list(dev_matrix: Tuple[int, ...], rank_id: int) -> List[int]:
    """
    Converts a rank ID to a list of device IDs based on the device matrix.

    Args:
        dev_matrix: Shape of the device matrix
        rank_id: Global rank ID to convert

    Returns:
        List of device IDs corresponding to the rank
    """
    dims = len(dev_matrix)
    dev_id_list = [0] * dims

    for i in range(dims - 1, -1, -1):
        dev_id_list[i] = rank_id % dev_matrix[i]
        rank_id = rank_id // dev_matrix[i]

    return dev_id_list


def infer_intersection(
        area_a: Tuple[Tuple[int, int], ...],
        area_b: Tuple[Tuple[int, int], ...]
) -> Optional[Tuple[Tuple[int, int], ...]]:
    """
    Calculates the intersection of two tensor slice areas.

    Args:
        area_a: First area to intersect
        area_b: Second area to intersect

    Returns:
        Tuple of intersection boundaries or None if no intersection
    """
    # Validate input formats
    def is_valid_axis_list(axis_list: Any) -> None:
        if not isinstance(axis_list, (tuple, list)):
            raise TypeError("Area must be a tuple of ranges")
        for axis_range in axis_list:
            if (not isinstance(axis_range, (tuple, list)) \
                or len(axis_range) != 2):
                raise TypeError("Each axis range must be a 2-element tuple")

    is_valid_axis_list(area_a)
    is_valid_axis_list(area_b)

    # Check dimension compatibility
    if len(area_a) != len(area_b):
        raise ValueError(
            f"Area dimension mismatch: {len(area_a)} vs {len(area_b)}"
        )

    # Calculate intersection for each dimension
    intersection: List[Tuple[int, int]] = []
    for axis_range_a, axis_range_b in zip(area_a, area_b):
        left = max(axis_range_a[0], axis_range_b[0])
        right = min(axis_range_a[1], axis_range_b[1])

        if left >= right:  # No intersection in this dimension
            return None

        intersection.append((left, right))

    return tuple(intersection)


def infer_slice_area_by_rank(
        dev_matrix: Tuple[int, ...],
        tensor_map: Union[List[int], Tuple[int, ...]],
        rank_id: int,
        full_shape: Tuple[int, ...]
) -> Tuple[Tuple[int, int], ...]:
    """
    Calculates the tensor slice boundaries for a specific rank.

    Args:
        dev_matrix: Shape of the device matrix
        tensor_map: Mapping of tensor dimensions to device dimensions
        rank_id: Rank ID to calculate slice for
        full_shape: Complete shape of the original tensor

    Returns:
        Tuple of (start, end) boundaries for each tensor dimension
    """
    # Helper to get device count along a dimension
    def _get_dev_num_along_dim(dim: int) -> int:
        return dev_matrix[-dim - 1] if dim != -1 else 1

    dims = len(full_shape)
    dev_id_list = rank_id_to_dev_id_list(dev_matrix, rank_id)
    area: List[Tuple[int, int]] = []

    for axis in range(dims):
        mapping = tensor_map[axis]
        if isinstance(mapping, int):
            mapping = (mapping,)  # Convert to tuple for consistent handling

        # Calculate total number of splits for this axis
        split_num = 1
        for dim in mapping:
            split_num *= _get_dev_num_along_dim(dim)

        # Calculate slice ID for this rank
        slice_id = 0
        coef = 1
        for dim in reversed(mapping):
            if dim == -1:
                continue
            slice_id += dev_id_list[-dim - 1] * coef
            coef *= _get_dev_num_along_dim(dim)

        # Calculate start/end indices for this slice
        slice_size = full_shape[axis] // split_num
        start = slice_id * slice_size
        end = start + slice_size
        area.append((start, end))

    return tuple(area)


class ReshardHandler:
    """
    Handles tensor resharding between different distributed layouts.

    This class manages the process of reshaping and redistributing tensors between
    different parallel layouts. It calculates necessary tensor slices, validates
    input layouts, and assembles the final tensor for the target rank.

    Args:
        param_name: Name of the parameter (without pipeline stage prefix)
        full_shape: Complete shape of the tensor before sharding
        from_layout: Source layout containing device matrix, tensor map, and rank list
        to_layout: Target layout containing device matrix, tensor map, and rank list
        to_rank_id: Target rank ID to receive the resharded tensor

    Raises:
        ValueError: If both layouts are None or layouts contain invalid attributes
        TypeError: If layout components are not tuples/lists
    """
    def __init__(
            self,
            param_name: str,
            full_shape: Tuple[int, ...],
            from_layout: Optional[Any],
            to_layout: Optional[Any],
            to_rank_id: int
    ):
        # Validate input layouts
        check_layout(from_layout, 'from_layout')
        check_layout(to_layout, 'to_layout')

        if from_layout is None and to_layout is None:
            raise ValueError("`from_layout` and `to_layout` cannot both be None.")

        # Initialize basic attributes
        self.param_name = param_name
        self.full_shape = full_shape

        # Process source layout configuration
        if from_layout is None:
            self.from_dev_matrix = (1,)
            self.from_tensor_map = tuple(0 for _ in full_shape)
            self.from_rank_list = [0]
        else:
            from_layout_dict = from_layout.to_dict()
            self.from_dev_matrix = from_layout_dict["device_matrix"]
            self.from_tensor_map = from_layout_dict["tensor_map"]
            self.from_rank_list = from_layout_dict["rank_list"]

        # Process target layout configuration
        if to_layout is None:
            self.to_dev_matrix = (1,)
            self.to_tensor_map = tuple(0 for _ in full_shape)
            self.to_rank_list = [0]
            self.to_rank_id = 0
        else:
            to_layout_dict = to_layout.to_dict()
            self.to_dev_matrix = to_layout_dict["device_matrix"]
            self.to_tensor_map = to_layout_dict["tensor_map"]
            self.to_rank_list = to_layout_dict["rank_list"]
            self.to_rank_id = to_rank_id

        # Calculate device counts and internal rank mappings
        self.from_dev_num = len(self.from_rank_list)
        self.inner_from_rank_list = range(self.from_dev_num)
        self.inner_to_rank_id = self.to_rank_list.index(self.to_rank_id)

        # Compute redundancy information
        self.inner_deredundancy_from_rank_list = (
            self._infer_inner_deredundancy_rank_list_by_from_layout()
            if from_layout else [0]
        )
        self.global_union_area_map: Dict[int, Tuple[Tuple[int, int], ...]] = {}

    def _infer_inner_deredundancy_rank_list_by_from_layout(self) -> List[int]:
        """
        Infers ranks containing non-redundant data from the source layout.

        Returns:
            List of ranks with unique data slices
        """
        inner_deredundancy_rank_list: List[int] = []
        from_dev_map = set()
        dev_dim = len(self.from_dev_matrix)

        # Collect relevant device dimensions from tensor map
        for map_dev in self.from_tensor_map:
            if isinstance(map_dev, (list, tuple)):
                for map_dev_inner in map_dev:
                    from_dev_map.add(dev_dim - map_dev_inner - 1)
            else:
                from_dev_map.add(dev_dim - map_dev - 1)

        # Filter ranks with non-redundant data
        for rank_id in self.inner_from_rank_list:
            dev_id_list = rank_id_to_dev_id_list(self.from_dev_matrix, rank_id)
            if any([
                    dim not in from_dev_map and dev_id_list[dim] > 0
                    for dim in range(dev_dim)
            ]):
                continue
            inner_deredundancy_rank_list.append(rank_id)

        return inner_deredundancy_rank_list

    def infer_all_tensor_offset(self) -> Dict[int, Tuple[Tuple[int, int], ...]]:
        """
        Calculates required tensor slices from each source rank.

        Determines which parts of the tensor need to be collected from each source
        rank to assemble the target tensor slice.

        Returns:
            Dictionary mapping source ranks to their required slice offsets
        """
        # Calculate target area for current rank
        self.to_area = infer_slice_area_by_rank(
            self.to_dev_matrix,
            self.to_tensor_map,
            self.inner_to_rank_id,
            self.full_shape
        )

        # Calculate required slices from each source rank
        local_union_areas_map: Dict[int, Tuple[Tuple[int, int], ...]] = {}
        self.global_union_area_map.clear()

        for inner_rank_id in self.inner_deredundancy_from_rank_list:
            # Get source area for this rank
            from_area = infer_slice_area_by_rank(
                self.from_dev_matrix,
                self.from_tensor_map,
                inner_rank_id,
                self.full_shape
            )

            # Find overlapping area between source and target
            union_area = infer_intersection(from_area, self.to_area)
            if union_area is not None:
                source_rank = self.from_rank_list[inner_rank_id]
                self.global_union_area_map[source_rank] = union_area

                # Calculate relative offsets within source slice
                local_union_areas_map[source_rank] = tuple(
                    (union_range[0] - from_range[0], union_range[1] - from_range[0])
                    for union_range, from_range in zip(union_area, from_area)
                )

        return local_union_areas_map

    def get_real_tensor(self, from_tensor_map: Dict[int, Tensor]) -> Tensor:
        """
        Assembles the final tensor for the target rank from collected slices.

        Args:
            from_tensor_map: Dictionary mapping source ranks to their tensor slices

        Returns:
            Assembled tensor for the target rank

        Raises:
            ValueError: If input slices are missing or have incorrect shapes
        """
        if not from_tensor_map:
            raise ValueError("Input from_tensor_map cannot be empty")

        # Validate input slices
        for from_rank_id, from_area in self.global_union_area_map.items():
            if from_rank_id not in from_tensor_map:
                raise ValueError(
                    f"Missing slice data from rank {from_rank_id}. "
                    "Please provide all required slices from infer_all_tensor_offset."
                )

            # Validate slice shape matches expected size
            expected_shape = tuple(end - start for start, end in from_area)
            actual_shape = from_tensor_map[from_rank_id].shape
            if expected_shape != actual_shape:
                raise ValueError(
                    f"Slice from rank {from_rank_id} has incorrect shape. "
                    f"Expected {expected_shape}, got {actual_shape}."
                )

        # Create target tensor and assign slices
        to_slice_shape = [end - start for start, end in self.to_area]
        dtype = next(iter(from_tensor_map.values())).dtype
        real_tensor = Tensor(np.zeros(to_slice_shape), dtype)

        for from_rank_id, from_slice in from_tensor_map.items():
            from_area = self.global_union_area_map[from_rank_id]

            # Calculate assignment indices in target tensor
            assign_slices = tuple(
                slice(from_axis[0] - to_axis[0], from_axis[1] - to_axis[0])
                for from_axis, to_axis in zip(from_area, self.to_area)
            )

            real_tensor[assign_slices] = from_slice

        return real_tensor
