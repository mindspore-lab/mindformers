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
"""Muon utils"""

import math
from fnmatch import fnmatch
from mindspore import ops as P
from mindspore.ops.operations import Morph
from mindspore import nn


class BlockSplitReshape(nn.Cell):
    """
    Reshape tensor by splitting its last dimension into blocks.
    
    This operation takes a tensor and splits its last dimension into equal-sized blocks,
    adding a new dimension for the block index.
    
    Args:
        block: Block size for splitting the last dimension.
    """

    def __init__(
            self,
            block
    ):
        super().__init__()
        self.block = block
        self.local_reshape = Morph(self.reshape_fn,
                                    self.reshape_infer_shape,
                                    self.reshape_infer_dtype
                                    )

    def reshape_fn(self, x, shp):
        """Reshape function."""
        return P.Reshape()(x, shp)

    def reshape_infer_shape(self, *args):
        *prefix, dim = args[0]
        t = prefix + [dim // self.block, self.block]
        return t

    def reshape_infer_dtype(self, *args):
        return args[0]

    def construct(self, tensor, shp):
        return self.local_reshape(tensor, shp)


class TensorReshapeTo3D(nn.Cell):
    """
    Reshape tensor to 3D with specified middle and last dimensions.
    
    This operation reshapes a tensor to a 3-dimensional tensor where the first dimension
    is automatically calculated from the total size, and the last two dimensions are fixed.
    
    Args:
        dim1: The second dimension (middle dimension) of the output 3D tensor.
        dim2: The third dimension (last dimension) of the output 3D tensor.
    """

    def __init__(
            self,
            dim1,
            dim2,
    ):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.local_reshape = Morph(self.reshape_fn,
                                    self.reshape_infer_shape,
                                    self.reshape_infer_dtype
                                    )

    def reshape_fn(self, x, shp):
        """Reshape function."""
        return P.Reshape()(x, shp)

    def reshape_infer_shape(self, *args):
        tensor_shape = args[0]
        total = math.prod(tensor_shape)
        t = [total // (self.dim1 * self.dim2), self.dim1, self.dim2]
        return t

    def reshape_infer_dtype(self, *args):
        return args[0]

    def construct(self, tensor, shp):
        return self.local_reshape(tensor, shp)


class PrefixDimensionReshape(nn.Cell):
    """
    Reshape tensor with fixed prefix dimensions and calculated last dimension.
    
    This operation reshapes a tensor by specifying the leading (prefix) dimensions,
    while the last dimension is automatically calculated from the total size.
    
    Args:
        *prefix: Variable number of prefix dimensions for the output tensor shape.
    """

    def __init__(
            self,
            *prefix
    ):
        self.prefix = list(prefix)
        super().__init__()
        self.local_reshape = Morph(self.reshape_fn,
                                    self.reshape_infer_shape,
                                    self.reshape_infer_dtype
                                    )

    def reshape_fn(self, x, shp):
        """Reshape function."""
        return P.Reshape()(x, shp)

    def reshape_infer_shape(self, *args):
        tensor_shape = args[0]
        total = math.prod(tensor_shape)
        prefix_total = math.prod(self.prefix)
        t = self.prefix + [total // prefix_total]
        return t

    def reshape_infer_dtype(self, *args):
        return args[0]

    def construct(self, tensor, shp):
        return self.local_reshape(tensor, shp)


class TensorReshapeTo2D(nn.Cell):
    """
    Reshape tensor to 2D with specified last dimension.
    
    This operation flattens a tensor to a 2-dimensional tensor where the last dimension
    is fixed and the first dimension is automatically calculated from the total size.
    
    Args:
        dim: The second dimension (last dimension) of the output 2D tensor.
    """

    def __init__(
            self,
            dim
    ):
        self.dim = dim
        super().__init__()
        self.local_reshape = Morph(self.reshape_fn,
                                    self.reshape_infer_shape,
                                    self.reshape_infer_dtype
                                    )

    def reshape_fn(self, x, shp):
        """Reshape function."""
        return P.Reshape()(x, shp)

    def reshape_infer_shape(self, *args):
        tensor_shape = args[0]
        total = math.prod(tensor_shape)
        t = [total // self.dim, self.dim]
        return t

    def reshape_infer_dtype(self, *args):
        return args[0]

    def construct(self, tensor, shp):
        return self.local_reshape(tensor, shp)


def muon_split(tensor, part_a: int, part_b: int, num_blocks: int):
    """
    Split a 2D tensor into two periodic parts along its first dimension.
    The split pattern repeats every (part_a + part_b) elements.

    Args:
        tensor: Input tensor of shape (M, N).
        part_a: Number of elements in the first part of each block.
        part_b: Number of elements in the second part of each block.
        num_blocks: Total number of (part_a + part_b) blocks.

    Returns:
        A tuple of two tensors (first_part, second_part),
        where:
          - first_part contains all part_a segments of each block.
          - second_part contains all part_b segments of each block.
    """
    tensor = tensor.T
    *prefix, _ = tensor.shape
    block = part_a + part_b
    t = BlockSplitReshape(block)(tensor, (*prefix, -1, block))

    first_part = PrefixDimensionReshape(*prefix)(t[..., :part_a], (*prefix, -1)).T
    second_part = PrefixDimensionReshape(*prefix)(t[..., part_a:], (*prefix, -1)).T
    return first_part, second_part


def muon_merge(tensor_a, tensor_b, part_a: int, part_b: int, num_blocks: int):
    """
    Merge two tensors back into the original periodic layout
    that was split by muon_split().

    Args:
        tensor_a: Tensor containing the first part of each block.
        tensor_b: Tensor containing the second part of each block.
        part_a: Number of elements in the first part of each block.
        part_b: Number of elements in the second part of each block.
        num_blocks: Total number of (part_a + part_b) blocks.

    Returns:
        A single tensor of the same shape as before muon_split().
    """
    tensor_a = tensor_a.T
    tensor_b = tensor_b.T
    *prefix, _ = tensor_a.shape

    a = BlockSplitReshape(part_a)(tensor_a, (*prefix, -1, part_a))
    b = BlockSplitReshape(part_b)(tensor_b, (*prefix, -1, part_b))
    t = P.Concat(axis=-1)([a, b])
    out = PrefixDimensionReshape(*prefix)(t, (*prefix, -1)).T
    return out


def _eval_tuple(spec, name, tensor):
    return spec(name, tensor) if callable(spec) else spec


def make_muon_fns(schema):
    """
    Generate two generic functions:
      - split_one(param_name, tensor) -> List[tensor]
      - merge_one(param_name, parts_list) -> tensor

    Dimensions in schema should be either numbers or callback functions:
      - periodic:   rule["parts"] = (a, b, num_blocks) or lambda(name, tensor)->(a,b,blocks)
      - reshape_* : rule["reshape"] = (x, y, z) or lambda(name, tensor)->(x,y,z)
    """

    def split_fn(param_name, tensor):
        """
        Input a 2D tensor, split it according to schema rules, and return several segments (List[tensor]).
        """

        for rule in schema:
            if not any(fnmatch(param_name, pat) for pat in rule["patterns"]):
                continue

            kind = rule["kind"]

            if kind == "periodic":
                part_a, part_b, num_blocks = _eval_tuple(rule["parts"], param_name, tensor)
                first_part, second_part = muon_split(tensor, part_a, part_b, num_blocks)
                return [first_part, second_part]

            if kind == "reshape_concat":
                # e.g. experts.weight1: first reshape to [E, H, 2I], then split into two halves along the last dimension
                _, hidden_size, total_intermediate = _eval_tuple(rule["reshape"], param_name, tensor)
                half_intermediate = total_intermediate // 2
                t3 = TensorReshapeTo3D(hidden_size, total_intermediate)(tensor, (-1, hidden_size, total_intermediate))
                return [t3[..., :half_intermediate], t3[..., half_intermediate:]]

            if kind == "reshape_only":
                # e.g. experts.weight2: just reshape to [E, I, H], no split
                _, intermediate_size, hidden_size = _eval_tuple(rule["reshape"], param_name, tensor)
                return [TensorReshapeTo3D(intermediate_size, hidden_size)(tensor, (-1, intermediate_size, hidden_size))]

            if kind == "alt_pair_periodic":
                # Alternating rows 1,1 (blocks = M//2)
                num_blocks = tensor.shape[0] // 2
                a, b = muon_split(tensor, 1, 1, num_blocks)
                return [a, b]

        # Default: no processing, return as whole block
        return [tensor]

    def merge_fn(param_name, parts_list):
        """
        Merge the output of split_one (List[tensor]) back to 2D according to the same rules.
        """
        concat = P.Concat(axis=-1)

        for rule in schema:
            if not any(fnmatch(param_name, pat) for pat in rule["patterns"]):
                continue

            kind = rule["kind"]

            if kind == "periodic":
                part_a, part_b, num_blocks = _eval_tuple(rule["parts"], param_name, parts_list[0])
                # Convention: periodic always has two segments
                return muon_merge(parts_list[0], parts_list[1], part_a, part_b, num_blocks)

            if kind == "reshape_concat":
                _, hidden_size, total_intermediate = _eval_tuple(rule["reshape"], param_name, parts_list[0])
                cat = concat([parts_list[0], parts_list[1]])  # [..., I] + [..., I] -> [..., 2I]
                return TensorReshapeTo2D(total_intermediate)(cat, (-1, total_intermediate))

            if kind == "reshape_only":
                _, _, hidden_size = _eval_tuple(rule["reshape"], param_name, parts_list[0])
                # Only one segment, directly restore to 2D
                return TensorReshapeTo2D(hidden_size)(parts_list[0], (-1, hidden_size))

            if kind == "alt_pair_periodic":
                num_blocks = parts_list[0].shape[0]  # 1 row per block
                return muon_merge(parts_list[0], parts_list[1], 1, 1, num_blocks)

        # Default: directly take the first segment
        return parts_list[0]

    return split_fn, merge_fn
