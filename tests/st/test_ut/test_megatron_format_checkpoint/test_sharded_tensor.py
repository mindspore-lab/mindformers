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
"""Test sharded tensor."""

import pytest
import mindspore as ms
from mindspore import nn
from mindspore.parallel import Layout
from mindspore.common.initializer import initializer, Normal

from mindformers.checkpoint.sharded_tensor import (
    ShardedTensor,
    build_sharded_tensor,
    is_main_replica,
    get_sharded_tensor_list_from_cell,
    convert_sharded_tensor_list_to_dict,
    get_value_type_from_layout,
    get_param_name_from_layout,
    get_strategy_info_from_sharded_tensor,
    _rank_id_with_slice_id,
    _alias_name_with_rank_id,
    _flatten_tensor_map,
    _tensor_map_with_rank_id
)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sharded_tensor_creation():
    """
    Feature: ShardedTensor creation
    Description: Create a ShardedTensor instance with all required parameters
    Expectation: ShardedTensor is created successfully with correct attributes
    """
    st = ShardedTensor(
        key="test.weight",
        org_key="original.test.weight",
        dtype=ms.float32,
        local_shape=(10,),
        global_shape=(100,),
        global_offset=(0,),
        axis_fragmentations=(10,)
    )

    assert st.key == "test.weight"
    assert st.org_key == "original.test.weight"
    assert st.dtype == ms.float32
    assert st.local_shape == (10,)
    assert st.global_shape == (100,)
    assert st.global_offset == (0,)
    assert st.axis_fragmentations == (10,)
    assert st.replica_id == 0
    assert st.allow_shape_mismatch is False
    assert st.allow_to_save is True


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_build_sharded_tensor():
    """
    Feature: build_sharded_tensor function
    Description: Call build_sharded_tensor helper function to create ShardedTensor
    Expectation: ShardedTensor is created successfully with correct attributes
    """
    st = build_sharded_tensor(
        param_name="layer.weight",
        param_dtype=ms.float16,
        local_shape=[5],
        global_shape=[50],
        axis_fragmentations=[10],
        global_offset=[0]
    )

    assert isinstance(st, ShardedTensor)
    assert st.key == "layer.weight"
    assert st.dtype == ms.float16
    assert st.local_shape == (5,)
    assert st.global_shape == (50,)
    assert st.axis_fragmentations == (10,)
    assert st.global_offset == (0,)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_is_main_replica_zero():
    """
    Feature: is_main_replica function
    Description: Check if integer replica_id 0 is considered main replica
    Expectation: Returns True for replica_id 0
    """
    result = is_main_replica(0)
    assert result is True


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_is_main_replica_tuple_all_zeros():
    """
    Feature: is_main_replica function
    Description: Check if tuple of all zeros is considered main replica
    Expectation: Returns True for tuple with all zero elements
    """
    result = is_main_replica((0, 0))
    assert result is True


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_is_main_replica_nonzero_integer():
    """
    Feature: is_main_replica function
    Description: Check if nonzero integer is considered main replica
    Expectation: Returns False for nonzero integer replica_id
    """
    result = is_main_replica(1)
    assert result is False


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_is_main_replica_mixed_tuple():
    """
    Feature: is_main_replica function
    Description: Check if tuple with mixed values is considered main replica
    Expectation: Returns False for tuple containing non-zero elements
    """
    result = is_main_replica((0, 1))
    assert result is False


class SimpleNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.dense = nn.Dense(10, 5)

    def construct(self, x):
        return self.dense(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_sharded_tensor_list_from_cell():
    """
    Feature: get_sharded_tensor_list_from_cell function
    Description: Extract sharded tensors from a neural network cell
    Expectation: Returns list of ShardedTensor objects for cell parameters
    """
    net = SimpleNet()

    # Initialize parameters
    net.dense.weight.set_data(initializer(Normal(), net.dense.weight.shape, net.dense.weight.dtype))
    net.dense.bias.set_data(initializer('zeros', net.dense.bias.shape, net.dense.bias.dtype))

    sharded_tensors = get_sharded_tensor_list_from_cell(net)

    assert len(sharded_tensors) >= 2  # Weight and bias

    weight_tensor = next(t for t in sharded_tensors if 'weight' in t.key)
    bias_tensor = next(t for t in sharded_tensors if 'bias' in t.key)

    assert weight_tensor.local_shape == net.dense.weight.shape
    assert bias_tensor.local_shape == net.dense.bias.shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_convert_sharded_tensor_list_to_dict():
    """
    Feature: convert_sharded_tensor_list_to_dict function
    Description: Convert list of ShardedTensor objects to dictionary
    Expectation: Returns dictionary mapping tensor keys to ShardedTensor objects
    """
    tensors = [
        ShardedTensor(key=f"param_{i}", org_key="", dtype=ms.float32,
                      local_shape=(10,), global_shape=(100,),
                      global_offset=(i * 10,), axis_fragmentations=(10,))
        for i in range(3)
    ]

    tensor_dict = convert_sharded_tensor_list_to_dict(tensors)

    assert len(tensor_dict) == 3
    for i in range(3):
        assert f"param_{i}" in tensor_dict
        assert tensor_dict[f"param_{i}"].key == f"param_{i}"


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sharded_tensor_with_custom_attributes():
    """
    Feature: ShardedTensor creation with custom attributes
    Description: Create a ShardedTensor instance with custom replica_id, allow_shape_mismatch and allow_to_save
    Expectation: ShardedTensor is created successfully with custom attributes
    """
    st = ShardedTensor(
        key="test.weight",
        org_key="original.test.weight",
        dtype=ms.float32,
        local_shape=(10,),
        global_shape=(100,),
        global_offset=(0,),
        axis_fragmentations=(10,),
        replica_id=(1, 2),
        allow_shape_mismatch=True,
        allow_to_save=False
    )

    assert st.replica_id == (1, 2)
    assert st.allow_shape_mismatch is True
    assert st.allow_to_save is False


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_build_sharded_tensor_with_layout():
    """
    Feature: build_sharded_tensor function with layout
    Description: Call build_sharded_tensor helper function with layout parameter
    Expectation: ShardedTensor is created successfully with layout
    """
    layout = Layout(device_matrix=(2, 2), alias_name=("dp", "mp"))

    st = build_sharded_tensor(
        param_name="layer.weight",
        param_dtype=ms.float16,
        local_shape=[5],
        global_shape=[50],
        axis_fragmentations=[10],
        global_offset=[0],
        layout=layout
    )

    assert isinstance(st, ShardedTensor)
    assert st.layout == layout


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_is_main_replica_single_element_tuple():
    """
    Feature: is_main_replica function
    Description: Check if single element tuple with zero is considered main replica
    Expectation: Returns True for single element tuple with zero
    """
    result = is_main_replica((0,))
    assert result is True


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_is_main_replica_single_nonzero_element_tuple():
    """
    Feature: is_main_replica function
    Description: Check if single element tuple with nonzero value is considered main replica
    Expectation: Returns False for single element tuple with nonzero value
    """
    result = is_main_replica((1,))
    assert result is False


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_sharded_tensor_list_from_cell_with_optimizer():
    """
    Feature: get_sharded_tensor_list_from_cell function with optimizer
    Description: Extract sharded tensors from a neural network cell and optimizer
    Expectation: Returns list of ShardedTensor objects for both cell and optimizer parameters
    """
    net = SimpleNet()

    # Initialize parameters
    net.dense.weight.set_data(initializer(Normal(), net.dense.weight.shape, net.dense.weight.dtype))
    net.dense.bias.set_data(initializer('zeros', net.dense.bias.shape, net.dense.bias.dtype))

    # Create optimizer
    optim = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)

    sharded_tensors = get_sharded_tensor_list_from_cell(net, optim)

    # Should have weight, bias from net and optimizer states (momentum, etc.)
    assert len(sharded_tensors) >= 2


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_convert_empty_sharded_tensor_list_to_dict():
    """
    Feature: convert_sharded_tensor_list_to_dict function
    Description: Convert empty list of ShardedTensor objects to dictionary
    Expectation: Returns empty dictionary
    """
    tensor_dict = convert_sharded_tensor_list_to_dict([])
    assert len(tensor_dict) == 0


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_convert_sharded_tensor_list_to_dict_duplicate_keys():
    """
    Feature: convert_sharded_tensor_list_to_dict function
    Description: Convert list of ShardedTensor objects with duplicate keys to dictionary
    Expectation: Later tensor overwrites earlier one with same key
    """
    tensors = [
        ShardedTensor(key="param", org_key="", dtype=ms.float32,
                      local_shape=(10,), global_shape=(100,),
                      global_offset=(0,), axis_fragmentations=(10,)),
        ShardedTensor(key="param", org_key="", dtype=ms.float32,
                      local_shape=(5,), global_shape=(50,),
                      global_offset=(10,), axis_fragmentations=(5,))
    ]

    tensor_dict = convert_sharded_tensor_list_to_dict(tensors)

    assert len(tensor_dict) == 1
    assert tensor_dict["param"].local_shape == (5,)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_strategy_info_from_sharded_tensor():
    """
    Feature: get_strategy_info_from_sharded_tensor function
    Description: Extract strategy information from a ShardedTensor object
    Expectation: Returns global_shape, axis_fragmentations, and global_offset as a tuple
    """
    st = ShardedTensor(
        key="test.weight",
        org_key="original.test.weight",
        dtype=ms.float32,
        local_shape=(10,),
        global_shape=(100,),
        global_offset=(5,),
        axis_fragmentations=(10,)
    )

    global_shape, axis_fragmentations, global_offset = get_strategy_info_from_sharded_tensor(st)

    assert global_shape == (100,)
    assert axis_fragmentations == (10,)
    assert global_offset == (5,)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_flatten_tensor_map():
    """
    Feature: _flatten_tensor_map function
    Description: Flatten nested tensor map structure
    Expectation: Returns flattened list of tensor map elements
    """
    # Test with nested structure
    tensor_map = [[1, 2], [3, [4, 5]], 6]
    flattened = _flatten_tensor_map(tensor_map)
    assert flattened == [1, 2, 3, 4, 5, 6]

    # Test with simple list
    tensor_map = [1, 2, 3]
    flattened = _flatten_tensor_map(tensor_map)
    assert flattened == [1, 2, 3]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_alias_name_with_rank_id():
    """
    Feature: _alias_name_with_rank_id function
    Description: Generate alias name to rank list mapping
    Expectation: Returns dictionary with alias names mapped to device numbers and rank tables
    """
    dev_matrix = [2, 2]
    alias_name = ["dp", "mp"]
    rank_list = [0, 1, 2, 3]

    result = _alias_name_with_rank_id(dev_matrix, alias_name, rank_list)

    assert "dp" in result
    assert "mp" in result
    assert len(result["dp"]) == 2
    assert len(result["mp"]) == 2


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tensor_map_with_rank_id():
    """
    Feature: _tensor_map_with_rank_id function
    Description: Map tensor dimensions to rank IDs
    Expectation: Returns list with rank tables and strides for each tensor dimension
    """
    dev_matrix = [2, 2]
    alias_name = ["dp", "mp"]
    rank_list = [0, 1, 2, 3]
    tensor_map = [0, 1]

    dev_arrange = _alias_name_with_rank_id(dev_matrix, alias_name, rank_list)

    flat_tensor_map = _flatten_tensor_map(tensor_map)
    result = _tensor_map_with_rank_id(dev_matrix, flat_tensor_map, alias_name, dev_arrange)

    assert len(result) == len(flat_tensor_map)
    # Each element should be a list with rank table and stride
    for elem in result:
        if elem is not None:
            assert len(elem) == 2


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_rank_id_with_slice_id():
    """
    Feature: _rank_id_with_slice_id function
    Description: Convert alias rank stride information to rank ID vs slice ID mapping
    Expectation: Returns rank slice table and global offset tuple
    """
    # Mock alias_rank_stride data
    alias_rank_stride = [
        [[0, 1, 0, 1], 2],  # rank_table, stride
        [[0, 0, 1, 1], 1]
    ]

    rank_slice_table, global_offset = _rank_id_with_slice_id(alias_rank_stride)

    assert isinstance(rank_slice_table, list)
    assert isinstance(global_offset, tuple)
    assert len(rank_slice_table) == 4  # 4 ranks
    assert len(global_offset) == 4


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_param_name_from_layout():
    """
    Feature: get_param_name_from_layout function
    Description: Extract parameter names from layout information
    Expectation: Returns list of parameter names
    """
    param_infos = [
        {
            "weight": (None, None, None)
        },
        {
            "bias": (None, None, None)
        }
    ]

    names = get_param_name_from_layout(param_infos)
    assert names == ["weight", "bias"]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_value_type_from_layout():
    """
    Feature: get_value_type_from_layout function
    Description: Extract parameter types from layout information
    Expectation: Returns list of parameter types
    """
    param_infos = [
        {
            "weight": (None, ms.float32, None)
        },
        {
            "bias": (None, ms.float16, None)
        }
    ]

    types = get_value_type_from_layout(param_infos)
    assert types == [ms.float32, ms.float16]
