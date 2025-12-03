#  Copyright 2024 Huawei Technologies Co., Ltd
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ============================================================================
"""Test for fully_parallel.py"""
# pylint: disable=W0621, W0212, W0613
import os
from unittest.mock import patch, MagicMock

import pytest
from mindspore import nn

from mindformers.checkpoint.utils import FileType
from mindformers.checkpoint.fully_parallel import (
    BalancedSaveStrategy,
    distribute_shards,
    apply_balance_shard_strategy
)


class MockShardTensor:
    """Mock ShardTensor class for testing"""

    def __init__(self, key, global_offset, local_shape, dtype, size=100):
        self.key = key
        self.global_offset = global_offset
        self.local_shape = local_shape
        self.dtype = dtype
        self.size = size


@pytest.fixture
def mock_network():
    """Create a mock network for testing"""
    network = MagicMock(spec=nn.Cell)
    return network


@pytest.fixture
def mock_get_all_sharded_tensor():
    """Mock get_all_sharded_tensor function"""
    mock_shard_tensor1 = MockShardTensor("param1", (0,), (10,), "float32")
    mock_shard_tensor2 = MockShardTensor("param2", (10,), (10,), "float32")
    mock_shard_tensor3 = MockShardTensor("param3", (0,), (10,), "float32")

    with patch("mindformers.checkpoint.fully_parallel.get_all_sharded_tensor") as mock:
        mock.return_value = [
            [mock_shard_tensor1, mock_shard_tensor2],
            [mock_shard_tensor3]
        ]
        yield mock


@pytest.fixture
def mock_get_rank():
    """Mock get_rank function"""
    with patch("mindformers.checkpoint.fully_parallel.get_rank") as mock:
        mock.return_value = 0
        yield mock


@pytest.fixture
def mock_get_real_rank():
    """Mock get_real_rank function"""
    with patch("mindformers.checkpoint.fully_parallel.get_real_rank") as mock:
        mock.return_value = 0
        yield mock


@pytest.fixture
def mock_save_checkpoint():
    """Mock save_checkpoint function"""
    with patch("mindformers.checkpoint.fully_parallel.save_checkpoint") as mock:
        yield mock


@pytest.fixture
def mock_get_metadata_filename():
    """Mock get_metadata_filename function"""
    with patch("mindformers.checkpoint.fully_parallel.get_metadata_filename") as mock:
        mock.return_value = "metadata.json"
        yield mock


@pytest.fixture
def mock_get_checkpoint_name():
    """Mock get_checkpoint_name function"""
    with patch("mindformers.checkpoint.fully_parallel.get_checkpoint_name") as mock:
        mock.return_value = "checkpoint_0-2"
        yield mock


@pytest.fixture
def mock_get_checkpoint_iter_dir():
    """Mock get_checkpoint_iter_dir function"""
    with patch("mindformers.checkpoint.fully_parallel.get_checkpoint_iter_dir") as mock:
        mock.return_value = "./checkpoint_iter_0"
        yield mock


@pytest.fixture
def mock_save_metadata():
    """Mock save_metadata function"""
    with patch("mindformers.checkpoint.fully_parallel.save_metadata") as mock:
        yield mock


@pytest.fixture
def mock_load_metadata():
    """Mock load_metadata function"""
    with patch("mindformers.checkpoint.fully_parallel.load_metadata") as mock:
        mock.return_value = ({}, {})
        yield mock


@pytest.fixture
def mock_reverse_sharded_tensor_shard_id():
    """Mock _reverse_sharded_tensor_shard_id function"""
    with patch("mindformers.checkpoint.fully_parallel._reverse_sharded_tensor_shard_id") as mock:
        mock.return_value = "param1"
        yield mock


@pytest.fixture
def mock_sharded_tensor_shard_id():
    """Mock sharded_tensor_shard_id function"""
    with patch("mindformers.checkpoint.fully_parallel.sharded_tensor_shard_id") as mock:
        mock.side_effect = lambda key, offset: f"{key}_{offset}"
        yield mock


@pytest.fixture
def mock_get_shard_size():
    """Mock _get_shard_size function"""
    with patch("mindformers.checkpoint.fully_parallel._get_shard_size") as mock:
        mock.return_value = 100
        yield mock


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_distribute_shards_basic():
    """
    Feature: distribute_shards function basic functionality
    Description: Test distribute_shards function with basic input data, including different shard coverage and sizes
    Expectation: All shards are assigned to valid ranks, and each shard is assigned to a rank that covers it
    """
    shard_coverage = {
        "shard1": [0, 1],
        "shard2": [0],
        "shard3": [1]
    }
    shard_sizes = {
        "shard1": 100,
        "shard2": 200,
        "shard3": 150
    }
    total_ranks = 2

    result = distribute_shards(shard_coverage, shard_sizes, total_ranks)

    # Check that all shards are assigned
    assert len(result) == 3
    # Check that each shard is assigned to a valid rank
    for rank in result.values():
        assert 0 <= rank < total_ranks
    # Check that shards are assigned to ranks that cover them
    for shard_id, rank in result.items():
        assert rank in shard_coverage[shard_id]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_distribute_shards_empty():
    """
    Feature: distribute_shards function with empty input
    Description: Test distribute_shards function when shard_coverage and shard_sizes are empty
    Expectation: Return an empty dictionary
    """
    shard_coverage = {}
    shard_sizes = {}
    total_ranks = 2

    result = distribute_shards(shard_coverage, shard_sizes, total_ranks)

    assert result == {}


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_distribute_shards_single_rank():
    """
    Feature: distribute_shards function with single rank
    Description: Test distribute_shards function when there is only one rank available
    Expectation: All shards are assigned to the single rank
    """
    shard_coverage = {
        "shard1": [0],
        "shard2": [0]
    }
    shard_sizes = {
        "shard1": 100,
        "shard2": 200
    }
    total_ranks = 1

    result = distribute_shards(shard_coverage, shard_sizes, total_ranks)

    assert result == {"shard1": 0, "shard2": 0}


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_apply_balance_shard_strategy(
        mock_network, mock_get_all_sharded_tensor, mock_get_real_rank,
        mock_sharded_tensor_shard_id, mock_get_shard_size
):
    """
    Feature: apply_balance_shard_strategy function
    Description: Test apply_balance_shard_strategy function with mock network and related fixtures
    Expectation: Return three dictionaries: shard_to_saving_rank, shard_id_to_tensor, and dst_sharded_tensor_metas
    """
    result = apply_balance_shard_strategy(mock_network, None)

    assert len(result) == 4
    shard_to_saving_rank, shard_id_to_tensor, dst_sharded_tensor_metas, param_redundancy = result

    assert isinstance(shard_to_saving_rank, dict)
    assert isinstance(shard_id_to_tensor, dict)
    assert isinstance(dst_sharded_tensor_metas, dict)
    assert isinstance(param_redundancy, dict)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_balanced_save_strategy_init(mock_network, mock_get_rank):
    """
    Feature: BalancedSaveStrategy initialization
    Description: Test BalancedSaveStrategy class initialization with various parameters
    Expectation: All attributes are correctly set according to the input parameters
    """

    strategy = BalancedSaveStrategy(
        network=mock_network,
        user_prefix="test",
        do_cache_distribution=True,
        checkpoint_path="./checkpoint"
    )

    assert strategy.network == mock_network
    assert strategy.user_prefix == "test"
    assert strategy.do_cache_distribution is True
    assert strategy.cached_distribution is None
    assert strategy.checkpoint_path == "./checkpoint"
    assert strategy.file_type == FileType.MODEL


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_balanced_save_strategy_apply_saving_parallelization(
        mock_network, mock_get_rank, mock_get_all_sharded_tensor,
        mock_sharded_tensor_shard_id, mock_get_shard_size
):
    """
    Feature: BalancedSaveStrategy.apply_saving_parallelization method
    Description: Test apply_saving_parallelization method without cache
    Expectation: Return a tuple of two dictionaries: shared_distribution and id_to_tensor
    """
    strategy = BalancedSaveStrategy(
        network=mock_network,
        checkpoint_path="./checkpoint"
    )

    result = strategy.apply_saving_parallelization()

    assert len(result) == 2
    shard_id_to_ranks, shard_id_to_tensor = result
    assert isinstance(shard_id_to_ranks, dict)
    assert isinstance(shard_id_to_tensor, dict)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_balanced_save_strategy_apply_saving_parallelization_with_cache(
        mock_network, mock_get_rank, mock_get_all_sharded_tensor,
        mock_sharded_tensor_shard_id, mock_get_shard_size
):
    """
    Feature: BalancedSaveStrategy.apply_saving_parallelization method with cache
    Description: Test apply_saving_parallelization method with cache enabled
    Expectation: First call computes distribution, second call uses cached distribution without recomputing
    """
    strategy = BalancedSaveStrategy(
        network=mock_network,
        do_cache_distribution=True,
        checkpoint_path="./checkpoint"
    )

    # First call - should compute distribution
    result1 = strategy.apply_saving_parallelization()

    # Second call - should use cached distribution
    with patch("mindformers.checkpoint.fully_parallel.apply_balance_shard_strategy") as mock_apply:
        mock_apply.return_value = ({}, {})
        result2 = strategy.apply_saving_parallelization()
        # Check that apply_balance_shard_strategy was not called again
        mock_apply.assert_not_called()

    assert result1 == result2


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_balanced_save_strategy_get_total_files(
        mock_network, mock_get_rank, mock_get_all_sharded_tensor,
        mock_sharded_tensor_shard_id, mock_get_shard_size
):
    """
    Feature: BalancedSaveStrategy.get_total_files method
    Description: Test get_total_files method to get the total number of checkpoint files
    Expectation: Return a non-negative integer representing the total number of files
    """
    strategy = BalancedSaveStrategy(
        network=mock_network,
        checkpoint_path="./checkpoint"
    )

    total_files = strategy.get_total_files()

    assert isinstance(total_files, int)
    assert total_files >= 0


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_balanced_save_strategy_get_cur_rank_file_id(
        mock_network, mock_get_rank, mock_get_all_sharded_tensor,
        mock_sharded_tensor_shard_id, mock_get_shard_size
):
    """
    Feature: BalancedSaveStrategy.get_cur_rank_file_id method
    Description: Test get_cur_rank_file_id method to get the current rank's file ID
    Expectation: Return a non-negative integer representing the current rank's file ID
    """
    strategy = BalancedSaveStrategy(
        network=mock_network,
        checkpoint_path="./checkpoint"
    )

    cur_rank_file_id = strategy.get_cur_rank_file_id()

    assert isinstance(cur_rank_file_id, int)
    assert cur_rank_file_id >= 0


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_balanced_save_strategy_save(
        tmp_path, mock_network, mock_get_rank, mock_get_all_sharded_tensor,
        mock_sharded_tensor_shard_id, mock_get_shard_size, mock_save_checkpoint,
        mock_get_metadata_filename, mock_get_checkpoint_name, mock_get_checkpoint_iter_dir,
        mock_save_metadata, mock_load_metadata, mock_reverse_sharded_tensor_shard_id
):
    """
    Feature: BalancedSaveStrategy.save method
    Description: Test save method to save model checkpoint without existing metadata
    Expectation: save_checkpoint is called, get_checkpoint_iter_dir is called, get_checkpoint_name is called
    """
    checkpoint_path = str(tmp_path / "checkpoint")
    os.makedirs(checkpoint_path, exist_ok=True)

    strategy = BalancedSaveStrategy(
        network=mock_network,
        checkpoint_path=checkpoint_path
    )

    with patch("mindformers.checkpoint.fully_parallel.os.path.exists", return_value=False):
        strategy.save(0)

    # Check that save_checkpoint was called
    mock_save_checkpoint.assert_called_once()
    # Check that get_checkpoint_iter_dir was called
    mock_get_checkpoint_iter_dir.assert_called_once_with(checkpoint_path, 0)
    # Check that get_checkpoint_name was called
    mock_get_checkpoint_name.assert_called()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_balanced_save_strategy_save_with_existing_metadata(
        tmp_path, mock_network, mock_get_rank, mock_get_all_sharded_tensor,
        mock_sharded_tensor_shard_id, mock_get_shard_size, mock_save_checkpoint,
        mock_get_metadata_filename, mock_get_checkpoint_name, mock_get_checkpoint_iter_dir,
        mock_save_metadata, mock_reverse_sharded_tensor_shard_id
):
    """
    Feature: BalancedSaveStrategy.save method with existing metadata
    Description: Test save method to save model checkpoint with existing metadata file
    Expectation: save_checkpoint is called, load_metadata is called
    """
    checkpoint_path = str(tmp_path / "checkpoint")
    os.makedirs(checkpoint_path, exist_ok=True)

    # Create a mock metadata file
    metadata_file = os.path.join(checkpoint_path, "metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        f.write("{}")

    strategy = BalancedSaveStrategy(
        network=mock_network,
        checkpoint_path=checkpoint_path
    )

    with patch("mindformers.checkpoint.fully_parallel.os.path.exists", return_value=True):
        with patch("mindformers.checkpoint.fully_parallel.load_metadata") as mock_load:
            mock_load.return_value = ({"shard1": MagicMock()},
                                      {"param1": [{"file_name": "test.safetensors", "storage_rank": 0}]})
            strategy.save(0)

    # Check that save_checkpoint was called
    mock_save_checkpoint.assert_called_once()
    # Check that load_metadata was called
    mock_load.assert_called_once()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_balanced_save_strategy__get_rank_params_mappings(
        mock_network, mock_get_rank
):
    """
    Feature: BalancedSaveStrategy._get_rank_params_mappings method
    Description: Test _get_rank_params_mappings method to create mapping from rank IDs to parameter names
    Expectation: Return a dictionary mapping rank IDs to lists of parameter names
    """
    strategy = BalancedSaveStrategy(
        network=mock_network,
        checkpoint_path="./checkpoint"
    )

    # Create mock data
    shared_distribution = {
        "shard1": 0,
        "shard2": 1,
        "shard3": 0
    }

    mock_tensor1 = MagicMock()
    mock_tensor1.key = "param1"
    mock_tensor2 = MagicMock()
    mock_tensor2.key = "param2"
    mock_tensor3 = MagicMock()
    mock_tensor3.key = "param3"

    id_to_tensor = {
        "shard1": mock_tensor1,
        "shard2": mock_tensor2,
        "shard3": mock_tensor3
    }

    result = strategy._get_rank_params_mappings(shared_distribution, id_to_tensor)

    assert isinstance(result, dict)
    assert 0 in result
    assert 1 in result
    assert "param1" in result[0]
    assert "param3" in result[0]
    assert "param2" in result[1]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_balanced_save_strategy__get_rank_param_ids_mappings(
        mock_network, mock_get_rank
):
    """
    Feature: BalancedSaveStrategy._get_rank_param_ids_mappings method
    Description: Test _get_rank_param_ids_mappings method to create mapping from rank IDs to parameter IDs
    Expectation: Return a dictionary mapping rank IDs to lists of parameter IDs
    """
    strategy = BalancedSaveStrategy(
        network=mock_network,
        checkpoint_path="./checkpoint"
    )

    # Create mock data
    shared_distribution = {
        "shard1": 0,
        "shard2": 1,
        "shard3": 0
    }

    result = strategy._get_rank_param_ids_mappings(shared_distribution)

    assert isinstance(result, dict)
    assert 0 in result
    assert 1 in result
    assert "shard1" in result[0]
    assert "shard3" in result[0]
    assert "shard2" in result[1]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_balanced_save_strategy__get_total_files_num(
        mock_network, mock_get_rank
):
    """
    Feature: BalancedSaveStrategy._get_total_files_num method
    Description: Test _get_total_files_num method to calculate total number of files based on rank params mappings
    Expectation: Return the correct number of files based on the input mappings
    """
    strategy = BalancedSaveStrategy(
        network=mock_network,
        checkpoint_path="./checkpoint"
    )

    # Test with non-empty params
    rank_params_mappings = {
        0: ["param1", "param2"],
        1: ["param3"],
        2: []
    }

    result = strategy._get_total_files_num(rank_params_mappings)
    assert result == 2

    # Test with all empty params
    rank_params_mappings = {
        0: [],
        1: []
    }

    result = strategy._get_total_files_num(rank_params_mappings)
    assert result == 0


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_balanced_save_strategy__get_cur_rank_file_id(
        mock_network, mock_get_rank
):
    """
    Feature: BalancedSaveStrategy._get_cur_rank_file_id method
    Description: Test _get_cur_rank_file_id method to get the current rank's file ID based on rank params mappings
    Expectation: Return the correct file ID for the current rank based on the input mappings
    """
    strategy = BalancedSaveStrategy(
        network=mock_network,
        checkpoint_path="./checkpoint"
    )

    # Test when current rank has params
    rank_params_mappings = {
        0: [],
        1: ["param1"],
        2: ["param2"]
    }

    with patch.object(strategy, 'rank_id', 1):
        result = strategy._get_cur_rank_file_id(rank_params_mappings)
        assert result == 0

    # Test when current rank has no params
    rank_params_mappings = {
        0: ["param1"],
        1: [],
        2: ["param2"]
    }

    with patch.object(strategy, 'rank_id', 1):
        result = strategy._get_cur_rank_file_id(rank_params_mappings)
        assert result == 1

    # Test when current rank is not in mappings
    rank_params_mappings = {
        0: ["param1"],
        2: ["param2"]
    }

    with patch.object(strategy, 'rank_id', 1):
        result = strategy._get_cur_rank_file_id(rank_params_mappings)
        assert result is None


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_balanced_save_strategy_get_total_files_and_cur_rank_file_id(
        mock_network, mock_get_rank, mock_get_all_sharded_tensor,
        mock_sharded_tensor_shard_id, mock_get_shard_size
):
    """
    Feature: BalancedSaveStrategy.get_total_files and get_cur_rank_file_id methods with caching
    Description: Test that calling get_total_files and get_cur_rank_file_id caches the results
    Expectation: Second calls to these methods should use cached values without recomputing
    """
    strategy = BalancedSaveStrategy(
        network=mock_network,
        checkpoint_path="./checkpoint"
    )

    total_files = strategy.get_total_files()
    cur_rank_file_id = strategy.get_cur_rank_file_id()

    # Check that values are cached
    assert strategy.total_files_num == total_files
    assert strategy.cur_rank_file_id == cur_rank_file_id

    # Check that second calls use cached values
    with patch("mindformers.checkpoint.fully_parallel.apply_balance_shard_strategy") as mock_apply:
        mock_apply.return_value = ({}, {})
        total_files2 = strategy.get_total_files()
        cur_rank_file_id2 = strategy.get_cur_rank_file_id()

        # Mock should not be called since values are cached
        mock_apply.assert_not_called()

        assert total_files2 == total_files
        assert cur_rank_file_id2 == cur_rank_file_id
