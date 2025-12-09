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
"""Test cases for BlendedMegatronDatasetBuilder"""

import os
import subprocess
import time
import glob
from unittest.mock import patch
import pytest
import numpy as np

from mindformers.dataset.blended_datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
    _get_size_per_split_per_dataset
)
from mindformers.dataset.blended_datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from mindformers.dataset.blended_datasets.utils import Split
from mindformers.dataset.blended_datasets import utils as blended_utils_module
from mindformers.tools.logger import logger

try:
    from filelock import FileLock

    HAS_FILELOCK = True
except ImportError:
    FileLock = None
    HAS_FILELOCK = False


def _check_helpers_exists(helpers_dir):
    """Check if helpers.so exists and is valid."""
    so_pattern = os.path.join(helpers_dir, "helpers*.so")
    existing_so_files = glob.glob(so_pattern)
    return existing_so_files and any(os.path.getsize(f) > 1000 for f in existing_so_files)


def _compile_helpers_safe(helpers_dir, worker_id):
    """Compile helpers if not already compiled."""
    if _check_helpers_exists(helpers_dir):
        return

    logger.info(f"[{worker_id}] Starting compilation...")
    result = subprocess.run(["make", "-C", helpers_dir], capture_output=True, text=True, check=False)

    if result.returncode != 0 and not _check_helpers_exists(helpers_dir):
        raise RuntimeError(f"Failed to compile helpers: {result.stderr}")

    logger.info(f"[{worker_id}] Compilation completed")


@pytest.fixture(scope="session", autouse=True)
def ensure_helpers_compiled(request, tmp_path_factory):
    """Ensure helpers are compiled once with process-safe locking for pytest-xdist."""
    # Get worker_id if running with pytest-xdist, otherwise use 'master'
    worker_id = getattr(request.config, 'workerinput', {}).get('workerid', 'master')

    helpers_dir = os.path.abspath(os.path.dirname(blended_utils_module.__file__))

    # Quick check: if already compiled, all workers skip immediately
    if _check_helpers_exists(helpers_dir):
        logger.info(f"[{worker_id}] helpers.so already exists, using directly")
        yield
        return

    # Single process mode - compile directly
    if worker_id == "master":
        _compile_helpers_safe(helpers_dir, worker_id)
        yield
        return

    # Parallel mode - use file lock
    lock_file = tmp_path_factory.getbasetemp().parent / "helpers_compile.lock"

    if HAS_FILELOCK:
        with FileLock(str(lock_file), timeout=300):
            if not _check_helpers_exists(helpers_dir):
                _compile_helpers_safe(helpers_dir, worker_id)
    else:
        # Fallback: simple atomic lock without filelock library
        for _ in range(600):  # 5 min timeout (600 * 0.5s)
            try:
                fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    if not _check_helpers_exists(helpers_dir):
                        _compile_helpers_safe(helpers_dir, worker_id)
                finally:
                    os.close(fd)
                    os.unlink(str(lock_file))
                break
            except FileExistsError:
                time.sleep(0.5)
                if _check_helpers_exists(helpers_dir):
                    break
        else:
            raise TimeoutError("Timeout waiting for helpers compilation")

    yield

class DummyTokenizer:
    """A dummy tokenizer for testing purposes"""

    def __init__(self):
        self.pad = 0
        self.eod = 2
        self.unique_identifiers = {"class": "DummyTokenizer"}

    def encode(self, **_kwargs):
        return [1, 2, 3]

    def decode(self, **_kwargs):
        return "dummy text"


class DummyMegatronDataset:
    """A dummy MegatronDataset for testing purposes"""

    def __init__(self, low_level_dataset=None, dataset_path=None, indexed_indices=None,
                 num_samples=None, index_split=None, config=None, **_kwargs):
        """Initialize with the same signature as MegatronDataset

        Args:
            low_level_dataset: Low level dataset
            dataset_path: Dataset path
            indexed_indices: Indexed indices
            num_samples: Number of samples
            index_split: Index split
            config: Configuration
            **_kwargs: Additional keyword arguments
        """
        if num_samples is not None:
            self._data = list(range(num_samples))
        else:
            self._data = list(range(100))

        self.low_level_dataset = low_level_dataset
        self.dataset_path = dataset_path
        self.indexed_indices = indexed_indices
        self.num_samples = num_samples
        self.index_split = index_split
        self.config = config
        self.built_anew_on_cache_miss = False
        self.unique_identifiers = {"class": "DummyMegatronDataset"}

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    @staticmethod
    def numel_low_level_dataset(low_level_dataset, **_kwargs):
        if low_level_dataset is None:
            return 0
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(dataset_path=None, config=None, **kwargs):  # pylint: disable=unused-argument
        """Build a dummy low level dataset

        Args:
            dataset_path: Path to the dataset
            config: Configuration object
            **kwargs: Additional arguments

        Returns:
            list: A list representing the dataset
        """
        return list(range(100))


def is_built_on_rank_func():
    """Helper function to replace lambda
    Returns:
        bool: Always returns True for testing purposes
    """
    return True


def is_built_on_rank_func_false():
    """Helper function that returns False
    Returns:
        bool: Always returns False for testing purposes
    """
    return False


def create_test_config(**kwargs):
    """Create a BlendedMegatronDatasetConfig with default values and optional overrides

    Args:
        **kwargs: Optional config overrides

    Returns:
        BlendedMegatronDatasetConfig: Config instance
    """
    default_config = {
        "sequence_length": 32,
        "random_seed": 1234,
        "tokenizer": DummyTokenizer(),
        "path_to_cache": None,
        "mmap_bin_files": False
    }
    default_config.update(kwargs)
    return BlendedMegatronDatasetConfig(**default_config)


def create_test_builder(config_kwargs=None, builder_kwargs=None):
    """Create a BlendedMegatronDatasetBuilder with common setup

    Args:
        config_kwargs (dict, optional): Config overrides
        builder_kwargs (dict, optional): Builder overrides

    Returns:
        BlendedMegatronDatasetBuilder: Builder instance
    """
    if config_kwargs is None:
        config_kwargs = {}

    config = create_test_config(**config_kwargs)

    if builder_kwargs is None:
        builder_kwargs = {}

    default_builder_kwargs = {
        "cls": DummyMegatronDataset,
        "sizes": [10, 5, 15],
        "is_built_on_rank": is_built_on_rank_func,
        "config": config
    }
    default_builder_kwargs.update(builder_kwargs)

    return BlendedMegatronDatasetBuilder(**default_builder_kwargs)


class TestBlendedMegatronDatasetBuilder:
    """Test class for BlendedMegatronDatasetBuilder"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_builder_initialization(self):
        """
        Feature: BlendedMegatronDatasetBuilder initialization
        Description: Test BlendedMegatronDatasetBuilder can be initialized correctly
        Expectation: Builder initializes without error and has expected properties
        """
        config = create_test_config()
        sizes = [10, 5, 15]

        builder = BlendedMegatronDatasetBuilder(
            cls=DummyMegatronDataset,
            sizes=sizes,
            is_built_on_rank=is_built_on_rank_func,
            config=config
        )

        assert isinstance(builder, BlendedMegatronDatasetBuilder)
        assert builder.cls == DummyMegatronDataset
        assert builder.sizes == sizes
        assert builder.is_built_on_rank is is_built_on_rank_func
        assert builder.config == config

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_builder_initialization_assertion_error(self):
        """
        Feature: BlendedMegatronDatasetBuilder initialization assertion
        Description: Test BlendedMegatronDatasetBuilder raises AssertionError when size_is_none but weights_are_not_none
        Expectation: AssertionError is raised
        """
        config = create_test_config()
        config.mock = False
        config.blend = (["prefix1"], [0.5])
        sizes = [None, 5, 15]

        with pytest.raises(AssertionError, match="size_is_none => weights_are_none fails"):
            BlendedMegatronDatasetBuilder(
                cls=DummyMegatronDataset,
                sizes=sizes,
                is_built_on_rank=is_built_on_rank_func,
                config=config
            )

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_size_per_split_per_dataset(self):
        """
        Feature: _get_size_per_split_per_dataset utility function
        Description: Test _get_size_per_split_per_dataset function works correctly
        Expectation: Function returns expected sizes per dataset per split
        """
        normalized_weights = [0.3, 0.7]
        target_size_per_split = [100, 50, 200]

        result = _get_size_per_split_per_dataset(normalized_weights, target_size_per_split)

        assert len(result) == 2  # Two weights
        assert all(len(split_sizes) == 3 for split_sizes in result)  # Three splits

        expected_0 = [int(np.ceil(100 * 0.3 * 1.005)), int(np.ceil(50 * 0.3 * 1.005)), int(np.ceil(200 * 0.3 * 1.005))]
        expected_1 = [int(np.ceil(100 * 0.7 * 1.005)), int(np.ceil(50 * 0.7 * 1.005)), int(np.ceil(200 * 0.7 * 1.005))]

        assert result[0] == expected_0
        assert result[1] == expected_1

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_build_generic_dataset(self):
        """
        Feature: build_generic_dataset method
        Description: Test build_generic_dataset method works correctly
        Expectation: Method builds dataset correctly based on rank and conditions
        """
        builder = create_test_builder()

        low_level_dataset = list(range(100))
        dataset_path = None
        indexed_indices = np.arange(50)
        num_samples = 10
        index_split = Split.train

        with patch('mindformers.dataset.blended_datasets.blended_megatron_dataset_builder.get_real_group_size',
                   return_value=1):
            dataset = builder.build_generic_dataset(
                DummyMegatronDataset,
                is_built_on_rank_func,
                True,
                low_level_dataset,
                dataset_path,
                indexed_indices,
                num_samples,
                index_split,
                builder.config
            )

        assert isinstance(dataset, DummyMegatronDataset)
        assert len(dataset) == num_samples

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_build_generic_dataset_with_oserror(self):
        """
        Feature: build_generic_dataset method error handling
        Description: Test build_generic_dataset handles OSError correctly
        Expectation: Exception is raised with proper message
        """
        with patch('mindformers.dataset.blended_datasets.blended_megatron_dataset_builder.get_real_group_size',
                   return_value=2):
            with patch('mindformers.dataset.blended_datasets.blended_megatron_dataset_builder.get_real_rank',
                       return_value=0):
                def mock_cls_raises_oserror(*args, **kwargs):
                    raise OSError("Test OS Error")

                with pytest.raises(Exception, match="Failed to write dataset materials to the data cache directory"):
                    BlendedMegatronDatasetBuilder.build_generic_dataset(
                        mock_cls_raises_oserror,
                        is_built_on_rank_func,
                        True
                    )

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_build_generic_dataset_distributed_rank_nonzero(self):
        """
        Feature: build_generic_dataset method in distributed environment
        Description: Test build_generic_dataset works when rank is not zero
        Expectation: Dataset is built correctly on non-zero rank
        """
        with patch('mindformers.dataset.blended_datasets.blended_megatron_dataset_builder.get_real_group_size',
                   return_value=2):
            with patch('mindformers.dataset.blended_datasets.blended_megatron_dataset_builder.get_real_rank',
                       return_value=1):
                with patch(
                        'mindformers.dataset.blended_datasets.blended_megatron_dataset_builder.skip_barrier_controller'
                ):
                    dataset = BlendedMegatronDatasetBuilder.build_generic_dataset(
                        DummyMegatronDataset,
                        is_built_on_rank_func,
                        True,
                        list(range(10)),
                        None,
                        np.arange(5),
                        5,
                        Split.train,
                        create_test_config()
                    )

                    assert isinstance(dataset, DummyMegatronDataset)
                    assert len(dataset) == 5

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_build_generic_dataset_distributed_rank_zero_not_built(self):
        """
        Feature: build_generic_dataset method in distributed environment
        Description: Test build_generic_dataset when rank is zero but not built on rank
        Expectation: Returns None
        """
        with patch('mindformers.dataset.blended_datasets.blended_megatron_dataset_builder.get_real_group_size',
                   return_value=2):
            with patch('mindformers.dataset.blended_datasets.blended_megatron_dataset_builder.get_real_rank',
                       return_value=0):
                with patch(
                        'mindformers.dataset.blended_datasets.blended_megatron_dataset_builder.skip_barrier_controller'
                ):
                    dataset = BlendedMegatronDatasetBuilder.build_generic_dataset(
                        DummyMegatronDataset,
                        is_built_on_rank_func_false,
                        True,
                        list(range(10)),
                        None,
                        np.arange(5),
                        5,
                        Split.train,
                        create_test_config()
                    )

                    assert dataset is None

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_build_method_mock_dataset(self):
        """
        Feature: build method with mock dataset
        Description: Test build method works with mock configuration
        Expectation: Method builds mock datasets correctly
        """
        config = create_test_config()
        config.mock = True

        builder = create_test_builder(config_kwargs={}, builder_kwargs={"config": config})

        with patch('mindformers.dataset.blended_datasets.blended_megatron_dataset_builder.get_real_group_size',
                   return_value=1):
            with patch('mindformers.dataset.blended_datasets.blended_megatron_dataset_builder.get_real_rank',
                       return_value=0):
                config.split_matrix = [None, None, None]
                config.split_matrix[0] = (0.0, 1.0)

                datasets = builder.build()

        assert isinstance(datasets, list)
        assert len(datasets) == len(Split)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_build_method_mock_dataset_failure(self):
        """
        Feature: build method with mock dataset failure
        Description: Test build method handles mock dataset failure correctly
        Expectation: Proper exception is raised
        """
        config = create_test_config()
        config.mock = True

        class FailingDummyMegatronDataset(DummyMegatronDataset):
            """A dummy MegatronDataset that always fails during low level dataset building for testing purposes"""

            @staticmethod
            def build_low_level_dataset(dataset_path=None, config=None, **kwargs):  # pylint: disable=unused-argument
                """Build a failing dummy low level dataset

                Args:
                    dataset_path: Path to the dataset
                    config: Configuration object
                    **kwargs: Additional arguments

                Raises:
                    Exception: Always raised for testing purposes
                """
                raise Exception("Mock build failure")

        builder = create_test_builder(
            builder_kwargs={
                "cls": FailingDummyMegatronDataset,
                "config": config
            }
        )

        with patch('mindformers.dataset.blended_datasets.blended_megatron_dataset_builder.get_real_group_size',
                   return_value=1):
            with patch('mindformers.dataset.blended_datasets.blended_megatron_dataset_builder.get_real_rank',
                       return_value=0):
                config.split_matrix = [None, None, None]
                config.split_matrix[0] = (0.0, 1.0)

                with pytest.raises(Exception,
                                   match="FailingDummyMegatronDataset failed to build as a mock data generator"):
                    builder.build()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_build_blended_dataset_single_prefix(self):
        """
        Feature: build method with single prefix blend
        Description: Test build method works with single prefix blend configuration
        Expectation: Method builds datasets correctly
        """
        config = create_test_config()
        config.mock = False
        config.blend = (["single_prefix"], None)

        builder = create_test_builder(
            builder_kwargs={"config": config}
        )

        with patch('mindformers.dataset.blended_datasets.blended_megatron_dataset_builder.get_real_group_size',
                   return_value=1):
            with patch('mindformers.dataset.blended_datasets.blended_megatron_dataset_builder.get_real_rank',
                       return_value=0):
                config.split_matrix = [None, None, None]
                config.split_matrix[0] = (0.0, 1.0)

                datasets = builder.build()
                assert isinstance(datasets, list)
                assert len(datasets) == len(Split)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_build_with_blend_per_split(self):
        """
        Feature: build method with blend_per_split
        Description: Test build method works with blend_per_split configuration
        Expectation: Method builds datasets correctly
        """
        config = create_test_config()
        config.mock = False
        config.blend = None
        config.blend_per_split = [(["prefix1"], None), None, None]

        builder = create_test_builder(
            builder_kwargs={"config": config}
        )

        with patch('mindformers.dataset.blended_datasets.blended_megatron_dataset_builder.get_real_group_size',
                   return_value=1):
            with patch('mindformers.dataset.blended_datasets.blended_megatron_dataset_builder.get_real_rank',
                       return_value=0):
                config.split_matrix = [None, None, None]
                config.split_matrix[0] = (0.0, 1.0)

                datasets = builder.build()
                assert isinstance(datasets, list)
                assert len(datasets) == len(Split)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_build_with_blend_per_split_single_prefix(self):
        """
        Feature: build method with blend_per_split single prefix
        Description: Test build method works with blend_per_split single prefix configuration
        Expectation: Method builds datasets correctly
        """
        config = create_test_config()
        config.mock = False
        config.blend = None
        config.blend_per_split = [(["single_prefix"], None), None, None]

        builder = create_test_builder(
            builder_kwargs={"config": config}
        )

        with patch('mindformers.dataset.blended_datasets.blended_megatron_dataset_builder.get_real_group_size',
                   return_value=1):
            with patch('mindformers.dataset.blended_datasets.blended_megatron_dataset_builder.get_real_rank',
                       return_value=0):
                config.split_matrix = [None, None, None]
                config.split_matrix[0] = (0.0, 1.0)

                datasets = builder.build()
                assert isinstance(datasets, list)
                assert len(datasets) == len(Split)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_build_with_blend_weights_and_size(self):
        """
        Feature: build method with blend weights and size
        Description: Test build method works with blend configuration having weights and size
        Expectation: Method builds datasets correctly with weights processing
        """
        config = create_test_config()
        config.mock = False
        config.blend = (["prefix1", "prefix2"], [0.3, 0.7])

        builder = create_test_builder(
            builder_kwargs={"config": config}
        )

        with patch('mindformers.dataset.blended_datasets.blended_megatron_dataset_builder.get_real_group_size',
                   return_value=1):
            with patch('mindformers.dataset.blended_datasets.blended_megatron_dataset_builder.get_real_rank',
                       return_value=0):
                config.split_matrix = [None, None, None]
                config.split_matrix[0] = (0.0, 1.0)

                datasets = builder.build()
                assert isinstance(datasets, list)
                assert len(datasets) == len(Split)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_build_verification_logic(self):
        """
        Feature: build method verification logic
        Description: Test build method verification logic with cached dataset
        Expectation: Verification passes correctly
        """
        config = create_test_config()
        config.mock = False
        config.blend = (["prefix1", "prefix2"], [0.5, 0.5])

        builder = create_test_builder(
            builder_kwargs={"config": config}
        )

        class MockBlendedDataset:
            """A mock blended dataset for testing verification logic"""

            def __init__(self):
                self.built_anew_on_cache_miss = True
                self.split = type('Split', (), {'name': 'train'})()
                self.size = 10
                self.dataset_index = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

                class MockDataset:
                    def __init__(self, length):
                        self.length = length

                    def __len__(self):
                        return self.length

                mock_dataset1 = MockDataset(5)
                mock_dataset2 = MockDataset(5)
                self.datasets = [mock_dataset1, mock_dataset2]

            def __len__(self):
                return 10

        with patch('mindformers.dataset.blended_datasets.blended_megatron_dataset_builder.get_real_group_size',
                   return_value=1):
            with patch('mindformers.dataset.blended_datasets.blended_megatron_dataset_builder.get_real_rank',
                       return_value=0):
                with patch.object(builder, '_build_blended_dataset_splits',
                                  return_value=[MockBlendedDataset()]):
                    datasets = builder.build()
                    assert isinstance(datasets, list)
                    assert len(datasets) == 1

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_build_verification_logic_cached_no_check(self):
        """
        Feature: build method verification logic for cached datasets
        Description: Test build method skips verification for fully cached datasets
        Expectation: Verification is skipped and info logged
        """
        config = create_test_config()
        config.mock = False
        config.blend = (["prefix1"], None)

        builder = create_test_builder(
            builder_kwargs={"config": config}
        )

        class MockCachedBlendedDataset:
            """A mock cached blended dataset for testing verification logic"""

            def __init__(self):
                self.built_anew_on_cache_miss = False
                self.split = type('Split', (), {'name': 'train'})()
                self.size = None
                self.dataset_index = np.array([0, 1, 0, 1])

                class MockDataset:
                    def __len__(self):
                        return 5

                self.datasets = [MockDataset()]

            def __len__(self):
                return 5

        with patch('mindformers.dataset.blended_datasets.blended_megatron_dataset_builder.get_real_group_size',
                   return_value=1):
            with patch('mindformers.dataset.blended_datasets.blended_megatron_dataset_builder.get_real_rank',
                       return_value=0):
                with patch.object(builder, '_build_blended_dataset_splits',
                                  return_value=[MockCachedBlendedDataset()]):
                    datasets = builder.build()
                    assert isinstance(datasets, list)
                    assert len(datasets) == 1

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_build_parallel_datasets(self):
        """
        Feature: _build_megatron_datasets_parallel method
        Description: Test parallel building of megatron datasets
        Expectation: Method builds datasets in parallel correctly
        """
        config = create_test_config()
        config.mock = False
        config.blend = (["prefix1", "prefix2"], [0.5, 0.5])
        config.num_dataset_builder_threads = 2

        builder = create_test_builder(
            builder_kwargs={"config": config}
        )

        with patch('mindformers.dataset.blended_datasets.blended_megatron_dataset_builder.get_real_group_size',
                   return_value=1):
            with patch('mindformers.dataset.blended_datasets.blended_megatron_dataset_builder.get_real_rank',
                       return_value=0):
                config.split_matrix = [None, None, None]
                config.split_matrix[0] = (0.0, 1.0)

                datasets = builder.build()
                assert isinstance(datasets, list)
                assert len(datasets) == len(Split)
