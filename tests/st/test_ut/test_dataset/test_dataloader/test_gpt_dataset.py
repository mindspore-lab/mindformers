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
"""test gpt dataset"""

import os
import subprocess
import tempfile
import time
import shutil
import glob

import pytest
import numpy as np

from mindformers.dataset.blended_datasets.indexed_dataset import IndexedDataset, IndexedDatasetBuilder
from mindformers.dataset.blended_datasets.gpt_dataset import (
    GPTDatasetConfig,
    GPTDataset,
    MockGPTDataset,
    MockGPTLowLevelDataset,
    _get_ltor_masks_and_position_ids,
    _get_eod_attention_mask,
    _build_document_index,
    _build_shuffle_index
)
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


def create_test_indexed_dataset(temp_dir_path):
    """Create a simple indexed dataset for testing"""
    bin_file = os.path.join(temp_dir_path, "test_dataset.bin")
    idx_file = os.path.join(temp_dir_path, "test_dataset.idx")

    data_size = 10
    seq_length = 64
    random_ids = [np.random.randint(low=1, high=100, size=seq_length) for _ in range(data_size)]

    builder = IndexedDatasetBuilder(bin_file, dtype=np.int32)
    for random_id in random_ids:
        builder.add_document(random_id, [len(random_id)])
    builder.finalize(idx_file)

    return bin_file.replace(".bin", "")


def create_test_config(**kwargs):
    """Create a GPTDatasetConfig with default values and optional overrides"""
    default_config = {
        "sequence_length": 32,
        "random_seed": 1234,
        "tokenizer": DummyTokenizer(),
        "reset_position_ids": False,
        "reset_attention_mask": False,
        "eod_mask_loss": False,
        "path_to_cache": None,
        "mmap_bin_files": False
    }
    default_config.update(kwargs)
    return GPTDatasetConfig(**default_config)


def create_test_dataset(config_kwargs=None, dataset_kwargs=None):
    """Create a MockGPTDataset with common setup"""
    if config_kwargs is None:
        config_kwargs = {}

    config = create_test_config(**config_kwargs)

    if dataset_kwargs is None:
        dataset_kwargs = {}

    default_dataset_kwargs = {
        "dataset": MockGPTLowLevelDataset(config.tokenizer),
        "dataset_path": None,
        "indices": np.arange(100),
        "num_samples": 10,
        "index_split": Split.train,
        "config": config
    }
    default_dataset_kwargs.update(dataset_kwargs)

    return MockGPTDataset(**default_dataset_kwargs)


class TestGPTDatasetInitialization:
    """Test GPT dataset initialization"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_gpt_dataset_real_initialization(self):
        """
        Feature: GPTDataset initialization
        Description: Test GPTDataset can be initialized correctly with real dataset
        Expectation: Dataset initializes without error and has expected properties
        """
        temp_dir_path = tempfile.mkdtemp()
        try:
            # Test real GPTDataset initialization
            dataset_path = create_test_indexed_dataset(temp_dir_path)
            config = create_test_config(
                sequence_length=64,
                reset_position_ids=False,
                reset_attention_mask=False,
                eod_mask_loss=False
            )

            indexed_dataset = IndexedDataset(dataset_path, multimodal=False, mmap=False)

            gpt_dataset = GPTDataset(
                indexed_dataset=indexed_dataset,
                dataset_path=dataset_path,
                indexed_indices=np.arange(len(indexed_dataset)),
                num_samples=5,
                index_split=Split.train,
                config=config
            )

            assert isinstance(gpt_dataset, GPTDataset)
            assert len(gpt_dataset) > 0
            item = gpt_dataset[0]
            assert len(item) >= 4
        finally:
            shutil.rmtree(temp_dir_path)


class TestMockGPTDatasetFunctionality:
    """Test Mock GPT dataset functionality"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_mock_gpt_dataset_configurations(self):
        """
        Feature: MockGPTDataset configurations
        Description: Test MockGPTDataset with various configurations
        Expectation: Dataset works correctly with all configurations
        """
        # Test MockGPTDataset with various configurations
        test_configs = [
            # Default config
            {},
            # With attention mask
            {"create_attention_mask": True},
            # With compressed EOD mask
            {"create_compressed_eod_mask": True, "eod_pad_length": 64},
            # Minimal config
            {
                "create_attention_mask": False,
                "create_compressed_eod_mask": False
            }
        ]

        for config_kwargs in test_configs:
            dataset = create_test_dataset(config_kwargs)
            item = dataset[0]

            # Verify basic structure
            assert len(item) >= 4

        # Test batch padding separately
        dataset = create_test_dataset({})
        padding_item = dataset[None]
        assert len(padding_item) >= 4
        assert np.all(padding_item[2] == 0)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_mock_gpt_dataset_advanced_features(self):
        """
        Feature: MockGPTDataset advanced features
        Description: Test MockGPTDataset with advanced features
        Expectation: Dataset works correctly with advanced features
        """
        # Test with all major features
        dataset = create_test_dataset({
            "reset_position_ids": True,
            "reset_attention_mask": True,
            "eod_mask_loss": True,
            "create_attention_mask": True,
            "add_extra_token_to_sequence": False
        })
        item = dataset[0]
        assert len(item) >= 4


class TestGPTDatasetComponents:
    """Test GPT dataset components"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_mock_gpt_low_level_dataset(self):
        """
        Feature: MockGPTLowLevelDataset
        Description: Test MockGPTLowLevelDataset functionality
        Expectation: Dataset works correctly
        """
        # Test MockGPTLowLevelDataset
        tokenizer = DummyTokenizer()
        mock_dataset = MockGPTLowLevelDataset(tokenizer)
        assert len(mock_dataset) == 100000
        item = mock_dataset[0]
        assert isinstance(item, np.ndarray)
        sliced_item = mock_dataset.get(0, offset=0, length=10)
        assert len(sliced_item) == 10

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_utility_functions(self):
        """
        Feature: Utility functions
        Description: Test utility functions
        Expectation: Functions work correctly
        """
        # Test utility functions
        test_data = np.array([1, 2, 2, 3, 4, 2, 5, 6])
        eod_token = 2

        # Test _get_ltor_masks_and_position_ids with different combinations
        combinations = [
            (False, False, False, True),  # Basic case
            (True, False, False, True),  # Reset position IDs
            (False, True, False, True),  # Reset attention mask
            (False, False, True, True),  # EOD mask loss
            (True, True, True, True),  # All features
            (False, False, False, False)  # No attention mask
        ]

        for reset_pos, reset_attn, eod_loss, create_attn in combinations:
            attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
                test_data, eod_token, reset_pos, reset_attn, eod_loss, create_attn
            )
            assert loss_mask.shape == (8,)
            assert position_ids.shape == (8,)

            if create_attn:
                assert attention_mask is not None
            else:
                assert attention_mask is None

        # Test _get_eod_attention_mask
        eod_mask = _get_eod_attention_mask(test_data, eod_token, 10)
        assert eod_mask.shape == (10,)

        # Test _build_document_index
        documents = np.array([1, 2, 3], dtype=np.int32)
        numpy_random_state = np.random.RandomState(1234)
        doc_index = _build_document_index(documents, 2, numpy_random_state, False)
        assert len(doc_index) == 6
        assert doc_index.dtype == np.int32

        # Test _build_shuffle_index
        shuffle_idx = _build_shuffle_index(5, 5, numpy_random_state)
        assert len(shuffle_idx) == 5

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_config_validation(self):
        """
        Feature: GPTDatasetConfig validation
        Description: Test GPTDatasetConfig validation
        Expectation: Validation works correctly
        """
        # Test config validation
        with pytest.raises(AssertionError):
            GPTDatasetConfig(
                sequence_length=32,
                random_seed=1234,
                tokenizer=DummyTokenizer()
            )

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_cacheability_logic(self):
        """
        Feature: Cacheability logic
        Description: Test cacheability logic
        Expectation: Logic works correctly
        """
        # Test cacheability logic
        dataset = create_test_dataset({
            "reset_position_ids": False,
            "reset_attention_mask": False,
            "eod_mask_loss": False
        })
        assert dataset.masks_and_position_ids_are_cacheable is True

        dataset = create_test_dataset({
            "reset_position_ids": True,
            "reset_attention_mask": False,
            "eod_mask_loss": False
        })
        assert dataset.masks_and_position_ids_are_cacheable is False

        # Test pad and eod token id handling using public methods if available
        # pylint: disable=protected-access
        assert dataset._pad_token_id == 0
        assert dataset._eod_token_id == 2
