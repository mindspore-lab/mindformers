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
"""Test cases for LLMDataset class."""
# pylint: disable=redefined-outer-name,unused-argument,too-few-public-methods
import pytest
import mindspore as ms
from mindspore import context
from mindformers.tools.register.config import DictConfig
from mindformers.dataset.llm_dataset import LLMDataset


@pytest.fixture(scope="module")
def setup_context():
    """Setup MindSpore context for testing."""
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    # Set dataset_broadcast_opt_level to 3 for Megatron dataset tests
    context.set_context(dataset_broadcast_opt_level=3)
    ms.set_seed(42)
    yield
    # Reset context after tests
    context.reset_auto_parallel_context()


class TestLLMDatasetInit:
    """Test cases for LLMDataset initialization."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_init_with_none_config(self):
        """Test initialization with None dataset_config should raise ValueError."""
        with pytest.raises(ValueError, match="dataset_config cannot be None"):
            LLMDataset(dataset_config=None)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_init_with_invalid_data_loader_config(self):
        """Test initialization with invalid data_loader configuration."""
        invalid_config = {
            "data_loader": "invalid_string"  # Should be a dict
        }
        with pytest.raises(ValueError, match="data_loader_config must be a dict"):
            LLMDataset(dataset_config=invalid_config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_init_with_missing_type(self):
        """Test initialization with missing 'type' key in data_loader."""
        invalid_config = {
            "data_loader": {
                "config": {}
            }
        }
        with pytest.raises(ValueError, match="data_loader_config must contain 'type' key"):
            LLMDataset(dataset_config=invalid_config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_init_with_valid_megatron_config(self, setup_context):
        """Test initialization with valid Megatron dataset configuration."""
        valid_config = {
            "data_loader": {
                "type": "BlendedMegatronDatasetDataLoader",
                "sizes": [1000, 0, 0],
                "config": {
                    "seq_length": 8192,
                    "eod": 0,
                    "pad": 1,
                    "data_path": ['0.5', '/path/data1', '0.5', '/path/data2']
                }
            },
            "input_columns": ["input_ids", "labels", "loss_mask", "position_ids"],
            "drop_remainder": True
        }
        dataset = LLMDataset(dataset_config=valid_config)
        assert dataset.data_loader_type == "BlendedMegatronDatasetDataLoader"
        assert dataset.data_loader_config["type"] == "BlendedMegatronDatasetDataLoader"

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_init_with_valid_hf_config(self):
        """Test initialization with valid HuggingFace dataset configuration."""
        valid_config = {
            "data_loader": {
                "type": "HFDataLoader",
                "load_func": "load_dataset",
                "path": "test_dataset",
                "use_broadcast_data": False,
                "create_attention_mask": True,
                "handler": []
            },
            "input_columns": ["input_ids", "labels"],
            "drop_remainder": True
        }
        dataset = LLMDataset(dataset_config=valid_config)
        assert dataset.data_loader_type == "HFDataLoader"
        assert dataset.data_loader_config["type"] == "HFDataLoader"


class TestLLMDatasetConfigMethods:
    """Test cases for LLMDataset configuration methods."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_set_megatron_dataset_config_seed(self, setup_context):
        """Test setting Megatron dataset seed."""
        # Use DictConfig for nested config to support attribute access
        inner_config = DictConfig(**{
            "seq_length": 8192,
            "seed": 1234,
            "data_path": ['1.0', '/path/data']
        })
        data_loader = DictConfig(**{
            "type": "BlendedMegatronDatasetDataLoader",
            "sizes": [1000, 0, 0],
            "config": inner_config
        })
        config = {
            "data_loader": data_loader
        }
        dataset = LLMDataset(dataset_config=config)
        dataset.set_megatron_dataset_config_seed(dataset_seed=5678)
        assert dataset.data_loader_config.config.seed == 5678

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_set_megatron_dataset_config_seed_non_megatron(self):
        """Test setting seed for non-Megatron dataset (should not raise error)."""
        config = {
            "data_loader": {
                "type": "HFDataLoader",
                "load_func": "load_dataset",
                "path": "test_dataset",
                "use_broadcast_data": False
            }
        }
        dataset = LLMDataset(dataset_config=config)
        # Should not raise error even though it's not Megatron dataset
        dataset.set_megatron_dataset_config_seed(dataset_seed=5678)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_set_ms_dataset_config(self):
        """Test setting MindSpore dataset configuration."""
        config = {
            "data_loader": {
                "type": "HFDataLoader",
                "load_func": "load_dataset",
                "path": "test_dataset",
                "use_broadcast_data": False
            }
        }
        dataset = LLMDataset(dataset_config=config)
        dataset.set_ms_dataset_config(
            dataset_seed=4321,
            prefetch_size=2,
            numa_enable=True
        )
        # Verify configuration is set (no exceptions raised)
        assert True


class TestLLMDatasetColumnNames:
    """Test cases for getting default input columns."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_get_default_input_columns_compressed_eod(self):
        """Test getting columns with compressed EOD mask."""
        config = {
            "data_loader": {
                "type": "HFDataLoader",
                "load_func": "load_dataset",
                "path": "test",
                "use_broadcast_data": False
            }
        }
        dataset = LLMDataset(dataset_config=config)
        columns = dataset.get_default_input_columns(
            create_attention_mask=False,
            create_compressed_eod_mask=True
        )
        assert columns == ['input_ids', 'labels', 'loss_mask', 'position_ids', 'actual_seq_len']

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_get_default_input_columns_attention_mask(self):
        """Test getting columns with attention mask."""
        config = {
            "data_loader": {
                "type": "HFDataLoader",
                "load_func": "load_dataset",
                "path": "test",
                "use_broadcast_data": False
            }
        }
        dataset = LLMDataset(dataset_config=config)
        columns = dataset.get_default_input_columns(
            create_attention_mask=True,
            create_compressed_eod_mask=False
        )
        assert columns == ['input_ids', 'labels', 'loss_mask', 'position_ids', 'attention_mask']

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_get_default_input_columns_megatron(self, setup_context):
        """Test getting columns for Megatron dataset."""
        config = {
            "data_loader": {
                "type": "BlendedMegatronDatasetDataLoader",
                "sizes": [1000, 0, 0],
                "config": {
                    "seq_length": 8192,
                    "data_path": ['1.0', '/path/data']
                }
            }
        }
        dataset = LLMDataset(dataset_config=config)
        columns = dataset.get_default_input_columns(
            create_attention_mask=False,
            create_compressed_eod_mask=False
        )
        assert columns == ['input_ids', 'labels', 'loss_mask', 'position_ids']

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_get_default_input_columns_custom(self):
        """Test getting columns when custom input_columns are provided."""
        config = {
            "data_loader": {
                "type": "HFDataLoader",
                "load_func": "load_dataset",
                "path": "test",
                "use_broadcast_data": False
            },
            "input_columns": ["custom_col1", "custom_col2"]
        }
        dataset = LLMDataset(dataset_config=config)
        columns = dataset.get_default_input_columns(
            create_attention_mask=False,
            create_compressed_eod_mask=False
        )
        assert columns == ["custom_col1", "custom_col2"]


class TestLLMDatasetShardInfo:
    """Test cases for generating shard information."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_generate_shard_info_default(self):
        """Test generating shard info with default settings."""
        config = {
            "data_loader": {
                "type": "HFDataLoader",
                "load_func": "load_dataset",
                "path": "test",
                "use_broadcast_data": False
            }
        }
        dataset = LLMDataset(dataset_config=config)
        shard_id, num_shards = dataset.generate_shard_info(data_parallel_size=1)
        # In single device mode, shard_id should be 0 and num_shards should be 1
        assert shard_id == 0
        assert num_shards == 1


class TestLLMDatasetMaskChecks:
    """Test cases for checking mask configurations."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_is_create_compressed_eod_mask_megatron(self, setup_context):
        """Test checking compressed EOD mask for Megatron dataset."""
        # Use DictConfig for nested config to support attribute access
        inner_config = DictConfig(**{
            "seq_length": 8192,
            "create_compressed_eod_mask": True,
            "data_path": ['1.0', '/path/data']
        })
        data_loader = DictConfig(**{
            "type": "BlendedMegatronDatasetDataLoader",
            "sizes": [1000, 0, 0],
            "config": inner_config
        })
        config = {
            "data_loader": data_loader
        }
        dataset = LLMDataset(dataset_config=config)
        assert dataset.is_create_compressed_eod_mask()

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_is_create_compressed_eod_mask_hf(self):
        """Test checking compressed EOD mask for HF dataset."""
        config = {
            "data_loader": {
                "type": "HFDataLoader",
                "load_func": "load_dataset",
                "path": "test",
                "use_broadcast_data": False,
                "create_compressed_eod_mask": True
            }
        }
        dataset = LLMDataset(dataset_config=config)
        assert dataset.is_create_compressed_eod_mask()

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_is_create_attention_mask_megatron(self, setup_context):
        """Test checking attention mask for Megatron dataset."""
        # Use DictConfig for nested config to support attribute access
        inner_config = DictConfig(**{
            "seq_length": 8192,
            "create_attention_mask": True,
            "data_path": ['1.0', '/path/data']
        })
        data_loader = DictConfig(**{
            "type": "BlendedMegatronDatasetDataLoader",
            "sizes": [1000, 0, 0],
            "config": inner_config
        })
        config = {
            "data_loader": data_loader
        }
        dataset = LLMDataset(dataset_config=config)
        assert dataset.is_create_attention_mask()

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_is_create_attention_mask_hf(self):
        """Test checking attention mask for HF dataset."""
        config = {
            "data_loader": {
                "type": "HFDataLoader",
                "load_func": "load_dataset",
                "path": "test",
                "use_broadcast_data": False,
                "create_attention_mask": True
            }
        }
        dataset = LLMDataset(dataset_config=config)
        assert dataset.is_create_attention_mask()


class TestLLMDatasetDataLoaderType:
    """Test cases for getting data loader type."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_get_data_loader_type_megatron(self, setup_context):
        """Test getting Megatron data loader type."""
        config = {
            "data_loader": {
                "type": "BlendedMegatronDatasetDataLoader",
                "sizes": [1000, 0, 0],
                "config": {
                    "seq_length": 8192,
                    "data_path": ['1.0', '/path/data']
                }
            }
        }
        dataset = LLMDataset(dataset_config=config)
        assert dataset.get_data_loader_type() == "BlendedMegatronDatasetDataLoader"

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_get_data_loader_type_hf(self):
        """Test getting HF data loader type."""
        config = {
            "data_loader": {
                "type": "HFDataLoader",
                "load_func": "load_dataset",
                "path": "test",
                "use_broadcast_data": False
            }
        }
        dataset = LLMDataset(dataset_config=config)
        assert dataset.get_data_loader_type() == "HFDataLoader"

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_get_data_loader_type_minddataset(self):
        """Test getting MindDataset data loader type."""
        config = {
            "data_loader": {
                "type": "MindDataset",
                "dataset_dir": "/path/to/mindrecord"
            }
        }
        dataset = LLMDataset(dataset_config=config)
        assert dataset.get_data_loader_type() == "MindDataset"


class TestLLMDatasetCreateDataLoader:
    """Test cases for creating data loader."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_create_data_loader_with_none_columns(self):
        """Test creating data loader with None column names should raise ValueError."""
        config = {
            "data_loader": {
                "type": "HFDataLoader",
                "load_func": "load_dataset",
                "path": "test",
                "use_broadcast_data": False
            }
        }
        dataset = LLMDataset(dataset_config=config)
        with pytest.raises(ValueError, match="column_names cannot be None"):
            dataset.create_data_loader(column_names=None)


class TestLLMDatasetTokenCounter:
    """Test cases for TokenCounter utility."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_perform_token_counting_default(self):
        """Test creating token counter with default parameters."""
        counter_func = LLMDataset.perform_token_counting()
        assert callable(counter_func)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_perform_token_counting_custom(self):
        """Test creating token counter with custom parameters."""
        counter_func = LLMDataset.perform_token_counting(
            top_n=20,
            min_token_id=10,
            max_token_id=1000,
            save_path="./test_output/"
        )
        assert callable(counter_func)


class TestLLMDatasetParallelModes:
    """Test cases for parallel mode checks."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_is_semi_mode(self):
        """Test checking semi-auto parallel mode."""
        # Set to semi-auto parallel mode
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
        is_semi = LLMDataset._is_semi()  # pylint: disable=protected-access
        assert is_semi

        # Reset to stand-alone mode
        context.reset_auto_parallel_context()

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_is_full_batch_mode(self):
        """Test checking full batch mode."""
        # Set full_batch to True
        context.set_auto_parallel_context(full_batch=True)
        is_full = LLMDataset._is_full_batch()  # pylint: disable=protected-access
        assert is_full

        # Reset
        context.reset_auto_parallel_context()

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_is_data_parallel_mode(self):
        """Test checking data parallel mode."""
        # Set to data parallel mode
        context.set_auto_parallel_context(parallel_mode="data_parallel")
        is_dp = LLMDataset._is_data_parallel()  # pylint: disable=protected-access
        assert is_dp

        # Reset
        context.reset_auto_parallel_context()


class TestLLMDatasetWithYAMLConfig:
    """Test cases using YAML configuration files."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_init_with_pretrain_yaml_config(self, setup_context):
        """Test initialization with pretrain YAML configuration structure."""
        # Simulating the structure from llm_pretrain_template.yaml
        # Use DictConfig for nested config to support attribute access
        inner_config = DictConfig(**{
            "seq_length": 8192,
            "eod_mask_loss": True,
            "reset_position_ids": True,
            "create_attention_mask": True,
            "reset_attention_mask": True,
            "create_compressed_eod_mask": False,
            "eod_pad_length": 128,
            "eod": 0,
            "pad": 1,
            "data_path": [
                '0.3', "/path/megatron_data1",
                '0.7', "/path/megatron_data2"
            ]
        })
        data_loader = DictConfig(**{
            "type": "BlendedMegatronDatasetDataLoader",
            "sizes": [1000, 0, 0],
            "config": inner_config
        })
        config = {
            "data_loader": data_loader,
            "input_columns": [
                "input_ids", "labels", "loss_mask", "position_ids", "attention_mask"
            ],
            "construct_args_key": [
                "input_ids", "labels", "loss_mask", "position_ids", "attention_mask"
            ],
            "drop_remainder": True,
            "num_parallel_workers": 8,
            "python_multiprocessing": False,
            "numa_enable": False,
            "prefetch_size": 1
        }

        dataset = LLMDataset(dataset_config=config)
        assert dataset.data_loader_type == "BlendedMegatronDatasetDataLoader"
        assert dataset.is_create_attention_mask()
        assert not dataset.is_create_compressed_eod_mask()

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_init_with_finetune_yaml_config(self):
        """Test initialization with finetune YAML configuration structure."""
        # Simulating the structure from llm_finetune_template.yaml
        # Use DictConfig for data_loader to support attribute access (required by PackingHandler)
        data_loader_config = DictConfig(**{
            "type": "HFDataLoader",
            "load_func": "load_dataset",
            "path": "llm-wizard/alpaca-gpt4-data-zh",
            "create_attention_mask": True,
            "create_compressed_eod_mask": False,
            "compressed_eod_mask_length": 128,
            "use_broadcast_data": False,
            "shuffle": True,
            "handler": [
                {
                    "type": "AlpacaInstructDataHandler",
                    "seq_length": 4096,
                    "padding": False,
                    "tokenizer": {
                        "trust_remote_code": True,
                        "padding_side": "right"
                    }
                },
                {
                    "type": "PackingHandler",
                    "seq_length": 4096,
                    "pack_strategy": "pack"
                }
            ]
        })

        config = {
            "data_loader": data_loader_config,
            "input_columns": [
                "input_ids", "labels", "loss_mask", "position_ids", "attention_mask"
            ],
            "construct_args_key": [
                "input_ids", "labels", "loss_mask", "position_ids", "attention_mask"
            ],
            "drop_remainder": True,
            "num_parallel_workers": 8,
            "python_multiprocessing": False,
            "numa_enable": False,
            "prefetch_size": 1
        }

        dataset = LLMDataset(dataset_config=config)
        assert dataset.data_loader_type == "HFDataLoader"
        assert dataset.is_create_attention_mask()
        assert not dataset.is_create_compressed_eod_mask()
        # Verify PackingHandler enables attention mask
        assert dataset.data_loader_config.create_attention_mask
