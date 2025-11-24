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
"""Test cases for LLMDataset class with YAML configuration files."""
# pylint: disable=redefined-outer-name,unused-argument,too-few-public-methods
import os
import pytest
import mindspore as ms
from mindspore import context
from mindformers.tools.register.config import MindFormerConfig, DictConfig
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


@pytest.fixture
def get_config_dir():
    """Get the directory containing test configuration files."""
    return os.path.dirname(os.path.abspath(__file__))


class TestLLMDatasetWithYAMLFiles:
    """Test cases for loading LLMDataset configuration from YAML files."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_load_pretrain_config_from_yaml(self, setup_context, get_config_dir):
        """Test loading pretrain configuration from YAML file using MindFormerConfig."""
        yaml_path = os.path.join(get_config_dir, "test_pretrain_config.yaml")

        if not os.path.exists(yaml_path):
            pytest.skip(f"Test configuration file not found: {yaml_path}")

        # Load configuration using MindFormerConfig
        config = MindFormerConfig(yaml_path)

        # Create LLMDataset with the loaded configuration
        dataset = LLMDataset(dataset_config=config.train_dataset)

        # Verify dataset properties
        assert dataset.data_loader_type == "BlendedMegatronDatasetDataLoader"
        assert dataset.is_create_attention_mask()
        assert not dataset.is_create_compressed_eod_mask()

        # Verify configuration details
        assert dataset.data_loader_config.config.seq_length == 8192
        assert dataset.data_loader_config.config.eod == 0
        assert dataset.data_loader_config.config.pad == 1

        # Verify new configuration parameters
        assert dataset.data_loader_config.config.eod_mask_loss
        assert dataset.data_loader_config.config.reset_position_ids
        assert dataset.data_loader_config.config.reset_attention_mask
        assert dataset.data_loader_config.config.eod_pad_length == 128

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_load_finetune_config_from_yaml(self, setup_context, get_config_dir):
        """Test loading finetune configuration from YAML file using MindFormerConfig."""
        yaml_path = os.path.join(get_config_dir, "test_finetune_config.yaml")

        if not os.path.exists(yaml_path):
            pytest.skip(f"Test configuration file not found: {yaml_path}")

        # Load configuration using MindFormerConfig
        config = MindFormerConfig(yaml_path)

        # Create LLMDataset with the loaded configuration
        dataset = LLMDataset(dataset_config=config.train_dataset)

        # Verify dataset properties
        assert dataset.data_loader_type == "HFDataLoader"
        assert dataset.is_create_attention_mask()
        assert not dataset.is_create_compressed_eod_mask()

        # Verify configuration details
        assert dataset.data_loader_config.load_func == "load_dataset"
        assert dataset.data_loader_config.path == "llm-wizard/alpaca-gpt4-data-zh"
        assert not dataset.data_loader_config.shuffle
        assert dataset.data_loader_config.use_broadcast_data
        assert dataset.data_loader_config.compressed_eod_mask_length == 128

        # Verify handler configuration
        handler_list = dataset.data_loader_config.get("handler")
        assert handler_list is not None
        assert len(handler_list) == 3  # take, AlpacaInstructDataHandler, PackingHandler
        assert handler_list[0]['type'] == 'take'
        assert handler_list[0]['n'] == 2000
        assert handler_list[1]['type'] == 'AlpacaInstructDataHandler'
        assert handler_list[2]['type'] == 'PackingHandler'

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_get_default_input_columns_from_yaml(self, setup_context, get_config_dir):
        """Test getting default input columns from YAML configuration."""
        yaml_path = os.path.join(get_config_dir, "test_pretrain_config.yaml")

        if not os.path.exists(yaml_path):
            pytest.skip(f"Test configuration file not found: {yaml_path}")

        config = MindFormerConfig(yaml_path)
        dataset = LLMDataset(dataset_config=config.train_dataset)

        # Get default input columns
        create_compressed_eod_mask = dataset.is_create_compressed_eod_mask()
        create_attention_mask = dataset.is_create_attention_mask()
        input_columns = dataset.get_default_input_columns(
            create_attention_mask,
            create_compressed_eod_mask
        )

        # Verify columns
        expected_columns = ["input_ids", "labels", "loss_mask", "position_ids", "attention_mask"]
        assert input_columns == expected_columns

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_set_dataset_config_from_yaml(self, setup_context, get_config_dir):
        """Test setting dataset configuration from YAML."""
        yaml_path = os.path.join(get_config_dir, "test_pretrain_config.yaml")

        if not os.path.exists(yaml_path):
            pytest.skip(f"Test configuration file not found: {yaml_path}")

        config = MindFormerConfig(yaml_path)
        dataset = LLMDataset(dataset_config=config.train_dataset)

        # Set MindSpore dataset configuration
        dataset_seed = 1234
        prefetch_size = config.train_dataset.get("prefetch_size", 1)
        numa_enable = config.train_dataset.get("numa_enable", False)

        dataset.set_ms_dataset_config(
            dataset_seed=dataset_seed,
            prefetch_size=prefetch_size,
            numa_enable=numa_enable
        )

        # Verify no exceptions were raised
        assert True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_generate_shard_info_from_yaml(self, setup_context, get_config_dir):
        """Test generating shard info with configuration from YAML."""
        yaml_path = os.path.join(get_config_dir, "test_finetune_config.yaml")

        if not os.path.exists(yaml_path):
            pytest.skip(f"Test configuration file not found: {yaml_path}")

        config = MindFormerConfig(yaml_path)
        dataset = LLMDataset(dataset_config=config.train_dataset)

        # Generate shard info
        data_parallel_size = 1
        shard_id, num_shards = dataset.generate_shard_info(
            data_parallel_size=data_parallel_size
        )

        # Verify shard info (in single device mode)
        assert shard_id == 0
        assert num_shards == 1


class TestLLMDatasetUsagePattern:
    """Test cases following the usage pattern from llm_trainer.py."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_create_dataset_like_trainer(self, setup_context, get_config_dir):
        """Test creating dataset following the pattern used in llm_trainer.py."""
        yaml_path = os.path.join(get_config_dir, "test_pretrain_config.yaml")

        if not os.path.exists(yaml_path):
            pytest.skip(f"Test configuration file not found: {yaml_path}")

        # Load configuration
        config = MindFormerConfig(yaml_path)

        # Step 1: Create LLMDataset instance (similar to line 1221 in llm_trainer.py)
        llm_dataset = LLMDataset(dataset_config=config.train_dataset)

        # Step 2: Set dataset seed (similar to lines 1222-1228 in llm_trainer.py)
        dataset_seed = 1234
        llm_dataset.set_ms_dataset_config(
            dataset_seed=dataset_seed,
            prefetch_size=config.train_dataset.get("prefetch_size", 1),
            numa_enable=config.train_dataset.get("numa_enable", False)
        )

        # Step 3: Generate shard info (similar to lines 1230-1232 in llm_trainer.py)
        data_parallel_size = 1
        shard_id, num_shards = llm_dataset.generate_shard_info(
            data_parallel_size=data_parallel_size
        )

        # Step 4: Get input columns (similar to lines 1234-1237 in llm_trainer.py)
        create_compressed_eod_mask = llm_dataset.is_create_compressed_eod_mask()
        create_attention_mask = llm_dataset.is_create_attention_mask()
        input_columns = llm_dataset.get_default_input_columns(
            create_attention_mask,
            create_compressed_eod_mask
        )

        # Verify all steps completed successfully
        assert llm_dataset is not None
        assert shard_id is not None
        assert num_shards is not None
        assert input_columns is not None
        assert len(input_columns) > 0

        # Verify expected values
        assert llm_dataset.data_loader_type == "BlendedMegatronDatasetDataLoader"
        assert shard_id == 0
        assert num_shards == 1
        expected_columns = ["input_ids", "labels", "loss_mask", "position_ids", "attention_mask"]
        assert input_columns == expected_columns


class TestLLMDatasetDetailedConfiguration:
    """Test cases for detailed configuration parameters from YAML files."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_pretrain_megatron_dataset_sizes(self, setup_context, get_config_dir):
        """Test Megatron dataset sizes configuration."""
        yaml_path = os.path.join(get_config_dir, "test_pretrain_config.yaml")

        if not os.path.exists(yaml_path):
            pytest.skip(f"Test configuration file not found: {yaml_path}")

        config = MindFormerConfig(yaml_path)
        dataset = LLMDataset(dataset_config=config.train_dataset)

        # Verify sizes configuration
        sizes = dataset.data_loader_config.get("sizes")
        assert sizes is not None
        assert len(sizes) == 3
        assert sizes[0] == 1000  # train size
        assert sizes[1] == 0     # test size
        assert sizes[2] == 0     # eval size

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_pretrain_data_path_mixing(self, setup_context, get_config_dir):
        """Test Megatron dataset multiple data path mixing configuration."""
        yaml_path = os.path.join(get_config_dir, "test_pretrain_config.yaml")

        if not os.path.exists(yaml_path):
            pytest.skip(f"Test configuration file not found: {yaml_path}")

        config = MindFormerConfig(yaml_path)
        dataset = LLMDataset(dataset_config=config.train_dataset)

        # Verify data_path mixing configuration
        data_path = dataset.data_loader_config.config.get("data_path")
        assert data_path is not None
        assert len(data_path) == 4
        assert data_path[0] == '0.3'  # First dataset weight
        assert data_path[2] == '0.7'  # Second dataset weight
        assert '/path/megatron_data1' in data_path[1]
        assert '/path/megatron_data2' in data_path[3]

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_finetune_handler_pipeline(self, setup_context, get_config_dir):
        """Test finetune dataset handler pipeline configuration."""
        yaml_path = os.path.join(get_config_dir, "test_finetune_config.yaml")

        if not os.path.exists(yaml_path):
            pytest.skip(f"Test configuration file not found: {yaml_path}")

        config = MindFormerConfig(yaml_path)
        dataset = LLMDataset(dataset_config=config.train_dataset)

        # Verify complete handler pipeline
        handler_list = dataset.data_loader_config.get("handler")

        # Check AlpacaInstructDataHandler configuration
        alpaca_handler = handler_list[1]
        assert alpaca_handler['type'] == 'AlpacaInstructDataHandler'
        assert alpaca_handler['seq_length'] == 4096
        assert not alpaca_handler['padding']
        assert 'tokenizer' in alpaca_handler
        assert alpaca_handler['tokenizer']['trust_remote_code']
        assert alpaca_handler['tokenizer']['padding_side'] == 'right'

        # Check PackingHandler configuration
        packing_handler = handler_list[2]
        assert packing_handler['type'] == 'PackingHandler'
        assert packing_handler['seq_length'] == 4096
        assert packing_handler['pack_strategy'] == 'pack'

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_dataset_parallel_workers_config(self, get_config_dir):
        """Test dataset parallel workers configuration."""
        yaml_path = os.path.join(get_config_dir, "test_pretrain_config.yaml")

        if not os.path.exists(yaml_path):
            pytest.skip(f"Test configuration file not found: {yaml_path}")

        config = MindFormerConfig(yaml_path)

        # Verify parallel processing configuration
        assert config.train_dataset.get("num_parallel_workers") == 8
        assert not config.train_dataset.get("python_multiprocessing")
        assert not config.train_dataset.get("numa_enable")
        assert config.train_dataset.get("prefetch_size") == 1

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_dataset_input_output_columns(self, get_config_dir):
        """Test dataset input and output columns configuration."""
        yaml_path = os.path.join(get_config_dir, "test_finetune_config.yaml")

        if not os.path.exists(yaml_path):
            pytest.skip(f"Test configuration file not found: {yaml_path}")

        config = MindFormerConfig(yaml_path)

        # Verify columns configuration
        expected_columns = ["input_ids", "labels", "loss_mask", "position_ids", "attention_mask"]
        assert config.train_dataset.get("input_columns") == expected_columns
        assert config.train_dataset.get("construct_args_key") == expected_columns
        assert config.train_dataset.get("drop_remainder")

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_pretrain_eod_and_pad_tokens(self, setup_context, get_config_dir):
        """Test EOD and PAD token configuration for pretrain dataset."""
        yaml_path = os.path.join(get_config_dir, "test_pretrain_config.yaml")

        if not os.path.exists(yaml_path):
            pytest.skip(f"Test configuration file not found: {yaml_path}")

        config = MindFormerConfig(yaml_path)
        dataset = LLMDataset(dataset_config=config.train_dataset)

        # Verify EOD and PAD token IDs
        assert dataset.data_loader_config.config.eod == 0
        assert dataset.data_loader_config.config.pad == 1

        # Verify EOD-related settings
        assert dataset.data_loader_config.config.eod_mask_loss
        assert dataset.data_loader_config.config.eod_pad_length == 128


class TestLLMDatasetEdgeCases:
    """Test cases for edge cases and error handling."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_dataset_with_empty_handler(self):
        """Test dataset with empty handler list."""
        config = {
            "data_loader": {
                "type": "HFDataLoader",
                "load_func": "load_dataset",
                "path": "test",
                "use_broadcast_data": False,
                "handler": []
            }
        }
        dataset = LLMDataset(dataset_config=config)
        assert dataset.data_loader_type == "HFDataLoader"

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_dataset_with_packing_handler(self):
        """Test dataset with PackingHandler automatically enables attention mask."""
        # Use DictConfig for data_loader to support attribute access (required by PackingHandler)
        data_loader_config = DictConfig(**{
            "type": "HFDataLoader",
            "load_func": "load_dataset",
            "path": "test",
            "use_broadcast_data": False,
            "create_attention_mask": False,  # Initially False
            "handler": [
                {
                    "type": "PackingHandler",
                    "seq_length": 4096,
                    "pack_strategy": "pack"
                }
            ]
        })
        config = {
            "data_loader": data_loader_config
        }
        dataset = LLMDataset(dataset_config=config)
        # PackingHandler should automatically enable create_attention_mask
        assert dataset.data_loader_config.create_attention_mask

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_dataset_config_deepcopy(self):
        """Test that dataset config is deep copied and modifications don't affect original."""
        original_config = {
            "data_loader": {
                "type": "HFDataLoader",
                "load_func": "load_dataset",
                "path": "test",
                "use_broadcast_data": False
            }
        }

        dataset = LLMDataset(dataset_config=original_config)

        # Modify dataset's config
        dataset.data_loader_config["path"] = "modified_path"

        # Original config should remain unchanged
        assert original_config["data_loader"]["path"] == "test"
        assert dataset.data_loader_config["path"] == "modified_path"
