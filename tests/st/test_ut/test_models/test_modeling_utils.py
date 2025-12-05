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
"""ModelingUtils test cases"""

from unittest.mock import patch, MagicMock, DEFAULT
import os
import json
import shutil
import tempfile
import pytest
import mindspore as ms
from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.models.modeling_utils import dtype_byte_size, save_checkpoint, shard_checkpoint, \
    load_sharded_checkpoint, _add_variant, PreTrainedModel


# pylint: disable=W0212
class TestModelingUtils:
    """Test cases for modeling utilities functions"""

    def setup_method(self):
        """Set up test environment"""
        # Create mock parameters with different sizes
        self.mock_param1 = MagicMock()
        self.mock_param1.numel.return_value = 1000000  # 1M elements
        self.mock_param1.dtype = ms.float32  # 4 bytes per element = ~4MB

        self.mock_param2 = MagicMock()
        self.mock_param2.numel.return_value = 2000000  # 2M elements
        self.mock_param2.dtype = ms.float32  # 4 bytes per element = ~8MB

        self.mock_param3 = MagicMock()
        self.mock_param3.numel.return_value = 500000  # 0.5M elements
        self.mock_param3.dtype = ms.float32  # 4 bytes per element = ~2MB

        # Create state dict
        self.state_dict = {
            "param1": self.mock_param1,
            "param2": self.mock_param2,
            "param3": self.mock_param3
        }

        # Create a mock config object
        self.mock_config = MagicMock()
        self.mock_config.parallel_config = MagicMock()
        self.mock_config.parallel_config.pipeline_stage = 0
        self.mock_config.pp_interleave_num = 1

        # Create a mock model instance
        self.model = PreTrainedModel.__new__(PreTrainedModel)
        self.model.config = self.mock_config

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.modeling_utils.convert_file_size_to_int')
    @patch('mindformers.models.modeling_utils.dtype_byte_size')
    def test_shard_checkpoint_single_shard(self, mock_dtype_byte_size, mock_convert_file_size_to_int):
        """Test shard_checkpoint with single shard (all weights fit in one shard)"""
        # Mock byte size to be small enough for all weights to fit in one shard
        mock_dtype_byte_size.return_value = 1e-6  # 1 byte per element
        mock_convert_file_size_to_int.return_value = 10000000  # 10MB limit

        shards, index = shard_checkpoint(self.state_dict, "10MB")

        # Should have only one shard
        assert len(shards) == 1
        assert "mindspore_model.ckpt" in shards

        # Index should be None for single shard
        assert index is None

        # All parameters should be in the shard
        shard_content = shards["mindspore_model.ckpt"]
        assert "param1" in shard_content
        assert "param2" in shard_content
        assert "param3" in shard_content

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.modeling_utils.convert_file_size_to_int')
    @patch('mindformers.models.modeling_utils.dtype_byte_size')
    def test_shard_checkpoint_multiple_shards(self, mock_dtype_byte_size, mock_convert_file_size_to_int):
        """Test shard_checkpoint with multiple shards"""
        # Mock byte size to create multiple shards
        mock_dtype_byte_size.return_value = 1  # 1 byte per element
        mock_convert_file_size_to_int.return_value = 1500000  # 1.5MB limit

        shards, index = shard_checkpoint(self.state_dict, "1.5MB")

        # Should have multiple shards
        assert len(shards) > 1

        # Index should not be None
        assert index is not None
        assert "metadata" in index
        assert "weight_map" in index

        # Check that all parameters are in the weight map
        weight_map = index["weight_map"]
        assert "param1" in weight_map
        assert "param2" in weight_map
        assert "param3" in weight_map

        # Check that parameters are distributed across shards
        shard_files = list(shards.keys())
        assert len(shard_files) > 1

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.modeling_utils.convert_file_size_to_int')
    @patch('mindformers.models.modeling_utils.dtype_byte_size')
    def test_shard_checkpoint_large_weight(self, mock_dtype_byte_size, mock_convert_file_size_to_int):
        """Test shard_checkpoint with a weight larger than max_shard_size"""
        # Mock a very large parameter
        mock_large_param = MagicMock()
        mock_large_param.numel.return_value = 10000000  # 10M elements
        mock_large_param.dtype = ms.float32  # 4 bytes per element = ~40MB

        large_state_dict = {
            "small_param": self.mock_param1,
            "large_param": mock_large_param
        }

        # Set limit to be smaller than the large parameter
        mock_dtype_byte_size.return_value = 1  # 1 byte per element
        mock_convert_file_size_to_int.return_value = 5000000  # 5MB limit

        shards, _ = shard_checkpoint(large_state_dict, "5MB")

        # Should have multiple shards since large param exceeds limit
        assert len(shards) > 1

        # Large parameter should be in its own shard
        large_param_shard = None
        for _, shard_content in shards.items():
            if "large_param" in shard_content:
                large_param_shard = shard_content
                break

        assert large_param_shard is not None
        assert "large_param" in large_param_shard

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_add_variant_no_variant(self):
        """Test _add_variant function with no variant"""
        weights_name = "mindspore_model.ckpt"
        result = _add_variant(weights_name, None)
        assert result == weights_name

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_add_variant_with_variant(self):
        """Test _add_variant function with variant"""
        weights_name = "mindspore_model.ckpt"
        variant = "fp16"
        expected = "mindspore_model.fp16.ckpt"
        result = _add_variant(weights_name, variant)
        assert result == expected

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_add_variant_complex_name(self):
        """Test _add_variant function with complex file name"""
        weights_name = "model.custom.extension.ckpt"
        variant = "quantized"
        expected = "model.custom.extension.quantized.ckpt"
        result = _add_variant(weights_name, variant)
        assert result == expected

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.modeling_utils.ms.get_auto_parallel_context')
    def test_not_semi_auto_parallel_mode(self, mock_get_auto_parallel_context):
        """Test when parallel mode is not semi_auto_parallel, should not raise any exception"""
        # Set parallel mode to standalone
        mock_get_auto_parallel_context.return_value = "stand_alone"
        self.mock_config.parallel_config.pipeline_stage = 2  # pp > 1

        # Should not raise any exception
        try:
            self.model.check_pipeline_stage()
        except Exception as e:
            pytest.fail(f"check_pipeline_stage() raised an exception unexpectedly: {e}")

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.modeling_utils.ms.get_auto_parallel_context')
    def test_pipeline_stage_less_than_2(self, mock_get_auto_parallel_context):
        """Test when pipeline_stage <= 1, should not raise any exception"""
        # Set parallel mode to semi_auto_parallel
        mock_get_auto_parallel_context.return_value = "semi_auto_parallel"
        self.mock_config.parallel_config.pipeline_stage = 1  # pp <= 1

        # Should not raise any exception
        try:
            self.model.check_pipeline_stage()
        except Exception as e:
            pytest.fail(f"check_pipeline_stage() raised an exception unexpectedly: {e}")

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.modeling_utils.ms.get_auto_parallel_context')
    def test_missing_num_layers_attribute(self, mock_get_auto_parallel_context):
        """Test when pipeline_stage > 1 but num_layers is not found, should raise ValueError"""
        # Set parallel mode to semi_auto_parallel and pipeline_stage > 1
        mock_get_auto_parallel_context.return_value = "semi_auto_parallel"
        self.mock_config.parallel_config.pipeline_stage = 2

        # Remove num_layers and num_hidden_layers attributes
        if hasattr(self.mock_config, 'num_layers'):
            delattr(self.mock_config, 'num_layers')
        if hasattr(self.mock_config, 'num_hidden_layers'):
            delattr(self.mock_config, 'num_hidden_layers')

        # Should raise ValueError
        with pytest.raises(ValueError) as context:
            self.model.check_pipeline_stage()

        assert "is not found when pipeline_stage > 1" in str(context.value)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.modeling_utils.ms.get_auto_parallel_context')
    def test_num_layers_less_than_pipeline_stage(self, mock_get_auto_parallel_context):
        """Test when num_layers < pipeline_stage, should raise ValueError"""
        # Set parallel mode to semi_auto_parallel and pipeline_stage > 1
        mock_get_auto_parallel_context.return_value = "semi_auto_parallel"
        self.mock_config.parallel_config.pipeline_stage = 4  # pp = 4
        self.mock_config.num_layers = 3  # num_layers < pp

        # Should raise ValueError
        with pytest.raises(ValueError) as context:
            self.model.check_pipeline_stage()

        assert "num_layers (3) < pp(4)" in str(context.value)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.modeling_utils.ms.get_auto_parallel_context')
    def test_valid_num_layers_and_pipeline_stage(self, mock_get_auto_parallel_context):
        """Test when num_layers >= pipeline_stage, should not raise any exception"""
        # Set parallel mode to semi_auto_parallel and pipeline_stage > 1
        mock_get_auto_parallel_context.return_value = "semi_auto_parallel"
        self.mock_config.parallel_config.pipeline_stage = 3  # pp = 3
        self.mock_config.num_layers = 6  # num_layers > pp

        # Should not raise any exception
        try:
            self.model.check_pipeline_stage()
        except Exception as e:
            pytest.fail(f"check_pipeline_stage() raised an exception unexpectedly: {e}")

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.modeling_utils.ms.get_auto_parallel_context')
    def test_pipeline_interleave_valid_case(self, mock_get_auto_parallel_context):
        """Test when pipeline interleave is enabled and valid configuration, should not raise any exception"""
        # Set parallel mode to semi_auto_parallel and pipeline_stage > 1
        mock_get_auto_parallel_context.side_effect = [
            "semi_auto_parallel",  # First call for parallel_mode
            True  # Second call for pipeline_interleave
        ]
        self.mock_config.parallel_config.pipeline_stage = 2  # pp = 2
        self.mock_config.num_layers = 6  # num_layers = 6
        self.mock_config.pp_interleave_num = 2  # pp_interleave_num = 2
        # pp * pp_interleave_num = 4, which is < num_layers (6) - should be valid

        # Should not raise any exception
        try:
            self.model.check_pipeline_stage()
        except Exception as e:
            pytest.fail(f"check_pipeline_stage() raised an exception unexpectedly: {e}")

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.modeling_utils.ms.get_auto_parallel_context')
    def test_pipeline_interleave_invalid_case(self, mock_get_auto_parallel_context):
        """Test when pipeline interleave is enabled but pp * pp_interleave_num > num_layers, should raise ValueError"""
        # Set parallel mode to semi_auto_parallel and pipeline_stage > 1
        mock_get_auto_parallel_context.side_effect = [
            "semi_auto_parallel",  # First call for parallel_mode
            True  # Second call for pipeline_interleave
        ]
        self.mock_config.parallel_config.pipeline_stage = 3  # pp = 3
        self.mock_config.num_layers = 5  # num_layers = 5
        self.mock_config.pp_interleave_num = 2  # pp_interleave_num = 2
        # pp * pp_interleave_num = 6, which is > num_layers (5) - should be invalid

        # Should raise ValueError
        with pytest.raises(ValueError) as context:
            self.model.check_pipeline_stage()

        assert "num_layers : 5 and pp * pp_interleave_num = 6" in str(context.value)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.modeling_utils.ms.get_auto_parallel_context')
    def test_num_hidden_layers_fallback(self, mock_get_auto_parallel_context):
        """Test when num_layers is not present but num_hidden_layers is used as fallback"""
        # Set parallel mode to semi_auto_parallel and pipeline_stage > 1
        mock_get_auto_parallel_context.return_value = "semi_auto_parallel"
        self.mock_config.parallel_config.pipeline_stage = 3  # pp = 3

        # Remove num_layers but keep num_hidden_layers
        if hasattr(self.mock_config, 'num_layers'):
            delattr(self.mock_config, 'num_layers')
        self.mock_config.num_hidden_layers = 2  # num_hidden_layers < pp - should be invalid

        # Should raise ValueError
        with pytest.raises(ValueError) as context:
            self.model.check_pipeline_stage()

        assert "num_layers (2) < pp(3)" in str(context.value)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch("mindformers.tools.PushToHubMixin._get_files_timestamps")
    @patch("mindformers.tools.PushToHubMixin._create_repo")
    @patch("mindformers.tools.PushToHubMixin._upload_modified_files")
    def test_save_pretrained_in_json(self, mock_get_files_timestamps,
                                     mock_create_repo,
                                     mock_upload_modified_files):
        """Test save pretrained model"""
        mock_get_files_timestamps.return_value = {"test": "test"}
        mock_create_repo.return_value = "test"
        mock_upload_modified_files.return_value = "test"

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("mindformers.models.modeling_utils.shard_checkpoint") as mock_shard_checkpoint:
                mock_shard_checkpoint.return_value = {"test": {"test": "test"}}, None
                self.model.save_pretrained(save_directory=temp_dir, save_json=True,
                                           token="test", state_dict={"test": "test"}, push_to_hub=True)


# pylint: disable=W0212
class TestLoadShardedCheckpoint:
    """Test cases for load_sharded_checkpoint function"""

    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()

        # Create mock model
        self.mock_model = MagicMock()

        # Create sample index file content
        self.index_content = {
            "metadata": {"total_size": 1000000},
            "weight_map": {
                "param1": "mindspore_model-00001-of-00002.ckpt",
                "param2": "mindspore_model-00001-of-00002.ckpt",
                "param3": "mindspore_model-00002-of-00002.ckpt"
            }
        }

        # Create index file
        self.index_file = os.path.join(self.temp_dir, "mindspore_model.ckpt.index.json")
        with open(self.index_file, 'w', encoding="utf-8") as f:
            json.dump(self.index_content, f)

        # Create mock checkpoint files
        self.shard1_file = os.path.join(self.temp_dir, "mindspore_model-00001-of-00002.ckpt")
        self.shard2_file = os.path.join(self.temp_dir, "mindspore_model-00002-of-00002.ckpt")

        # Create empty files for shards
        with open(self.shard1_file, 'w', encoding="utf-8") as f:
            f.write("")
        with open(self.shard2_file, 'w', encoding="utf-8") as f:
            f.write("")

    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.modeling_utils.load_checkpoint')
    @patch('mindformers.models.modeling_utils.load_param_into_net')
    def test_load_sharded_checkpoint(self, mock_load_param_into_net, mock_load_checkpoint):
        """Test load_sharded_checkpoint function"""

        # Mock load_checkpoint to return different state dicts for different files
        def mock_load_checkpoint_side_effect(file_path):
            if "00001" in file_path:
                return {"param1": "value1", "param2": "value2"}
            if "00002" in file_path:
                return {"param3": "value3"}
            return {}

        mock_load_checkpoint.side_effect = mock_load_checkpoint_side_effect

        # Mock load_param_into_net to return empty lists (no missing/unexpected keys)
        mock_load_param_into_net.return_value = ([], [])

        # Call the function
        result = load_sharded_checkpoint(self.mock_model, self.temp_dir)

        # Verify load_checkpoint was called for each shard
        mock_load_checkpoint.assert_any_call(self.shard1_file)
        mock_load_checkpoint.assert_any_call(self.shard2_file)

        # Verify load_param_into_net was called with combined state dict
        mock_load_param_into_net.assert_called_once()

        # Result should be the return value from load_param_into_net
        assert result == ([], [])

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.modeling_utils.load_checkpoint')
    @patch('mindformers.models.modeling_utils.load_param_into_net')
    def test_load_sharded_checkpoint_strict_false(self, mock_load_param_into_net, mock_load_checkpoint):
        """Test load_sharded_checkpoint function with strict=False"""
        # Mock load_checkpoint
        mock_load_checkpoint.return_value = {"param1": "value1"}

        # Mock load_param_into_net
        mock_load_param_into_net.return_value = (["missing_key"], ["unexpected_key"])

        # Call the function with strict=False
        result = load_sharded_checkpoint(self.mock_model, self.temp_dir, strict=False)

        # Verify load_param_into_net was called with strict_load=False
        mock_load_param_into_net.assert_called_once()
        call_args = mock_load_param_into_net.call_args
        assert call_args[1]['strict_load'] is False

        # Check result
        assert result == (["missing_key"], ["unexpected_key"])

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.modeling_utils.os.path.join')
    def test_load_sharded_checkpoint_invalid_folder(self, mock_join):
        """Test load_sharded_checkpoint with invalid folder"""
        # Mock join to return a path that doesn't exist
        mock_join.return_value = "/non/existent/path/index.json"

        # Should raise FileNotFoundError when trying to open non-existent index file
        with pytest.raises(FileNotFoundError):
            load_sharded_checkpoint(self.mock_model, "/non/existent/path")

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_dtype_byte_size_bool(self):
        """Test dtype_byte_size function with boolean type"""
        # Test boolean type which returns 1/8 byte
        result = dtype_byte_size(ms.bool_)
        assert result == 1/8

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_dtype_byte_size_standard_types(self):
        """Test dtype_byte_size function with standard numeric types"""
        # Test float32 type (32 bits = 4 bytes)
        result = dtype_byte_size(ms.float32)
        assert result == 4

        # Test int32 type (32 bits = 4 bytes)
        result = dtype_byte_size(ms.int32)
        assert result == 4

        # Test float16 type (16 bits = 2 bytes)
        result = dtype_byte_size(ms.float16)
        assert result == 2

        # Test int8 type (8 bits = 1 byte)
        result = dtype_byte_size(ms.int8)
        assert result == 1

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_dtype_byte_size_invalid_type(self):
        """Test dtype_byte_size function with invalid dtype"""
        # Create an invalid dtype string
        invalid_dtype = "invalid_dtype_123"

        # Patch ms.bool_ to test the invalid case
        with patch('mindformers.models.modeling_utils.re.search') as mock_search:
            mock_search.return_value = None
            with pytest.raises(ValueError) as context:
                dtype_byte_size(invalid_dtype)
            assert "is not a valid dtype" in str(context.value)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_dtype_byte_size_edge_cases(self):
        """Test dtype_byte_size function edge cases"""
        # Test with 64-bit type
        with patch('mindformers.models.modeling_utils.re.search') as mock_search:
            # Mock the regex search to return 64 bits
            mock_match = MagicMock()
            mock_match.groups.return_value = ['64']
            mock_search.return_value = mock_match

            # Test with a 64-bit type
            result = dtype_byte_size(ms.float64)
            assert result == 8  # 64 bits / 8 = 8 bytes

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.modeling_utils.ms.save_checkpoint')
    def test_save_checkpoint(self, mock_save_checkpoint):
        """Test save_checkpoint function"""
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock save object
            mock_save_obj = MagicMock()

            # Define checkpoint file name
            ckpt_file_name = os.path.join(temp_dir, "mindspore_model.ckpt")

            # Call the function
            save_checkpoint(mock_save_obj, temp_dir)

            # Verify save_checkpoint was called with correct arguments
            mock_save_checkpoint.assert_called_once_with(mock_save_obj, ckpt_file_name)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.modeling_utils.ms.save_checkpoint')
    def test_save_checkpoint_with_custom_weights_name(self, mock_save_checkpoint):
        """Test save_checkpoint function with custom weights name"""
        # Patch WEIGHTS_NAME to test custom name
        with patch('mindformers.models.modeling_utils.WEIGHTS_NAME', 'custom_model.ckpt'):
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create a mock save object
                mock_save_obj = MagicMock()

                # Define expected checkpoint file name
                ckpt_file_name = os.path.join(temp_dir, "custom_model.ckpt")

                # Call the function
                save_checkpoint(mock_save_obj, temp_dir)

                # Verify save_checkpoint was called with correct arguments
                mock_save_checkpoint.assert_called_once_with(mock_save_obj, ckpt_file_name)


# pylint: disable=W0212
class TestPreTrainedModelMethods:
    """Test cases for PreTrainedModel methods"""

    def setup_method(self):
        """Set up test environment"""
        # Create a mock config
        self.mock_config = MagicMock()
        self.mock_config.name_or_path = "test_model"

        # Create a mock model instance
        self.model = PreTrainedModel.__new__(PreTrainedModel)
        self.model.config = self.mock_config
        self.model.base_model_prefix = ""
        self.model._keys_to_ignore_on_save = None
        self.model._auto_class = None

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_can_generate_default_behavior(self):
        """Test can_generate method with default behavior"""
        # By default, model should be able to generate if it inherits from GenerationMixin
        result = PreTrainedModel.can_generate()
        # Since we're calling directly on PreTrainedModel, it should return True
        # because the methods are not from GeneratorMixin
        assert result is True

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_save_pretrained_origin_mode_default_values(self):
        """Test save_pretrained_origin_mode with default values"""
        with tempfile.TemporaryDirectory() as temp_dir, \
                patch.multiple('mindformers.models.modeling_utils',
                               DEFAULT_CHECKPOINT_SAVE_FOLDER=temp_dir,
                               ms=DEFAULT,
                               yaml=DEFAULT) as mocks:
            # Mock required methods
            self.model._inverse_parse_config = MagicMock(return_value=(self.mock_config, []))
            self.model._wrap_config = MagicMock(return_value={"model": {}})
            self.model.remove_type = MagicMock()
            mocks['yaml'].dump.return_value = "yaml_dump_result"

            # Call the method
            self.model.save_pretrained_origin_mode()

            # Verify yaml.dump was called
            mocks['yaml'].dump.assert_called_once()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_save_pretrained_origin_mode_custom_directory(self):
        """Test save_pretrained_origin_mode with custom directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_dir = os.path.join(temp_dir, "custom_save_dir")

            with patch.multiple('mindformers.models.modeling_utils',
                                ms=DEFAULT,
                                yaml=DEFAULT) as mocks:
                # Mock required methods
                self.model._inverse_parse_config = MagicMock(return_value=(self.mock_config, []))
                self.model._wrap_config = MagicMock(return_value={"model": {}})
                self.model.remove_type = MagicMock()
                mocks['yaml'].dump.return_value = "yaml_dump_result"

                # Call the method
                self.model.save_pretrained_origin_mode(save_directory=custom_dir)

                # Verify directory was created
                assert os.path.exists(custom_dir)

                # Verify yaml.dump was called
                mocks['yaml'].dump.assert_called_once()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_save_pretrained_origin_mode_invalid_types(self):
        """Test save_pretrained_origin_mode with invalid parameter types"""
        with pytest.raises(TypeError) as context:
            self.model.save_pretrained_origin_mode(save_directory=123, save_name="test")
        assert "save_directory and save_name should be a str" in str(context.value)

        with pytest.raises(TypeError) as context:
            self.model.save_pretrained_origin_mode(save_directory="/tmp", save_name=123)
        assert "save_directory and save_name should be a str" in str(context.value)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_save_pretrained_origin_mode_no_config(self):
        """Test save_pretrained_origin_mode when model has no config"""
        self.model.config = None
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(AttributeError) as context:
                self.model.save_pretrained_origin_mode(save_directory=temp_dir)
            assert "has no attribute" in str(context.value)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_save_pretrained_experimental_mode_with_file_path(self):
        """Test save_pretrained_experimental_mode with file path instead of directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "not_a_dir.txt")
            # Create a file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("test")

            with patch('mindformers.models.modeling_utils.logger') as mock_logger:
                self.model.save_pretrained_experimental_mode(save_directory=file_path)
                # Should log an error
                mock_logger.error.assert_called_once()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_remove_type_method(self):
        """Test remove_type method"""
        # Test with PretrainedConfig
        mock_config = PretrainedConfig()
        mock_config.__dict__ = {"type": "test_type", "other_attr": "value"}

        self.model.remove_type(mock_config)

        # Type should be removed
        assert "type" not in mock_config.__dict__

        # Other attributes should remain
        assert "other_attr" in mock_config.__dict__

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_inverse_parse_config_method(self):
        """Test _inverse_parse_config method"""

        # Create a config with various types of attributes
        config = PretrainedConfig()
        config.test_attr = "test_value"
        config.test_int = 42
        config.test_float = 3.14
        config.test_bool = True

        # Call the method
        result_config, _ = self.model._inverse_parse_config(config)

        # Check that type was added
        assert "type" in result_config.__dict__
        assert result_config.__dict__["type"] == "PretrainedConfig"

        # Check that basic types are preserved
        assert "test_attr" in result_config.__dict__
        assert "test_int" in result_config.__dict__
        assert "test_float" in result_config.__dict__
        assert "test_bool" in result_config.__dict__

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_wrap_config_method(self):
        """Test _wrap_config method"""

        # Create a config
        config = PretrainedConfig()
        config.test_attr = "test_value"

        # Mock to_dict method
        config.to_dict = MagicMock(return_value={"test_attr": "test_value"})

        # Call the method
        result = self.model._wrap_config(config)

        # Check the structure
        assert "model" in result
        assert "model_config" in result["model"]
        assert "arch" in result["model"]
        assert "type" in result["model"]["arch"]
        assert result["model"]["arch"]["type"] == self.model.__class__.__name__


# pylint: disable=W0212
class TestPreTrainedModelLoading:
    """Test cases for PreTrainedModel loading methods"""

    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_class = PreTrainedModel

        # Mock config
        self.mock_config = MagicMock()
        self.mock_config.name_or_path = "test_model"

        # Mock model
        self.mock_model = MagicMock()
        self.mock_model.config = self.mock_config

    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.modeling_utils.os.path.exists')
    def test_get_config_args_nonexistent_model(self, mock_exists):
        """Test _get_config_args with nonexistent model"""
        mock_exists.return_value = False
        self.model_class._support_list = ['supported_model']

        with pytest.raises(ValueError) as context:
            self.model_class._get_config_args('unsupported_model')

        assert "does not exist" in str(context.value)
        assert "not supported" in str(context.value)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.modeling_utils.os.path.exists')
    @patch('mindformers.models.modeling_utils.os.path.isdir')
    def test_get_config_args_file_instead_of_directory(self, mock_isdir, mock_exists):
        """Test _get_config_args with file path instead of directory"""
        mock_exists.return_value = True
        mock_isdir.return_value = False

        with pytest.raises(ValueError) as context:
            self.model_class._get_config_args('/path/to/file')

        assert "is not a directory" in str(context.value)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.modeling_utils.os.path.exists')
    @patch('mindformers.models.modeling_utils.os.path.isdir')
    @patch('mindformers.models.modeling_utils.os.listdir')
    def test_get_config_args_missing_files(self, mock_listdir, mock_isdir, mock_exists):
        """Test _get_config_args with missing yaml or ckpt files"""
        mock_exists.return_value = True
        mock_isdir.return_value = True
        mock_listdir.return_value = ['some_other_file.txt']  # No yaml or ckpt files

        with pytest.raises(FileNotFoundError) as context:
            self.model_class._get_config_args('/path/to/model_dir')

        assert "no yaml file for model config" in str(context.value)

    @patch('mindformers.models.modeling_utils.os.path.exists')
    @patch('mindformers.models.modeling_utils.os.path.isdir')
    @patch('mindformers.models.modeling_utils.os.listdir')
    @patch('mindformers.models.modeling_utils.MindFormerConfig')
    def test_get_config_args_local_directory(self, mock_config, mock_listdir, mock_isdir, mock_exists):
        """Test _get_config_args with local directory containing model files"""
        mock_exists.return_value = True
        mock_isdir.return_value = True
        mock_listdir.return_value = ['config.yaml', 'model.ckpt']
        mock_config.return_value = MagicMock()

        # Mock model type indices
        self.model_class._model_type = 0

        self.model_class._get_config_args('/path/to/model_dir')

        # Verify MindFormerConfig was called with correct yaml file
        mock_config.assert_called_once()

    @patch('mindformers.models.modeling_utils.os.path.exists')
    @patch('mindformers.models.modeling_utils.os.path.isdir')
    @patch('mindformers.models.modeling_utils.MindFormerConfig')
    def test_get_config_args_not_is_dir(self, mock_config, mock_isdir, mock_exists):
        """Test _get_config_args with local directory containing model files"""
        mock_exists.return_value = False
        mock_isdir.return_value = False
        mock_config.return_value = MagicMock()

        # Mock model type indices
        self.model_class._model_type = 0

        self.model_class._support_list = ['common']
        with pytest.raises(FileNotFoundError) as context:
            self.model_class._get_config_args('common')
        assert "default yaml file path must be correct" in str(context.value)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_is_experimental_mode_with_file(self):
        """Test is_experimental_mode with file path instead of directory"""
        with patch.multiple('mindformers.models.modeling_utils.os.path',
                            exists=MagicMock(return_value=True),
                            isdir=MagicMock(return_value=False)):
            result = self.model_class.is_experimental_mode('/path/to/file')
            assert result is False

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.modeling_utils.os.path.exists')
    @patch('mindformers.models.modeling_utils.os.path.isdir')
    @patch('mindformers.models.modeling_utils.os.listdir')
    def test_is_experimental_mode_with_config_json(self, mock_listdir, mock_isdir, mock_exists):
        """Test is_experimental_mode with config.json file present"""
        mock_exists.return_value = True
        mock_isdir.return_value = True
        mock_listdir.return_value = ['config.json']  # config.json but no yaml files

        result = self.model_class.is_experimental_mode('/path/to/model_dir')
        assert result is True

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.modeling_utils.os.path.exists')
    @patch('mindformers.models.modeling_utils.os.path.isdir')
    @patch('mindformers.models.modeling_utils.os.listdir')
    def test_is_experimental_mode_with_yaml_files(self, mock_listdir, mock_isdir, mock_exists):
        """Test is_experimental_mode with yaml files present"""
        mock_exists.return_value = True
        mock_isdir.return_value = True
        mock_listdir.return_value = ['config.yaml', 'model.ckpt']

        result = self.model_class.is_experimental_mode('/path/to/model_dir')
        assert result is False

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_is_experimental_mode_huggingface_style(self):
        """Test is_experimental_mode with HuggingFace-style model path"""
        result = self.model_class.is_experimental_mode('bert-base-uncased')
        assert result is False

        result = self.model_class.is_experimental_mode('mindspore/bert-base')
        assert result is False

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_from_pretrained_invalid_type(self):
        """Test from_pretrained with invalid pretrained_model_name_or_dir type"""
        with pytest.raises(TypeError) as context:
            self.model_class.from_pretrained(123)

        assert "pretrained_model_name_or_dir should be a str" in str(context.value)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.modeling_utils.PreTrainedModel.is_experimental_mode')
    @patch('mindformers.models.modeling_utils.PreTrainedModel.from_pretrained_experimental_mode')
    def test_from_pretrained_experimental_mode(self, mock_experimental, mock_is_experimental):
        """Test from_pretrained routes to experimental mode"""
        mock_is_experimental.return_value = True
        mock_experimental.return_value = self.mock_model

        result = self.model_class.from_pretrained('test_model')

        mock_experimental.assert_called_once()
        assert result == self.mock_model

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.modeling_utils.PreTrainedModel.is_experimental_mode')
    @patch('mindformers.models.modeling_utils.PreTrainedModel.from_pretrained_origin_mode')
    def test_from_pretrained_origin_mode(self, mock_origin, mock_is_experimental):
        """Test from_pretrained routes to origin mode"""
        mock_is_experimental.return_value = False
        mock_origin.return_value = self.mock_model

        result = self.model_class.from_pretrained('test_model')

        mock_origin.assert_called_once()
        assert result == self.mock_model

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.modeling_utils.PreTrainedModel._get_config_args')
    @patch('mindformers.models.modeling_utils.build_network')
    def test_from_pretrained_origin_mode_success(self, mock_build_network, mock_get_config_args):
        """Test from_pretrained_origin_mode success case"""
        mock_config_args = MagicMock()
        mock_config_args.model = MagicMock()
        mock_config_args.model.model_config = MagicMock()
        mock_config_args.get.return_value = None
        mock_get_config_args.return_value = mock_config_args
        mock_build_network.return_value = self.mock_model

        result = self.model_class.from_pretrained_origin_mode('test_model')

        mock_get_config_args.assert_called_once()
        mock_build_network.assert_called_once()
        assert result == self.mock_model

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_from_pretrained_origin_mode_invalid_type(self):
        """Test from_pretrained_origin_mode with invalid type"""
        with pytest.raises(TypeError) as context:
            self.model_class.from_pretrained_origin_mode(123)

        assert "pretrained_model_name_or_dir should be a str" in str(context.value)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.modeling_utils.load_param_into_net')
    def test_load_pretrained_model_success(self, mock_load_param):
        """Test _load_pretrained_model success case"""
        mock_model = MagicMock()
        mock_model.get_parameters.return_value = []
        mock_model.base_model_prefix = ""
        mock_model.config = MagicMock()
        mock_model.config.architectures = None

        mock_load_param.return_value = ([], [])  # missing_keys, unexpected_keys

        result_model, missing_keys, unexpected_keys, mismatched_keys = self.model_class._load_pretrained_model(
            mock_model,
            {},  # state_dict
            None,  # resolved_archive_file
            "test_model"
        )

        assert result_model == mock_model
        assert not missing_keys
        assert not unexpected_keys
        assert not mismatched_keys

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.modeling_utils.load_param_into_net')
    def test_load_pretrained_model_state_dict_none(self, mock_load_param):
        """Test load_pretrained_model_state_dict_none_case"""
        mock_model = MagicMock()
        mock_model.get_parameters.return_value = []
        mock_model.base_model_prefix = ""
        mock_model.config = MagicMock()
        mock_model.config.architectures = None

        mock_load_param.return_value = ([], [])  # missing_keys, unexpected_keys

        with pytest.raises(ValueError) as context:
            self.model_class._load_pretrained_model(
                mock_model,
                None,  # state_dict
                None,  # resolved_archive_file
                "test_model"
            )
        assert "should be str, list or tuple" in str(context.value)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.modeling_utils.load_param_into_net')
    def test_load_pretrained_model_resolved_archive_file_list(self, mock_load_param):
        """Test load_pretrained_model_resolved_archive_file_list_case"""
        mock_model = MagicMock()
        mock_model.get_parameters.return_value = []
        mock_model.base_model_prefix = ""
        mock_model.config = MagicMock()
        mock_model.config.architectures = None

        mock_load_param.return_value = ([], [])  # missing_keys, unexpected_keys

        with pytest.raises(ValueError) as context:
            self.model_class._load_pretrained_model(
                mock_model,
                None,  # state_dict
                ["test"],  # resolved_archive_file
                "test_model"
            )
        assert "resolved_archive_file_:test not found!" in str(context.value)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.modeling_utils.load_param_into_net')
    def test_load_pretrained_model_resolved_archive_file_str(self, mock_load_param):
        """Test load_pretrained_model_resolved_archive_file_str_case"""
        mock_model = MagicMock()
        mock_model.get_parameters.return_value = []
        mock_model.base_model_prefix = ""
        mock_model.config = MagicMock()
        mock_model.config.architectures = None

        mock_load_param.return_value = ([], [])  # missing_keys, unexpected_keys

        with pytest.raises(ValueError) as context:
            self.model_class._load_pretrained_model(
                mock_model,
                None,  # state_dict
                "test",  # resolved_archive_file
                "test_model"
            )
        assert "resolved_archive_file:test not found!" in str(context.value)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.modeling_utils.os.path.exists')
    def test_get_src_checkpoint_with_string_path(self, mock_exists):
        """Test _get_src_checkpoint with string path"""
        mock_exists.return_value = True

        with patch('mindformers.models.modeling_utils.make_soft_link') as mock_link:
            result = self.model_class._get_src_checkpoint(
                state_dict=None,
                resolved_archive_file='/path/to/checkpoint.ckpt',
                src_checkpoint='/path/to/src_checkpoint.ckpt'
            )

            mock_link.assert_called_once()
            assert result == '/path/to/src_checkpoint.ckpt'

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.modeling_utils.os.path.exists')
    def test_get_src_checkpoint_with_list_path(self, mock_exists):
        """Test _get_src_checkpoint with list path"""
        mock_exists.return_value = True

        with pytest.raises(ValueError) as context:
            self.model_class._get_src_checkpoint(
                state_dict=None,
                resolved_archive_file=['/path/to/checkpoint.ckpt'],
                src_checkpoint='/path/to/src_checkpoint.ckpt'
            )

        assert "Failed to read the checkpoint file /path/to/checkpoint.ckpt" in str(context.value)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_register_for_auto_class_invalid_class(self):
        """Test register_for_auto_class with invalid auto class"""
        with patch('mindformers.models.modeling_utils.auto_module') as mock_auto:
            mock_auto.InvalidClass = None
            self.model_class.register_for_auto_class('InvalidClass')


# pylint: disable=W0212
class TestFromPretrainedExperimentalMode:
    """Test cases for from_pretrained_experimental_mode method"""

    def setup_method(self):
        """Set up test environment"""

        # Create a mock model class
        class MockModel(PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.config = config

        self.mock_model_class = MockModel

        # Create a mock config
        self.mock_config = MagicMock(spec=PretrainedConfig)
        self.mock_config.name_or_path = "test_model"
        self.mock_config.__class__ = PretrainedConfig

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.modeling_utils.cached_file')
    def test_commit_hash_extraction_from_config(self, mock_cached_file):
        """Test commit hash extraction when config doesn't have _commit_hash"""
        mock_cached_file.return_value = "/path/to/config.json"

        with patch.multiple('mindformers.models.modeling_utils',
                            GenerationConfig=DEFAULT,
                            extract_commit_hash=DEFAULT) as mocks:
            mocks['extract_commit_hash'].return_value = "abc123"

            with patch.multiple(self.mock_model_class,
                                config_class=DEFAULT,
                                _load_pretrained_model=DEFAULT) as model_mocks:
                model_mocks['config_class'].from_pretrained.return_value = (self.mock_config, {})
                model_mocks['_load_pretrained_model'].return_value = (MagicMock(), [], [], [])

                # Call the method with config that doesn't have _commit_hash
                self.mock_model_class.from_pretrained_experimental_mode("test_model")

                # Verify extract_commit_hash was called
                mocks['extract_commit_hash'].assert_called_once()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.modeling_utils.os.path.isdir')
    @patch('mindformers.models.modeling_utils.os.path.isfile')
    def test_local_directory_with_weights_file(self, mock_isfile, mock_isdir):
        """Test handling of local directory with weights file"""
        mock_isdir.return_value = True
        mock_isfile.return_value = True

        with patch.multiple('mindformers.models.modeling_utils',
                            GenerationConfig=DEFAULT,
                            logger=DEFAULT) as mocks:
            with patch.multiple(self.mock_model_class,
                                config_class=DEFAULT,
                                _load_pretrained_model=DEFAULT) as model_mocks:
                model_mocks['config_class'].from_pretrained.return_value = (self.mock_config, {})
                model_mocks['_load_pretrained_model'].return_value = (MagicMock(), [], [], [])

                # Call the method with local directory path
                self.mock_model_class.from_pretrained_experimental_mode("/local/model/path")

                # Verify logger was called with loading info
                mocks['logger'].info.assert_called()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.models.modeling_utils.os.path.isdir')
    @patch('mindformers.models.modeling_utils.os.path.isfile')
    @patch('mindformers.models.modeling_utils.cached_file')
    def test_missing_weights_file_error(self, mock_cached_file, mock_isfile, mock_isdir):
        """Test error handling when weights file is missing"""
        mock_isdir.return_value = True
        mock_isfile.return_value = False  # No weights file found
        mock_cached_file.return_value = None  # No cached file either

        with patch.object(self.mock_model_class, 'config_class') as mock_config_class:
            mock_config_class.from_pretrained.return_value = (self.mock_config, {})

            with patch('mindformers.models.modeling_utils.has_file') as mock_has_file:
                mock_has_file.return_value = False

                # Should raise EnvironmentError when no weights file is found
                with pytest.raises(EnvironmentError):
                    self.mock_model_class.from_pretrained_experimental_mode("/local/model/path")
