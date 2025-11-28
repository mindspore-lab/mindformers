# Copyright 2024 Huawei Technologies Co., Ltd
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
"""test class ModelRunner."""
from unittest.mock import MagicMock, Mock, patch
from unittest import mock

import os
import sys
import json
import tempfile
import shutil
import pytest
import numpy as np

import mindspore as ms
from mindspore import Tensor

from mindformers.model_runner import (
    register_auto_class,
    get_model,
    ModelRunner,
    MindIEModelRunner,
    InputBuilder,
    _get_model_config,
    _load_distributed_safetensors,
    _load_safetensors,
    _check_valid_safetensors_path
)

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ============================================================================
# Test Utilities and Helpers
# ============================================================================

# pylint: disable=W0613
def _create_mock_config_with_dict_access(auto_map):
    """
    Create a MagicMock config that supports both attribute and dictionary access.

    This is needed because register_auto_class uses both:
    - config.model.model_config.auto_map (attribute access)
    - config["model"]["model_config"]["auto_map"] (dictionary access)

    Args:
        auto_map (dict): The auto_map dictionary
    
    Returns:
        MagicMock: A config mock that supports both access patterns
    """
    # Create model_config level mock
    model_config_mock = MagicMock()
    model_config_mock.auto_map = auto_map

    # Use lambda with *args to handle any number of arguments
    # MagicMock may pass self as first argument, so we take the last argument as key
    def model_config_getitem(*args, **kwargs):
        """__getitem__ accepts variable arguments"""
        # Get the last positional argument as key (handles both obj[key] and obj.__getitem__(key))
        key = args[-1] if args else None
        if key == "auto_map":
            return auto_map
        return MagicMock()
    model_config_mock.__getitem__ = model_config_getitem

    # Create model level mock
    model_mock = MagicMock()
    model_mock.model_config = model_config_mock

    def model_getitem(*args, **kwargs):
        """__getitem__ accepts variable arguments"""
        key = args[-1] if args else None
        if key == "model_config":
            return model_config_mock
        return MagicMock()
    model_mock.__getitem__ = model_getitem

    # Create config level mock
    config = MagicMock()
    config.model = model_mock

    def config_getitem(*args, **kwargs):
        """__getitem__ accepts variable arguments"""
        key = args[-1] if args else None
        if key == "model":
            return model_mock
        return MagicMock()
    config.__getitem__ = config_getitem

    return config


class MockRegistryFactory:
    """Factory for creating mock registries and module types"""

    @staticmethod
    def create_module_type_mock():
        """Create a mock MindFormerModuleType with standard enum values"""
        mock_module_type = MagicMock()
        mock_module_type.CONFIG = 'CONFIG'
        mock_module_type.TOKENIZER = 'TOKENIZER'
        mock_module_type.MODELS = 'MODELS'
        mock_module_type.PROCESSOR = 'PROCESSOR'
        return mock_module_type

    @staticmethod
    def create_empty_registry():
        """Create an empty registry dictionary"""
        return {
            'CONFIG': {},
            'TOKENIZER': {},
            'MODELS': {},
            'PROCESSOR': {}
        }

    @staticmethod
    def create_registry_with_config(config_name='test_config'):
        """Create a registry with a pre-registered config"""
        registry = MockRegistryFactory.create_empty_registry()
        registry['CONFIG'][config_name] = MagicMock()
        return registry


class MockConfigFactory:
    """Factory for creating mock configurations"""

    @staticmethod
    def create_auto_map_config():
        """Create a standard auto_map configuration"""
        return {
            'AutoConfig': 'module.ConfigClass',
            'AutoTokenizer': ['module.TokenizerSlow', 'module.TokenizerFast'],
            'AutoModel': 'module.ModelClass',
            'AutoProcessor': 'module.ProcessorClass'
        }

    @staticmethod
    def create_model_config(
        num_layers=2,
        num_heads=4,
        n_kv_heads=None,
        hidden_size=128,
        compute_dtype='float32',
        batch_size=1,
        seq_length=512,
        is_dynamic=False
    ):
        """Create a mock model configuration"""
        mock_config = MagicMock()
        mock_config.num_layers = num_layers
        mock_config.num_heads = num_heads
        mock_config.n_kv_heads = n_kv_heads
        mock_config.hidden_size = hidden_size
        mock_config.compute_dtype = compute_dtype
        mock_config.batch_size = batch_size
        mock_config.seq_length = seq_length
        mock_config.is_dynamic = is_dynamic

        # For non-legacy mode
        mock_config.num_hidden_layers = num_layers
        mock_config.num_attention_heads = num_heads
        mock_config.num_key_value_heads = n_kv_heads
        return mock_config

    @staticmethod
    def create_mindformer_config(
        arch_type='llama',
        use_parallel=False,
        moe_config=None,
        load_checkpoint=None
    ):
        """Create a mock MindFormerConfig"""
        mock_config = MagicMock()
        mock_config.use_parallel = use_parallel
        mock_config.model.arch.type = arch_type
        mock_config.model.model_config = {'type': arch_type}
        mock_config.moe_config = moe_config
        mock_config.load_checkpoint = load_checkpoint
        mock_config.context = MagicMock()
        mock_config.parallel_config = MagicMock()
        return mock_config


class MockRunnerFactory:
    """Factory for creating mock runners"""

    @staticmethod
    def create_runner_mock(
        num_layers=2,
        warmup_step=2,
        is_multi_modal=False,
        use_legacy=True
    ):
        """Create a mock MindIEModelRunner instance"""
        runner = MagicMock(spec=MindIEModelRunner)
        runner.num_layers = num_layers
        runner.model = MagicMock()
        runner.use_legacy = use_legacy
        runner.warmup_step = warmup_step
        runner.is_multi_modal_model = is_multi_modal
        runner.model_config = MagicMock()
        runner.processor = MagicMock()
        runner.config = MagicMock()
        return runner


class TempDirFixture:
    """Context manager for temporary directories"""

    def __init__(self):
        self.test_dir = None

    def __enter__(self):
        self.test_dir = tempfile.mkdtemp()
        return self.test_dir

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.test_dir and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)


# ============================================================================
# Test Classes
# ============================================================================

class TestRegisterAutoClass:
    """Test register_auto_class function"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures"""
        auto_map = MockConfigFactory.create_auto_map_config()
        self.config = _create_mock_config_with_dict_access(auto_map)
        self.config.model.model_config.type = 'test_config'
        self.config.model.arch.type = 'test_model'
        self.config.processor.tokenizer.type = 'test_tokenizer'
        self.model_path = '/test/path'

    def _setup_mocks(self, mock_get_class, mock_register, mock_module_type, registry=None):
        """Helper method to setup common mocks"""
        mock_module_type.CONFIG = 'CONFIG'
        mock_module_type.TOKENIZER = 'TOKENIZER'
        mock_module_type.MODELS = 'MODELS'
        mock_module_type.PROCESSOR = 'PROCESSOR'

        if registry is None:
            registry = MockRegistryFactory.create_empty_registry()
        mock_register.registry = registry

        if mock_get_class:
            mock_get_class.return_value = MagicMock()

    @patch('mindformers.model_runner.MindFormerModuleType')
    @patch('mindformers.model_runner.MindFormerRegister')
    @patch('mindformers.model_runner.get_class_from_dynamic_module')
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_register_auto_config(self, mock_get_class, mock_register, mock_module_type):
        """Test registering AutoConfig"""
        self._setup_mocks(mock_get_class, mock_register, mock_module_type)

        register_auto_class(self.config, self.model_path, 'AutoConfig')

        mock_get_class.assert_called_once_with('module.ConfigClass', self.model_path)
        mock_register.register_cls.assert_called_once()

    @patch('mindformers.model_runner.MindFormerModuleType')
    @patch('mindformers.model_runner.MindFormerRegister')
    @patch('mindformers.model_runner.get_class_from_dynamic_module')
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_register_auto_tokenizer_fast(self, mock_get_class, mock_register, mock_module_type):
        """Test registering AutoTokenizer with use_fast=True"""
        self._setup_mocks(mock_get_class, mock_register, mock_module_type)

        register_auto_class(self.config, self.model_path, 'AutoTokenizer', use_fast=True)

        mock_get_class.assert_called_once_with('module.TokenizerFast', self.model_path)

    @patch('mindformers.model_runner.MindFormerModuleType')
    @patch('mindformers.model_runner.MindFormerRegister')
    @patch('mindformers.model_runner.get_class_from_dynamic_module')
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_register_auto_tokenizer_slow(self, mock_get_class, mock_register, mock_module_type):
        """Test registering AutoTokenizer with use_fast=False"""
        self._setup_mocks(mock_get_class, mock_register, mock_module_type)

        register_auto_class(self.config, self.model_path, 'AutoTokenizer', use_fast=False)

        mock_get_class.assert_called_once_with('module.TokenizerSlow', self.model_path)

    @patch('mindformers.model_runner.MindFormerModuleType')
    @patch('mindformers.model_runner.MindFormerRegister')
    @patch('mindformers.model_runner.get_class_from_dynamic_module')
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_register_auto_tokenizer_no_fast(self, mock_get_class, mock_register, mock_module_type):
        """Test registering AutoTokenizer when fast tokenizer is None"""
        self.config.model.model_config.auto_map['AutoTokenizer'] = ['module.TokenizerSlow', None]
        self._setup_mocks(mock_get_class, mock_register, mock_module_type)

        register_auto_class(self.config, self.model_path, 'AutoTokenizer', use_fast=True)

        mock_get_class.assert_called_once_with('module.TokenizerSlow', self.model_path)

    @patch('mindformers.model_runner.MindFormerModuleType')
    @patch('mindformers.model_runner.MindFormerRegister')
    @patch('mindformers.model_runner.get_class_from_dynamic_module')
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_register_auto_model(self, mock_get_class, mock_register, mock_module_type):
        """Test registering AutoModel"""
        self._setup_mocks(mock_get_class, mock_register, mock_module_type)

        register_auto_class(self.config, self.model_path, 'AutoModel')

        mock_get_class.assert_called_once_with('module.ModelClass', self.model_path)

    @patch('mindformers.model_runner.MindFormerModuleType')
    @patch('mindformers.model_runner.MindFormerRegister')
    @patch('mindformers.model_runner.get_class_from_dynamic_module')
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_register_auto_processor(self, mock_get_class, mock_register, mock_module_type):
        """Test registering AutoProcessor"""
        self._setup_mocks(mock_get_class, mock_register, mock_module_type)

        register_auto_class(self.config, self.model_path, 'AutoProcessor')

        mock_get_class.assert_called_once_with('module.ProcessorClass', self.model_path)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_register_no_auto_map(self):
        """Test when config has no auto_map"""
        self.config.model.model_config.auto_map = None
        # Should not raise error
        register_auto_class(self.config, self.model_path, 'AutoConfig')

    @patch('mindformers.model_runner.MindFormerModuleType')
    @patch('mindformers.model_runner.MindFormerRegister')
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_register_already_registered(self, mock_register, mock_module_type):
        """Test when class is already registered"""
        registry = MockRegistryFactory.create_registry_with_config('test_config')
        self._setup_mocks(None, mock_register, mock_module_type, registry=registry)

        register_auto_class(self.config, self.model_path, 'AutoConfig')
        mock_register.register_cls.assert_not_called()


# pylint: disable=C2801
class TestGetModel:
    """Test get_model function"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Set up and tear down test fixtures"""
        self.temp_dir = TempDirFixture()
        self.test_dir = self.temp_dir.__enter__()
        self.yaml_file = os.path.join(self.test_dir, 'config.yaml')
        with open(self.yaml_file, 'w', encoding='utf-8') as f:
            f.write('model:\n  arch:\n    type: llama\n  model_config:\n    type: llama_config\n')
        yield
        self.temp_dir.__exit__(None, None, None)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_invalid_path(self):
        """Test with invalid path"""
        with pytest.raises(ValueError) as exc_info:
            get_model('/nonexistent/path')
        assert 'does not exist' in str(exc_info.value)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_not_directory(self):
        """Test with file instead of directory"""
        with pytest.raises(ValueError) as exc_info:
            get_model(self.yaml_file)
        assert 'does not exist or is not a directory' in str(exc_info.value)

    @patch('mindformers.model_runner.AutoTokenizer')
    @patch('mindformers.model_runner.MindFormerConfig')
    @patch('mindformers.model_runner.register_auto_class')
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_model_single_modal(self, mock_register, mock_config_cls, mock_tokenizer):
        """Test getting single modal model"""
        mock_config = MockConfigFactory.create_mindformer_config()
        mock_config.processor = MagicMock()
        mock_config_cls.return_value = mock_config

        mock_tok = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tok

        tokenizer, input_builder = get_model(self.test_dir)

        assert tokenizer == mock_tok
        assert isinstance(input_builder, InputBuilder)
        mock_register.assert_called_once()

    @patch('mindformers.model_runner.AutoTokenizer')
    @patch('mindformers.model_runner.MindFormerConfig')
    @patch('mindformers.model_runner.register_auto_class')
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_model_with_kwargs(self, mock_register, mock_config_cls, mock_tokenizer):
        """Test get_model with additional kwargs"""
        mock_config = MockConfigFactory.create_mindformer_config()
        mock_config_cls.return_value = mock_config

        mock_tok = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tok

        get_model(self.test_dir, revision='main', trust_remote_code=True, use_fast=False)

        mock_tokenizer.from_pretrained.assert_called_once_with(
            self.test_dir,
            revision='main',
            trust_remote_code=True,
            use_fast=False
        )


# pylint: disable=C2801
class TestGetModelConfig:
    """Test _get_model_config function"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Set up and tear down test fixtures"""
        self.temp_dir = TempDirFixture()
        self.test_dir = self.temp_dir.__enter__()
        yield
        self.temp_dir.__exit__(None, None, None)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_config_success(self):
        """Test successfully getting config file"""
        yaml_file = os.path.join(self.test_dir, 'model_config.yaml')
        with open(yaml_file, 'w', encoding='utf-8') as f:
            f.write('test: config')

        result = _get_model_config(self.test_dir)
        assert result == yaml_file

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_no_yaml_file(self):
        """Test when directory has no yaml file"""
        with pytest.raises(FileNotFoundError) as exc_info:
            _get_model_config(self.test_dir)
        assert 'no yaml file' in str(exc_info.value)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_path_not_exist(self):
        """Test when path doesn't exist"""
        with pytest.raises(ValueError) as exc_info:
            _get_model_config('/nonexistent/path')
        assert 'not exist' in str(exc_info.value)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_multiple_yaml_files(self):
        """Test when directory has multiple yaml files (should return first one)"""
        yaml1 = os.path.join(self.test_dir, 'config1.yaml')
        yaml2 = os.path.join(self.test_dir, 'config2.yaml')
        with open(yaml1, 'w', encoding='utf-8') as f:
            f.write('test: 1')
        with open(yaml2, 'w', encoding='utf-8') as f:
            f.write('test: 2')

        result = _get_model_config(self.test_dir)
        assert result.endswith('.yaml')


class TestModel:
    """
    Test Model.
    """
    def forward(self, input_ids, valid_length_each_example, block_tables, slot_mapping, prefill, use_past,
                position_ids=None, spec_mask=None, q_seq_lens=None, adapter_ids=None, prefill_head_indices=None,
                mindie_warm_up=False, key_cache=None, value_cache=None):
        """
        Check the info of inputs

        Args:
            input_ids (np.ndarray): rank is 2, and data type is int32.
            valid_length_each_example (Union[np.ndarray, list]): rank is 1, and data type is int32.
            block_tables (np.ndarray): rank is 2, and data type is int32.
            slot_mapping (np.ndarray): rank is 1, and data type is int32.
            prefill (bool).
            use_past (bool).
            position_ids (Union[np.ndarray, list]): rank is 1, and data type is int32.
            spec_mask (np.ndarray): rank is 2 or 3, and data type is float16.
            q_seq_lens (Union[np.ndarray, list]): rank is 1, and data type is int32.
            adapter_ids (list): rank is 1, and data type is string
            prefill_head_indices (Union[np.ndarray, list]): rank is 1, and data type is int32.
            mindie_warm_up (bool).

        Return:
            res: (Tensor): given that shape is (2, 16000).
            current_index (list): given that length is 1.
        """
        assert isinstance(input_ids, np.ndarray) and input_ids.ndim == 2 and input_ids.dtype == np.int32
        assert isinstance(valid_length_each_example, (np.ndarray, list))
        if isinstance(valid_length_each_example, np.ndarray):
            assert valid_length_each_example.ndim == 1 and valid_length_each_example.dtype == np.int32
        else:
            assert isinstance(valid_length_each_example[0], int)
        assert isinstance(block_tables, np.ndarray) and block_tables.ndim == 2 and block_tables.dtype == np.int32
        assert isinstance(slot_mapping, np.ndarray) and slot_mapping.ndim == 1 and slot_mapping.dtype == np.int32
        assert isinstance(prefill, bool)
        assert isinstance(use_past, bool)
        assert isinstance(mindie_warm_up, bool)
        assert isinstance(key_cache, type(None))
        assert isinstance(value_cache, type(None))
        if position_ids is not None:
            if isinstance(position_ids, np.ndarray):
                assert position_ids.ndim == 1 and position_ids.dtype == np.int32
            else:
                assert isinstance(position_ids[0], int)
        if spec_mask is not None:
            assert isinstance(spec_mask, np.ndarray) and (2 <= spec_mask.ndim <= 3) and spec_mask.dtype == np.float16
        if q_seq_lens is not None:
            if isinstance(q_seq_lens, np.ndarray):
                assert q_seq_lens.ndim == 1 and q_seq_lens.dtype == np.int32
            else:
                assert isinstance(q_seq_lens[0], int)
        if adapter_ids is not None:
            assert isinstance(adapter_ids, list) and adapter_ids.dtype == np.str

        if prefill_head_indices is not None:
            if isinstance(prefill_head_indices, np.ndarray):
                assert prefill_head_indices.ndim == 1 and prefill_head_indices.dtype == np.int32
            else:
                assert isinstance(prefill_head_indices[0], int)
        res = np.arange(32000).reshape(2, -1)
        current_index = [1]
        return Tensor.from_numpy(res), current_index


# pylint: disable=C2801
class TestModelRunner:
    """Test ModelRunner factory class"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Set up and tear down test fixtures"""
        self.temp_dir = TempDirFixture()
        self.test_dir = self.temp_dir.__enter__()
        self.yaml_file = os.path.join(self.test_dir, 'config.yaml')
        with open(self.yaml_file, 'w', encoding='utf-8') as f:
            f.write('model:\n  arch:\n    type: llama\n')
        yield
        self.temp_dir.__exit__(None, None, None)

    @patch('mindformers.model_runner.MindIEModelRunner')
    @patch('mindformers.model_runner.MindFormerConfig')
    @patch('mindformers.model_runner.models')
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_model_runner_default(self, mock_models, mock_config_cls, mock_runner_cls):
        """Test ModelRunner with default model type"""
        mock_config = MockConfigFactory.create_mindformer_config()
        mock_config_cls.return_value = mock_config
        mock_models.__all__ = ['llama']

        mock_runner = MagicMock()
        mock_runner_cls.return_value = mock_runner

        result = ModelRunner(
            model_path=self.test_dir,
            npu_mem_size=1,
            cpu_mem_size=1,
            block_size=128
        )

        assert result == mock_runner
        mock_runner_cls.assert_called_once()

    @patch('mindformers.model_runner.MindIEModelRunner')
    @patch('mindformers.model_runner.MindFormerConfig')
    @patch('mindformers.model_runner.models')
    @patch('importlib.import_module')
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_model_runner_custom_type_import_success(self, mock_import, mock_models,
                                                     mock_config_cls, mock_default_runner):
        """Test ModelRunner with custom model type that imports successfully"""
        mock_config = MockConfigFactory.create_mindformer_config(arch_type='custom_model')
        mock_config_cls.return_value = mock_config
        mock_models.__all__ = ['llama', 'gpt']

        # Mock custom runner class
        mock_custom_runner_cls = MagicMock()
        mock_custom_module = MagicMock()
        mock_custom_module.MindIEModelRunner = mock_custom_runner_cls
        mock_import.return_value = mock_custom_module

        mock_runner = MagicMock()
        mock_custom_runner_cls.return_value = mock_runner

        result = ModelRunner(
            model_path=self.test_dir,
            npu_mem_size=1,
            cpu_mem_size=1,
            block_size=128
        )

        assert result == mock_runner
        mock_import.assert_called_once_with('custom_model', ['MindIEModelRunner'])

    @patch('mindformers.model_runner.MindIEModelRunner')
    @patch('mindformers.model_runner.MindFormerConfig')
    @patch('mindformers.model_runner.models')
    @patch('importlib.import_module')
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_model_runner_custom_type_import_fail(self, mock_import, mock_models,
                                                   mock_config_cls, mock_default_runner):
        """Test ModelRunner with custom model type that fails to import"""
        mock_config = MockConfigFactory.create_mindformer_config(arch_type='custom_model')
        mock_config_cls.return_value = mock_config
        mock_models.__all__ = ['llama', 'gpt']

        # Simulate import error
        mock_import.side_effect = ImportError("Module not found")

        mock_runner = MagicMock()
        mock_default_runner.return_value = mock_runner

        result = ModelRunner(
            model_path=self.test_dir,
            npu_mem_size=1,
            cpu_mem_size=1,
            block_size=128
        )

        # Should fall back to default runner
        assert result == mock_runner
        mock_default_runner.assert_called_once()


# pylint: disable=W0613
# pylint: disable=C2801
class TestMindIEModelRunnerInit:
    """Test MindIEModelRunner initialization"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Set up and tear down test fixtures"""
        self.temp_dir = TempDirFixture()
        self.test_dir = self.temp_dir.__enter__()
        self.yaml_file = os.path.join(self.test_dir, 'config.yaml')

        config_content = """
        model:
        arch:
            type: llama
        model_config:
            type: llama_config
            num_hidden_layers: 2
            hidden_size: 128
            num_attention_heads: 4
            compute_dtype: float32
            seq_length: 512
            batch_size: 1
        processor:
        tokenizer:
            type: llama_tokenizer
        context:
        device_target: CPU
        mode: 0
        """
        with open(self.yaml_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        yield
        self.temp_dir.__exit__(None, None, None)

    def _setup_base_mocks(self):
        """Setup common mocks for initialization tests"""
        patches = {
            'ms_model': patch('mindspore.Model'),
            'auto_model': patch('mindformers.model_runner.AutoModel'),
            'auto_tokenizer': patch('mindformers.model_runner.AutoTokenizer'),
            'auto_config': patch('mindformers.model_runner.AutoConfig'),
            'build_context': patch('mindformers.model_runner.build_context'),
            'is_legacy': patch('mindformers.model_runner.is_legacy_model'),
            'config_cls': patch('mindformers.model_runner.MindFormerConfig'),
            'gen_config': patch('mindformers.model_runner.GenerationConfig'),
            'get_load_path': patch('mindformers.model_runner.get_load_path_after_hf_convert'),
            'no_init': patch('mindformers.model_runner.no_init_parameters'),
            'register': patch('mindformers.model_runner.register_auto_class'),
        }
        return patches

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_init_invalid_plugin_params(self):
        """Test initialization with invalid plugin params"""
        with patch('mindformers.model_runner.MindFormerConfig'):
            with pytest.raises(ValueError) as exc_info:
                MindIEModelRunner(
                    model_path=self.test_dir,
                    config_path=self.yaml_file,
                    npu_mem_size=1,
                    cpu_mem_size=1,
                    block_size=128,
                    plugin_params={'invalid': 'dict'}  # Should be string
                )
            assert 'plugin params should be str type' in str(exc_info.value)

    @patch('mindspore.Model')
    @patch('mindformers.model_runner.AutoModel')
    @patch('mindformers.model_runner.AutoTokenizer')
    @patch('mindformers.model_runner.AutoConfig')
    @patch('mindformers.model_runner.build_context')
    @patch('mindformers.model_runner.is_legacy_model')
    @patch('mindformers.model_runner.MindFormerConfig')
    @patch('mindformers.model_runner.GenerationConfig')
    @patch('mindformers.model_runner.get_load_path_after_hf_convert')
    @patch('mindformers.model_runner.no_init_parameters')
    @patch('mindformers.model_runner.register_auto_class')
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_init_single_modal_legacy(self, mock_register, mock_no_init, mock_get_load_path,
                                       mock_gen_config, mock_config_cls, mock_is_legacy,
                                       mock_build_ctx, mock_auto_config, mock_tokenizer,
                                       mock_auto_model, mock_ms_model):
        """Test initialization for single modal model with legacy mode"""
        mock_config = MockConfigFactory.create_mindformer_config()
        mock_config_cls.return_value = mock_config

        mock_model_config = MockConfigFactory.create_model_config()
        mock_auto_config.from_pretrained.return_value = mock_model_config

        mock_is_legacy.return_value = True

        mock_model = MagicMock()
        mock_model.__class__.__name__ = 'TestModel'
        mock_model.kvcache.return_value = (MagicMock(), MagicMock())
        mock_auto_model.from_config.return_value = mock_model

        mock_tok = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tok

        mock_get_load_path.return_value = None
        mock_gen_config.from_model_config.return_value = MagicMock()

        ctx_mgr = MagicMock()
        ctx_mgr.__enter__ = Mock(return_value=None)
        ctx_mgr.__exit__ = Mock(return_value=False)
        mock_no_init.return_value = ctx_mgr

        runner = MindIEModelRunner(
            model_path=self.test_dir,
            config_path=self.yaml_file,
            npu_mem_size=1,
            cpu_mem_size=1,
            block_size=128
        )

        assert runner.num_layers == 2
        assert runner.warmup_step == 2
        assert runner.is_multi_modal_model is False
        mock_register.assert_called()
        assert mock_register.call_count == 3

    @patch('mindspore.Model')
    @patch('mindformers.model_runner.AutoModel')
    @patch('mindformers.model_runner.AutoTokenizer')
    @patch('mindformers.model_runner.AutoConfig')
    @patch('mindformers.model_runner.build_context')
    @patch('mindformers.model_runner.is_legacy_model')
    @patch('mindformers.model_runner.MindFormerConfig')
    @patch('mindformers.model_runner.GenerationConfig')
    @patch('mindformers.model_runner.get_load_path_after_hf_convert')
    @patch('mindformers.model_runner.no_init_parameters')
    @patch('mindformers.model_runner.register_auto_class')
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_init_with_npu_device_ids_error(self, mock_register, mock_no_init, mock_get_load_path,
                                             mock_gen_config, mock_config_cls, mock_is_legacy,
                                             mock_build_ctx, mock_auto_config, mock_tokenizer,
                                             mock_auto_model, mock_ms_model):
        """Test initialization with multiple npu_device_ids in non-parallel mode"""
        mock_config = MockConfigFactory.create_mindformer_config()
        mock_config_cls.return_value = mock_config

        with pytest.raises(ValueError) as exc_info:
            MindIEModelRunner(
                model_path=self.test_dir,
                config_path=self.yaml_file,
                npu_mem_size=1,
                cpu_mem_size=1,
                block_size=128,
                npu_device_ids=[0, 1]  # Multiple devices in non-parallel mode
            )
        assert 'should only contain one device_id' in str(exc_info.value)

    @patch('mindspore.Model')
    @patch('mindformers.model_runner.AutoModel')
    @patch('mindformers.model_runner.AutoTokenizer')
    @patch('mindformers.model_runner.AutoConfig')
    @patch('mindformers.model_runner.build_context')
    @patch('mindformers.model_runner.is_legacy_model')
    @patch('mindformers.model_runner.MindFormerConfig')
    @patch('mindformers.model_runner.GenerationConfig')
    @patch('mindformers.model_runner.get_load_path_after_hf_convert')
    @patch('mindformers.model_runner.no_init_parameters')
    @patch('mindformers.model_runner.register_auto_class')
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_init_dynamic_kv_cache_error(self, mock_register, mock_no_init, mock_get_load_path,
                                          mock_gen_config, mock_config_cls, mock_is_legacy,
                                          mock_build_ctx, mock_auto_config, mock_tokenizer,
                                          mock_auto_model, mock_ms_model):
        """Test initialization with npu_mem_size=-1 for unsupported model"""
        mock_config = MockConfigFactory.create_mindformer_config()
        mock_config_cls.return_value = mock_config

        mock_model_config = MockConfigFactory.create_model_config()
        mock_auto_config.from_pretrained.return_value = mock_model_config

        mock_is_legacy.return_value = True

        mock_model = MagicMock()
        mock_model.__class__.__name__ = 'UnsupportedModel'  # Not in whitelist
        mock_auto_model.from_config.return_value = mock_model

        mock_get_load_path.return_value = None
        mock_gen_config.from_model_config.return_value = MagicMock()

        ctx_mgr = MagicMock()
        ctx_mgr.__enter__ = Mock(return_value=None)
        ctx_mgr.__exit__ = Mock(return_value=False)
        mock_no_init.return_value = ctx_mgr

        with pytest.raises(ValueError) as exc_info:
            MindIEModelRunner(
                model_path=self.test_dir,
                config_path=self.yaml_file,
                npu_mem_size=-1,  # Dynamic kv cache
                cpu_mem_size=1,
                block_size=128
            )
        assert 'npu_mem_size=-1 only support in parallel mode' in str(exc_info.value)

    @patch('mindspore.Model')
    @patch('mindformers.model_runner.AutoModel')
    @patch('mindformers.model_runner.AutoTokenizer')
    @patch('mindformers.model_runner.AutoConfig')
    @patch('mindformers.model_runner.build_context')
    @patch('mindformers.model_runner.is_legacy_model')
    @patch('mindformers.model_runner.MindFormerConfig')
    @patch('mindformers.model_runner.GenerationConfig')
    @patch('mindformers.model_runner.get_load_path_after_hf_convert')
    @patch('mindformers.model_runner.no_init_parameters')
    @patch('mindformers.model_runner.register_auto_class')
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_update_model_config_with_plugin_params(self, mock_register, mock_no_init,
                                                     mock_get_load_path, mock_gen_config,
                                                     mock_config_cls, mock_is_legacy,
                                                     mock_build_ctx, mock_auto_config,
                                                     mock_tokenizer, mock_auto_model, mock_ms_model):
        """Test update_model_config with plugin parameters"""
        mock_config = MockConfigFactory.create_mindformer_config()
        mock_config_cls.return_value = mock_config

        mock_model_config = MockConfigFactory.create_model_config()
        mock_auto_config.from_pretrained.return_value = mock_model_config

        mock_is_legacy.return_value = True

        mock_model = MagicMock()
        mock_model.__class__.__name__ = 'TestModel'
        mock_auto_model.from_config.return_value = mock_model

        mock_get_load_path.return_value = None
        mock_gen_config.from_model_config.return_value = MagicMock()

        ctx_mgr = MagicMock()
        ctx_mgr.__enter__ = Mock(return_value=None)
        ctx_mgr.__exit__ = Mock(return_value=False)
        mock_no_init.return_value = ctx_mgr

        plugin_params = json.dumps({'plugin_type': 'speculative_decoding'})

        runner = MindIEModelRunner(
            model_path=self.test_dir,
            config_path=self.yaml_file,
            npu_mem_size=1,
            cpu_mem_size=1,
            block_size=128,
            plugin_params=plugin_params
        )

        assert runner.model_config.parallel_decoding_params is not None

    @patch('mindspore.Model')
    @patch('mindformers.model_runner.AutoModel')
    @patch('mindformers.model_runner.AutoTokenizer')
    @patch('mindformers.model_runner.AutoConfig')
    @patch('mindformers.model_runner.build_context')
    @patch('mindformers.model_runner.build_parallel_config')
    @patch('mindformers.model_runner.is_legacy_model')
    @patch('mindformers.model_runner.MindFormerConfig')
    @patch('mindformers.model_runner.GenerationConfig')
    @patch('mindformers.model_runner.get_load_path_after_hf_convert')
    @patch('mindformers.model_runner.no_init_parameters')
    @patch('mindformers.model_runner.register_auto_class')
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_init_with_moe_config(self, mock_register, mock_no_init, mock_get_load_path,
                                   mock_gen_config, mock_config_cls, mock_is_legacy,
                                   mock_build_parallel, mock_build_ctx, mock_auto_config,
                                   mock_tokenizer, mock_auto_model, mock_ms_model):
        """Test initialization with MoE config"""
        mock_config = MockConfigFactory.create_mindformer_config(
            arch_type='llama_moe',
            use_parallel=True,
            moe_config={'num_experts': 8}
        )
        mock_config_cls.return_value = mock_config

        mock_model_config = MockConfigFactory.create_model_config()
        mock_auto_config.from_pretrained.return_value = mock_model_config

        mock_is_legacy.return_value = False

        mock_model = MagicMock()
        mock_model.__class__.__name__ = 'TestModel'
        mock_auto_model.from_config.return_value = mock_model

        mock_get_load_path.return_value = None
        mock_gen_config.from_model_config.return_value = MagicMock()

        ctx_mgr = MagicMock()
        ctx_mgr.__enter__ = Mock(return_value=None)
        ctx_mgr.__exit__ = Mock(return_value=False)
        mock_no_init.return_value = ctx_mgr

        runner = MindIEModelRunner(
            model_path=self.test_dir,
            config_path=self.yaml_file,
            npu_mem_size=1,
            cpu_mem_size=1,
            block_size=128
        )

        assert runner.model_config.moe_config == {'num_experts': 8}


# pylint: disable=W0612
# pylint: disable=W0613
class TestMindIEModelRunnerMethods:
    """Test MindIEModelRunner methods"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures"""
        self.runner = MockRunnerFactory.create_runner_mock()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_forward_legacy_prefill(self):
        """Test forward method in legacy mode with prefill"""
        input_ids = [[1, 2, 3, 4]]
        valid_length = [4]

        mock_logits = Tensor(np.random.randn(1, 4, 1000).astype(np.float32))
        current_idx = [0]
        self.runner.model.forward.return_value = (mock_logits, current_idx)

        result = MindIEModelRunner.forward(
            self.runner,
            input_ids=input_ids,
            valid_length_each_example=valid_length,
            prefill=True
        )

        self.runner.model.forward.assert_called_once()
        assert self.runner.warmup_step == 1

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_forward_non_legacy_decode(self):
        """Test forward method in non-legacy mode with decode"""
        self.runner.use_legacy = False
        self.runner.warmup_step = 0

        input_ids = [[1]]
        valid_length = [1]

        mock_logits = Tensor(np.random.randn(1, 1, 1000).astype(np.float32))
        current_idx = []
        self.runner.model.forward_mcore.return_value = (mock_logits, current_idx)

        result = MindIEModelRunner.forward(
            self.runner,
            input_ids=input_ids,
            valid_length_each_example=valid_length,
            prefill=False
        )

        self.runner.model.forward_mcore.assert_called_once()

    @patch('mindformers.model_runner.parallel_decoding_control')
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_forward_parallel_decoding(self, mock_parallel_ctrl):
        """Test forward with parallel decoding enabled"""
        mock_parallel_ctrl.return_value = True

        input_ids = [[1, 2]]
        valid_length = [2]

        mock_logits = Tensor(np.random.randn(1, 2, 1000).astype(np.float32))
        self.runner.model.forward.return_value = (mock_logits, [])

        result = MindIEModelRunner.forward(
            self.runner,
            input_ids=input_ids,
            valid_length_each_example=valid_length
        )

        mock_parallel_ctrl.assert_called_once()

    @patch('mindformers.model_runner.swap_cache')
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_swap(self, mock_swap_cache):
        """Test swap method"""
        self.runner.key_host = [MagicMock(), MagicMock()]
        self.runner.value_host = [MagicMock(), MagicMock()]
        self.runner.model.kvcache.side_effect = [
            (MagicMock(), MagicMock()),
            (MagicMock(), MagicMock())
        ]

        block_tables = [[0, 1], [2, 3]]
        swap_type = True

        MindIEModelRunner.swap(self.runner, block_tables, swap_type)

        assert mock_swap_cache.call_count == 4  # 2 layers * 2 (key + value)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_generate_position_ids_single_modal(self):
        """Test generate_position_ids for single modal model"""
        input_ids = [1, 2, 3, 4, 5]

        result = MindIEModelRunner.generate_position_ids(self.runner, input_ids)

        assert list(result) == [0, 1, 2, 3, 4]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_generate_position_ids_multi_modal_warmup(self):
        """Test generate_position_ids for multi-modal during warmup"""
        self.runner.is_multi_modal_model = True
        self.runner.warmup_step = 1
        input_ids = [1, 2, 3]

        result = MindIEModelRunner.generate_position_ids(self.runner, input_ids)

        assert list(result) == [0, 1, 2]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_generate_position_ids_multi_modal_no_warmup(self):
        """Test generate_position_ids for multi-modal after warmup"""
        self.runner.is_multi_modal_model = True
        self.runner.warmup_step = 0
        self.runner.processor.decode_position_ids_from_input_ids.return_value = [0, 5, 10]
        input_ids = [1, 2, 3]

        result = MindIEModelRunner.generate_position_ids(self.runner, input_ids)

        self.runner.processor.decode_position_ids_from_input_ids.assert_called_once_with(input_ids)
        assert result == [0, 5, 10]

    @patch('mindformers.model_runner.transform_and_load_checkpoint')
    @patch('mindformers.model_runner.get_load_path_after_hf_convert')
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_load_checkpoint_with_checkpoint(self, mock_get_load, mock_transform_load):
        """Test load_checkpoint when checkpoint exists"""
        self.runner.model_config.batch_size = 1
        self.runner.model_config.seq_length = 512
        mock_get_load.return_value = '/path/to/checkpoint'

        with patch('mindspore.Model') as mock_ms_model:
            MindIEModelRunner.load_checkpoint(self.runner)

            mock_transform_load.assert_called_once()
            self.runner.model.init_parameters_data.assert_called_once()

    @patch('mindformers.model_runner.get_load_path_after_hf_convert')
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_load_checkpoint_no_checkpoint(self, mock_get_load):
        """Test load_checkpoint when no checkpoint exists"""
        self.runner.model_config.batch_size = 1
        self.runner.model_config.seq_length = 512
        mock_get_load.return_value = None

        with patch('mindspore.Model'):
            MindIEModelRunner.load_checkpoint(self.runner)

            self.runner.model.init_parameters_data.assert_called_once()

    @patch('mindformers.model_runner.get_load_path_after_hf_convert')
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_load_checkpoint_legacy(self, mock_get_load):
        """Test load_checkpoint in legacy mode"""
        self.runner.model_config.batch_size = 1
        self.runner.model_config.seq_length = 512
        self.runner.use_legacy = True
        self.runner.model.prepare_inputs_for_predict_layout.return_value = MagicMock()
        mock_get_load.return_value = None

        with patch('mindspore.Model'):
            MindIEModelRunner.load_checkpoint(self.runner)

            self.runner.model.prepare_inputs_for_predict_layout.assert_called_once()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_update_model_config_default_plugin(self):
        """Test update_model_config with default plugin config"""
        self.runner.config.load_checkpoint = '/path/to/checkpoint'

        MindIEModelRunner.update_model_config(
            self.runner,
            plugin_params={'plugin_type': None}
        )

        assert self.runner.model_config.parallel_decoding_params is None

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_update_model_config_json_string(self):
        """Test update_model_config with JSON string"""
        self.runner.config.load_checkpoint = '/path/to/checkpoint'

        plugin_params = json.dumps({'plugin_type': 'medusa'})

        MindIEModelRunner.update_model_config(self.runner, plugin_params)

        assert self.runner.model_config.parallel_decoding_params is not None
        assert self.runner.model_config.parallel_decoding_params['parallel_decoding'] == 'medusa'

    @patch('mindformers.model_runner.need_nz')
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_update_llm_config_legacy_with_nz(self, mock_need_nz):
        """Test update_llm_config in legacy mode with NZ"""
        mock_need_nz.return_value = True
        self.runner.use_legacy = True

        config = MockConfigFactory.create_model_config(
            num_layers=4,
            num_heads=8,
            hidden_size=256,
            compute_dtype='float16',
            seq_length=1024
        )

        MindIEModelRunner.update_llm_config(
            self.runner,
            config=config,
            world_size=1,
            npu_mem_size=2,
            cpu_mem_size=1,
            block_size=64
        )

        assert self.runner.num_layers == 4
        assert self.runner.num_kv_heads == 8
        assert config.block_size == 64
        assert config.num_blocks is not None

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_update_llm_config_non_legacy(self):
        """Test update_llm_config in non-legacy mode"""
        self.runner.use_legacy = False
        self.runner.model_config.quantization_config = None

        config = MockConfigFactory.create_model_config(
            num_layers=6,
            num_heads=12,
            n_kv_heads=4,
            hidden_size=384,
            compute_dtype='float32',
            seq_length=2048
        )

        with patch('mindformers.model_runner.need_nz', return_value=False):
            MindIEModelRunner.update_llm_config(
                self.runner,
                config=config,
                world_size=2,
                npu_mem_size=4,
                cpu_mem_size=2,
                block_size=128
            )

        assert self.runner.num_layers == 6
        assert self.runner.num_kv_heads == 2  # 4 // 2
        assert self.runner.head_size == 32  # 384 // 12

    @patch('mindformers.model_runner.str_to_ms_type', {'int8': ms.int8})
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_update_llm_config_with_quantization(self):
        """Test update_llm_config with quantization config"""
        self.runner.use_legacy = False
        self.runner.model_config.quantization_config = MagicMock()
        self.runner.model_config.quantization_config.kvcache_dtype = 'int8'

        config = MockConfigFactory.create_model_config()

        with patch('mindformers.model_runner.need_nz', return_value=False):
            MindIEModelRunner.update_llm_config(
                self.runner,
                config=config,
                world_size=1,
                npu_mem_size=1,
                cpu_mem_size=1,
                block_size=64
            )

        assert self.runner.dtype is not None

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_update_llm_config_no_max_position_embedding(self):
        """Test update_llm_config when max_position_embedding is not set"""
        self.runner.use_legacy = False

        config = MockConfigFactory.create_model_config()
        config.max_position_embedding = None

        with patch('mindformers.model_runner.need_nz', return_value=False):
            MindIEModelRunner.update_llm_config(
                self.runner,
                config=config,
                world_size=1,
                npu_mem_size=1,
                cpu_mem_size=1,
                block_size=64
            )

        assert config.max_position_embedding == 512


# pylint: disable=W0212
# pylint: disable=W0612
class TestInputBuilder:
    """Test InputBuilder class"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures"""
        self.tokenizer = MagicMock()
        self.tokenizer.chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
        self.tokenizer.apply_chat_template = MagicMock(return_value=[1, 2, 3, 4, 5])

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_init_default(self):
        """Test InputBuilder initialization with defaults"""
        builder = InputBuilder(self.tokenizer)

        assert builder.tokenizer == self.tokenizer
        assert builder.system_role_name == 'system'
        assert builder.user_role_name == 'user'
        assert builder.max_length == 2048
        assert builder.rank == 0
        assert builder.adapt_to_max_length is False

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_init_with_params(self):
        """Test InputBuilder initialization with custom parameters"""
        builder = InputBuilder(
            self.tokenizer,
            chat_template="custom template",
            system_role_name="sys",
            user_role_name="usr",
            max_length=4096
        )

        assert builder.tokenizer.chat_template == "custom template"
        assert builder.system_role_name == 'sys'
        assert builder.user_role_name == 'usr'
        assert builder.max_length == 4096

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_make_context_default(self):
        """Test make_context with default parameters"""
        builder = InputBuilder(self.tokenizer)
        conversation = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there!'}
        ]

        result = builder.make_context(rank=0, conversation=conversation)

        assert result == [1, 2, 3, 4, 5]
        assert builder.rank == 0
        assert builder.adapt_to_max_length is False

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_make_context_with_params(self):
        """Test make_context with custom parameters"""
        builder = InputBuilder(self.tokenizer)
        conversation = [{'role': 'user', 'content': 'Test'}]

        result = builder.make_context(
            rank=1,
            conversation=conversation,
            add_generation_prompt=False,
            adapt_to_max_length=True
        )

        assert builder.rank == 1
        assert builder.adapt_to_max_length is True
        self.tokenizer.apply_chat_template.assert_called_once()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_make_context_with_kwargs(self):
        """Test make_context with additional kwargs"""
        builder = InputBuilder(self.tokenizer)
        conversation = [{'role': 'user', 'content': 'Test'}]

        result = builder.make_context(
            rank=0,
            conversation=conversation,
            max_length=512,
            padding=True
        )

        call_kwargs = self.tokenizer.apply_chat_template.call_args[1]
        assert 'max_length' in call_kwargs
        assert 'padding' in call_kwargs

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_apply_chat_template_no_method(self):
        """Test _apply_chat_template when tokenizer doesn't have the method"""
        tokenizer = MagicMock(spec=[])  # No apply_chat_template method
        builder = InputBuilder(tokenizer)

        with pytest.raises(RuntimeError) as exc_info:
            builder._apply_chat_template([])
        assert 'does not implement apply_chat_template' in str(exc_info.value)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_apply_chat_template_no_template(self):
        """Test _apply_chat_template when tokenizer has no chat_template"""
        tokenizer = MagicMock()
        tokenizer.chat_template = None
        builder = InputBuilder(tokenizer)

        with pytest.raises(RuntimeError) as exc_info:
            builder._apply_chat_template([])
        assert 'not configured with a `chat_template`' in str(exc_info.value)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_apply_chat_template_success(self):
        """Test _apply_chat_template successful execution"""
        builder = InputBuilder(self.tokenizer)
        conversation = [
            {'role': 'system', 'content': 'You are helpful'},
            {'role': 'user', 'content': 'Hello'}
        ]

        result = builder._apply_chat_template(conversation)

        assert result == [1, 2, 3, 4, 5]
        self.tokenizer.apply_chat_template.assert_called_once_with(conversation)


# pylint: disable=C2801
class TestSafetensorsHelpers:
    """Test safetensors helper functions"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Set up and tear down test fixtures"""
        self.temp_dir = TempDirFixture()
        self.test_dir = self.temp_dir.__enter__()
        yield
        self.temp_dir.__exit__(None, None, None)

    @patch('mindspore.load_distributed_checkpoint')
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_load_distributed_safetensors(self, mock_load_dist):
        """Test _load_distributed_safetensors"""
        model = MagicMock()
        strategy_path = '/path/to/strategy'
        load_path = '/path/to/safetensors'

        _load_distributed_safetensors(model, strategy_path, load_path)

        mock_load_dist.assert_called_once_with(
            network=model,
            predict_strategy=strategy_path,
            unified_safetensors_dir=load_path,
            format='safetensors'
        )

    @patch('mindspore.load_checkpoint')
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_load_safetensors_single(self, mock_load_ckpt):
        """Test _load_safetensors with single file"""
        sf_file = os.path.join(self.test_dir, 'model.safetensors')
        with open(sf_file, 'wb') as f:
            f.write(b'dummy content')

        model = MagicMock()

        _load_safetensors(model, self.test_dir)

        mock_load_ckpt.assert_called_once()
        call_args = mock_load_ckpt.call_args[1]
        assert call_args['ckpt_file_name'].endswith('.safetensors')
        assert call_args['net'] == model
        assert call_args['format'] == 'safetensors'

    @patch('mindspore.load_checkpoint')
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_load_safetensors_multiple(self, mock_load_ckpt):
        """Test _load_safetensors with multiple files"""
        for i in range(3):
            sf_file = os.path.join(self.test_dir, f'model_{i}.safetensors')
            with open(sf_file, 'wb') as f:
                f.write(f'dummy content {i}'.encode())

        model = MagicMock()

        _load_safetensors(model, self.test_dir)

        assert mock_load_ckpt.call_count == 3

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_load_safetensors_no_files(self):
        """Test _load_safetensors when no safetensors files exist"""
        model = MagicMock()

        with pytest.raises(FileNotFoundError) as exc_info:
            _load_safetensors(model, self.test_dir)
        assert 'no safetensors files' in str(exc_info.value)

    @patch('mindformers.model_runner.contains_safetensors_files')
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_check_valid_safetensors_path_valid(self, mock_contains):
        """Test _check_valid_safetensors_path with valid path"""
        mock_contains.return_value = True

        _check_valid_safetensors_path(self.test_dir)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_check_valid_safetensors_path_invalid_type(self):
        """Test _check_valid_safetensors_path with invalid type"""
        with pytest.raises(ValueError) as exc_info:
            _check_valid_safetensors_path(123)
        assert 'must be a str' in str(exc_info.value)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_check_valid_safetensors_path_not_exist(self):
        """Test _check_valid_safetensors_path with non-existent path"""
        with pytest.raises(ValueError) as exc_info:
            _check_valid_safetensors_path('/nonexistent/path')
        assert 'does not exist' in str(exc_info.value)

    @patch('mindformers.model_runner.contains_safetensors_files')
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_check_valid_safetensors_path_no_safetensors(self, mock_contains):
        """Test _check_valid_safetensors_path when directory has no safetensors"""
        mock_contains.return_value = False

        with pytest.raises(ValueError) as exc_info:
            _check_valid_safetensors_path(self.test_dir)
        assert 'not a valid path for safetensors' in str(exc_info.value)


class TestMindIEModelRunner:
    """
    Test MindIEModelRunner API.
    1. Check the type of `__init__` attributes.
    2. Check the type of `forward` inputs.
    3. Check the dimension and data type of `forward` inputs if they are `np.ndarray` or `Tensor`.
    4. Check the consistency between the `res` in model and the `logits` in model_runner which should be same
       if the `res` is `Tensor`, otherwise, `res` would be a list[Tensor], the shape of `res[0]` is compared.
    """
    def __init__(self, model_path, config_path, npu_mem_size, cpu_mem_size, block_size, rank_id=0, world_size=1,
                 npu_device_ids=None, plugin_params=None):
        """Test __init__ api"""
        self.warmup_step = 2
        self.is_multi_modal_model = False
        self.model = TestModel()
        assert isinstance(model_path, str)
        assert isinstance(config_path, str)
        assert isinstance(npu_mem_size, int)
        assert isinstance(cpu_mem_size, int)
        assert isinstance(block_size, int)
        assert isinstance(rank_id, int)
        assert isinstance(world_size, int)
        assert isinstance(npu_device_ids, list) and isinstance(npu_device_ids[0], int)
        if plugin_params:
            assert isinstance(plugin_params, str)

        assert config_path == os.path.join(model_path, 'test_config.yaml')


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@mock.patch('mindformers.model_runner.MindIEModelRunner.__init__', TestMindIEModelRunner.__init__)
def test_model_runner():
    """
    Feature: Test ModelRunner API.
    Description: Test ModelRunner API.
    Expectation: Success.
    """
    current_file_path = os.path.dirname(__file__)
    model_path = os.path.join(current_file_path, "test_files")
    npu_mem_size = 3
    cpu_mem_size = 1
    block_size = 128
    rank_id = 0
    world_size = 1
    npu_device_ids = [0]
    model_runner = ModelRunner(model_path=model_path, npu_mem_size=npu_mem_size, cpu_mem_size=cpu_mem_size,
                               block_size=block_size, rank_id=rank_id, world_size=world_size,
                               npu_device_ids=npu_device_ids)
    model_runner.use_legacy = True
    input_ids = np.arange(32 * 256).reshape(32, 256).astype(np.int32)
    valid_length_each_example = np.arange(32).astype(np.int32)
    block_tables = np.arange(32 * 256).reshape(32, 256).astype(np.int32)
    slot_mapping = np.arange(32 * 256).astype(np.int32)
    # Given that `prefill` is False, the shape of `logits` should be the same as that of `res` in model.forward
    prefill = False
    logits = model_runner.forward(input_ids=input_ids, valid_length_each_example=valid_length_each_example,
                                  block_tables=block_tables, slot_mapping=slot_mapping, prefill=prefill)
    assert logits.shape == (2, 16000)
    # Given that `prefill` is True,
    # the shape of `logits` should be the same as that of `res[current_index]` in model.forward
    prefill = True
    logits = model_runner.forward(input_ids=input_ids, valid_length_each_example=valid_length_each_example,
                                  block_tables=block_tables, slot_mapping=slot_mapping, prefill=prefill)
    assert logits.shape == (1, 16000)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
