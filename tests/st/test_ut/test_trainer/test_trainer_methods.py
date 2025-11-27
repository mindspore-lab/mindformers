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
"""test trainer methods."""
import os
import tempfile
import unittest
from copy import deepcopy
from contextlib import ExitStack
from unittest.mock import patch, MagicMock
from collections import OrderedDict
import pytest
import yaml

try:
    from mindspore.train import Callback
except ImportError:
    # Fallback for testing environments
    class Callback:
        """Dummy Callback class for testing."""

from mindformers import Trainer, MindFormerConfig
from mindformers.trainer.trainer import _reset_config_for_save, _save_config_to_yaml
from mindformers.models import PreTrainedTokenizerBase


# pylint: disable=W0212
class TestTrainerCallbackMethods(unittest.TestCase):
    """Test Trainer callback related methods."""

    @classmethod
    def setUpClass(cls):
        cls._exit_stack = ExitStack()
        cls.temp_dir = cls._exit_stack.enter_context(tempfile.TemporaryDirectory())
        # Load config from yaml file
        config_path = os.path.join(os.path.dirname(__file__), 'test_trainer_config.yaml')
        cls.config = MindFormerConfig(config_path)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_add_callback_with_class(self):
        """test add_callback with callback class."""
        trainer = Trainer(args=self.config, task='general', model_name='common')

        class DummyCallback(Callback):
            """Dummy callback for testing."""

        initial_len = len(trainer.callbacks)
        trainer.add_callback(DummyCallback)
        assert len(trainer.callbacks) == initial_len + 1
        assert isinstance(trainer.callbacks[-1], DummyCallback)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_add_callback_with_instance(self):
        """test add_callback with callback instance."""
        trainer = Trainer(args=self.config, task='general', model_name='common')

        class DummyCallback(Callback):
            """Dummy callback for testing."""

        callback_instance = DummyCallback()
        initial_len = len(trainer.callbacks)
        trainer.add_callback(callback_instance)
        assert len(trainer.callbacks) == initial_len + 1
        assert trainer.callbacks[-1] == callback_instance

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_add_duplicate_callback(self):
        """test add_callback with duplicate callback type."""
        trainer = Trainer(args=self.config, task='general', model_name='common')

        class DummyCallback(Callback):
            """Dummy callback for testing."""

        trainer.add_callback(DummyCallback)
        initial_len = len(trainer.callbacks)
        # Adding duplicate should still add but warn
        trainer.add_callback(DummyCallback)
        assert len(trainer.callbacks) == initial_len + 1

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_pop_callback_with_class(self):
        """test pop_callback with callback class."""
        trainer = Trainer(args=self.config, task='general', model_name='common')

        class DummyCallback(Callback):
            """Dummy callback for testing."""

        trainer.add_callback(DummyCallback)
        initial_len = len(trainer.callbacks)
        popped = trainer.pop_callback(DummyCallback)
        assert isinstance(popped, DummyCallback)
        assert len(trainer.callbacks) == initial_len - 1

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_pop_callback_with_instance(self):
        """test pop_callback with callback instance."""
        trainer = Trainer(args=self.config, task='general', model_name='common')

        class DummyCallback(Callback):
            """Dummy callback for testing."""

        callback_instance = DummyCallback()
        trainer.add_callback(callback_instance)
        initial_len = len(trainer.callbacks)
        popped = trainer.pop_callback(callback_instance)
        assert popped == callback_instance
        assert len(trainer.callbacks) == initial_len - 1

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_pop_callback_not_found(self):
        """test pop_callback when callback not in list."""
        trainer = Trainer(args=self.config, task='general', model_name='common')

        class DummyCallback(Callback):
            """Dummy callback for testing."""

        result = trainer.pop_callback(DummyCallback)
        assert result == DummyCallback

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_remove_callback_with_class(self):
        """test remove_callback with callback class."""
        trainer = Trainer(args=self.config, task='general', model_name='common')

        class DummyCallback(Callback):
            """Dummy callback for testing."""

        trainer.add_callback(DummyCallback)
        initial_len = len(trainer.callbacks)
        trainer.remove_callback(DummyCallback)
        assert len(trainer.callbacks) == initial_len - 1

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_remove_callback_with_instance(self):
        """test remove_callback with callback instance."""
        trainer = Trainer(args=self.config, task='general', model_name='common')

        class DummyCallback(Callback):
            """Dummy callback for testing."""

        callback_instance = DummyCallback()
        trainer.add_callback(callback_instance)
        initial_len = len(trainer.callbacks)
        trainer.remove_callback(callback_instance)
        assert len(trainer.callbacks) == initial_len - 1

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_callback_list_property(self):
        """test callback_list property."""
        trainer = Trainer(args=self.config, task='general', model_name='common')

        class DummyCallback1(Callback):
            """Dummy callback 1 for testing."""

        class DummyCallback2(Callback):
            """Dummy callback 2 for testing."""

        trainer.add_callback(DummyCallback1)
        trainer.add_callback(DummyCallback2)
        callback_list = trainer.callback_list
        assert isinstance(callback_list, str)
        assert 'DummyCallback1' in callback_list
        assert 'DummyCallback2' in callback_list

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        if hasattr(cls, '_exit_stack'):
            cls._exit_stack.close()


class TestTrainerParallelConfig(unittest.TestCase):
    """Test Trainer parallel configuration methods."""

    @classmethod
    def setUpClass(cls):
        cls._exit_stack = ExitStack()
        cls.temp_dir = cls._exit_stack.enter_context(tempfile.TemporaryDirectory())
        config_path = os.path.join(os.path.dirname(__file__), 'test_trainer_config.yaml')
        cls.config = MindFormerConfig(config_path)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_set_parallel_config_basic(self):
        """test set_parallel_config with basic parameters."""
        trainer = Trainer(args=self.config, task='general', model_name='common')
        trainer.set_parallel_config(
            data_parallel=2,
            model_parallel=2,
            pipeline_stage=1
        )
        assert trainer.config.parallel_config.data_parallel == 2
        assert trainer.config.parallel_config.model_parallel == 2
        assert trainer.config.parallel_config.pipeline_stage == 1
        assert trainer.reset_model is True

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_set_parallel_config_all_params(self):
        """test set_parallel_config with all parameters."""
        trainer = Trainer(args=self.config, task='general', model_name='common')
        trainer.set_parallel_config(
            data_parallel=2,
            model_parallel=2,
            context_parallel=1,
            expert_parallel=1,
            pipeline_stage=2,
            micro_batch_interleave_num=2,
            micro_batch_num=4,
            use_seq_parallel=True,
            optimizer_shard=True,
            gradient_aggregation_group=8,
            vocab_emb_dp=False
        )
        assert trainer.config.parallel_config.data_parallel == 2
        assert trainer.config.parallel_config.model_parallel == 2
        assert trainer.config.parallel_config.context_parallel == 1
        assert trainer.config.parallel_config.expert_parallel == 1
        assert trainer.config.parallel_config.pipeline_stage == 2
        assert trainer.config.parallel_config.use_seq_parallel is True
        assert trainer.config.parallel_config.optimizer_shard is True
        assert trainer.config.parallel_config.micro_batch_num == 4
        assert trainer.config.parallel_config.vocab_emb_dp is False
        assert trainer.config.parallel_config.gradient_aggregation_group == 8
        assert trainer.config.micro_batch_interleave_num == 2

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_set_parallel_config_default_values(self):
        """test set_parallel_config with default values."""
        trainer = Trainer(args=self.config, task='general', model_name='common')
        trainer.set_parallel_config()
        assert trainer.config.parallel_config.data_parallel == 1
        assert trainer.config.parallel_config.model_parallel == 1
        assert trainer.config.parallel_config.pipeline_stage == 1
        assert trainer.config.parallel_config.use_seq_parallel is False
        assert trainer.config.parallel_config.optimizer_shard is False

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        if hasattr(cls, '_exit_stack'):
            cls._exit_stack.close()


class TestTrainerRecomputeConfig(unittest.TestCase):
    """Test Trainer recompute configuration methods."""

    @classmethod
    def setUpClass(cls):
        cls._exit_stack = ExitStack()
        cls.temp_dir = cls._exit_stack.enter_context(tempfile.TemporaryDirectory())
        config_path = os.path.join(os.path.dirname(__file__), 'test_trainer_config.yaml')
        cls.config = MindFormerConfig(config_path)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_set_recompute_config_basic(self):
        """test set_recompute_config with basic parameters."""
        trainer = Trainer(args=self.config, task='general', model_name='common')
        trainer.set_recompute_config(recompute=True)
        assert trainer.config.recompute_config.recompute is True
        assert trainer.reset_model is True

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_set_recompute_config_all_params(self):
        """test set_recompute_config with all parameters."""
        trainer = Trainer(args=self.config, task='general', model_name='common')
        trainer.set_recompute_config(
            recompute=True,
            parallel_optimizer_comm_recompute=True,
            select_recompute=True,
            mp_comm_recompute=False,
            recompute_slice_activation=True
        )
        assert trainer.config.recompute_config.recompute is True
        assert trainer.config.recompute_config.parallel_optimizer_comm_recompute is True
        assert trainer.config.recompute_config.select_recompute is True
        assert trainer.config.recompute_config.mp_comm_recompute is False
        assert trainer.config.recompute_config.recompute_slice_activation is True

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_set_recompute_config_default_values(self):
        """test set_recompute_config with default values."""
        trainer = Trainer(args=self.config, task='general', model_name='common')
        trainer.set_recompute_config()
        assert trainer.config.recompute_config.recompute is False
        assert trainer.config.recompute_config.parallel_optimizer_comm_recompute is False
        assert trainer.config.recompute_config.select_recompute is False
        assert trainer.config.recompute_config.mp_comm_recompute is True
        assert trainer.config.recompute_config.recompute_slice_activation is False

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        if hasattr(cls, '_exit_stack'):
            cls._exit_stack.close()


class TestTrainerMoeConfig(unittest.TestCase):
    """Test Trainer MoE configuration methods."""

    @classmethod
    def setUpClass(cls):
        cls._exit_stack = ExitStack()
        cls.temp_dir = cls._exit_stack.enter_context(tempfile.TemporaryDirectory())
        config_path = os.path.join(os.path.dirname(__file__), 'test_trainer_config.yaml')
        cls.config = MindFormerConfig(config_path)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_set_moe_config_basic(self):
        """test _set_moe_config with basic parameters."""
        trainer = Trainer(args=self.config, task='general', model_name='common')
        trainer._set_moe_config(expert_num=4)
        assert trainer.config.moe_config.expert_num == 4
        assert trainer.reset_model is True

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_set_moe_config_all_params(self):
        """test _set_moe_config with all parameters."""
        trainer = Trainer(args=self.config, task='general', model_name='common')
        trainer._set_moe_config(
            expert_num=8,
            capacity_factor=1.5,
            aux_loss_factor=0.1,
            num_experts_chosen=2,
            expert_group_size=128,
            group_wise_a2a=True,
            comp_comm_parallel=True,
            comp_comm_parallel_degree=4
        )
        assert trainer.config.moe_config.expert_num == 8
        assert trainer.config.moe_config.capacity_factor == 1.5
        assert trainer.config.moe_config.aux_loss_factor == 0.1
        assert trainer.config.moe_config.num_experts_chosen == 2
        assert trainer.config.moe_config.expert_group_size == 128
        assert trainer.config.moe_config.group_wise_a2a is True
        assert trainer.config.moe_config.comp_comm_parallel is True
        assert trainer.config.moe_config.comp_comm_parallel_degree == 4

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_set_moe_config_default_values(self):
        """test _set_moe_config with default values."""
        trainer = Trainer(args=self.config, task='general', model_name='common')
        trainer._set_moe_config()
        assert trainer.config.moe_config.expert_num == 1
        assert trainer.config.moe_config.capacity_factor == 1.1
        assert trainer.config.moe_config.aux_loss_factor == 0.05
        assert trainer.config.moe_config.num_experts_chosen == 1
        assert trainer.config.moe_config.group_wise_a2a is False
        assert trainer.config.moe_config.comp_comm_parallel is False
        assert trainer.config.moe_config.comp_comm_parallel_degree == 2

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        if hasattr(cls, '_exit_stack'):
            cls._exit_stack.close()


class TestTrainerDataloaderMethods(unittest.TestCase):
    """Test Trainer dataloader related methods."""

    @classmethod
    def setUpClass(cls):
        cls._exit_stack = ExitStack()
        cls.temp_dir = cls._exit_stack.enter_context(tempfile.TemporaryDirectory())
        config_path = os.path.join(os.path.dirname(__file__), 'test_trainer_config.yaml')
        cls.config = MindFormerConfig(config_path)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_train_dataloader(self):
        """test get_train_dataloader method."""
        trainer = Trainer(args=self.config, task='general', model_name='common')
        # Mock the config to avoid actual dataloader building
        with patch('mindformers.trainer.trainer.build_dataset_loader') as mock_build:
            mock_build.return_value = MagicMock()
            dataloader = trainer.get_train_dataloader()
            assert dataloader is not None
            mock_build.assert_called_once()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_eval_dataloader(self):
        """test get_eval_dataloader method."""
        trainer = Trainer(args=self.config, task='general', model_name='common')
        # Mock the config to avoid actual dataloader building
        with patch('mindformers.trainer.trainer.build_dataset_loader') as mock_build:
            mock_build.return_value = MagicMock()
            dataloader = trainer.get_eval_dataloader()
            assert dataloader is not None
            mock_build.assert_called_once()

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        if hasattr(cls, '_exit_stack'):
            cls._exit_stack.close()


class TestTrainerCheckpointMethods(unittest.TestCase):
    """Test Trainer checkpoint related methods."""

    @classmethod
    def setUpClass(cls):
        cls.config_path = os.path.join(os.path.dirname(__file__), 'test_trainer_config.yaml')
        cls.config = MindFormerConfig(cls.config_path)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_load_checkpoint_valid_path(self):
        """test get_load_checkpoint with valid checkpoint path."""
        # Create a temporary checkpoint file
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt_path = os.path.join(temp_dir, 'test_checkpoint.ckpt')
            with open(ckpt_path, 'w', encoding='utf-8') as f:
                f.write('mock checkpoint')
            os.stat(ckpt_path)

            result = Trainer.get_load_checkpoint(ckpt_path)
            assert result == ckpt_path

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_load_checkpoint_none(self):
        """test get_load_checkpoint with None."""
        result = Trainer.get_load_checkpoint(None)
        assert result is None

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_load_checkpoint_invalid_path(self):
        """test get_load_checkpoint with invalid path."""
        with pytest.raises(ValueError) as exc_info:
            Trainer.get_load_checkpoint('/nonexistent/path/checkpoint.ckpt')
        assert 'not existed' in str(exc_info.value)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_load_checkpoint_invalid_type(self):
        """test get_load_checkpoint with invalid type."""
        with pytest.raises(TypeError) as exc_info:
            Trainer.get_load_checkpoint(123)
        assert 'should be a str' in str(exc_info.value)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_check_checkpoint_config_with_true(self):
        """test _check_checkpoint_config with True."""
        trainer = Trainer(args=self.config, task='general', model_name='common')

        with tempfile.TemporaryDirectory() as temp_dir:
            trainer.config.output_dir = temp_dir
            checkpoint_dir = os.path.join(temp_dir, 'checkpoint', 'rank_0')
            os.makedirs(checkpoint_dir, exist_ok=True)

            with open(os.path.join(checkpoint_dir, "1.safetensors"), 'w', encoding='utf-8') as f:
                f.write('mock')
            os.stat(os.path.join(checkpoint_dir, "1.safetensors"))

            with open(os.path.join(checkpoint_dir, "2.safetensors"), 'w', encoding='utf-8') as f:
                f.write('mock')
            os.stat(os.path.join(checkpoint_dir, "2.safetensors"))

            last_checkpoint_path = os.path.join(checkpoint_dir, "3.safetensors")
            with open(last_checkpoint_path, 'w', encoding='utf-8') as f:
                f.write('mock')
            os.stat(last_checkpoint_path)

            trainer._check_checkpoint_config(True)
            assert trainer.config.model.model_config.checkpoint_name_or_path == last_checkpoint_path

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_check_checkpoint_config_with_path(self):
        """test _check_checkpoint_config with checkpoint path."""
        trainer = Trainer(args=self.config, task='general', model_name='common')
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt_path = os.path.join(temp_dir, 'test.ckpt')
            with open(ckpt_path, 'w', encoding='utf-8') as f:
                f.write('mock')
            os.stat(ckpt_path)

            trainer._check_checkpoint_config(ckpt_path)
            assert trainer.config.model.model_config.checkpoint_name_or_path == ckpt_path

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_check_checkpoint_config_with_none(self):
        """test _check_checkpoint_config with None."""
        trainer = Trainer(args=self.config, task='general', model_name='common')
        trainer.default_checkpoint_name_or_path = 'default_ckpt'
        trainer._check_checkpoint_config(None)
        assert trainer.config.model.model_config.checkpoint_name_or_path == 'default_ckpt'


class TestTrainerConfigMethods(unittest.TestCase):
    """Test Trainer config related methods."""

    @classmethod
    def setUpClass(cls):
        cls._exit_stack = ExitStack()
        cls.temp_dir = cls._exit_stack.enter_context(tempfile.TemporaryDirectory())

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_config_init_with_none(self):
        """test _config_init with None args."""
        result = Trainer._config_init(None, None)
        assert isinstance(result, MindFormerConfig)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_config_init_with_mindformer_config(self):
        """test _config_init with MindFormerConfig."""
        config = MindFormerConfig()
        result = Trainer._config_init(config, None)
        assert result == config

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_config_init_with_yaml_path(self):
        """test _config_init with yaml path."""
        # Create a temporary yaml file
        yaml_path = os.path.join(self.temp_dir, 'test_config.yaml')
        config_dict = {
            'model': {'model_config': {'type': 'test_model'}},
            'trainer': {'type': 'test_trainer'}
        }
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f)

        result = Trainer._config_init(yaml_path, None)
        assert isinstance(result, MindFormerConfig)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_config_init_with_invalid_yaml_path(self):
        """test _config_init with invalid yaml path."""
        with pytest.raises(ValueError) as exc_info:
            Trainer._config_init('/nonexistent/config.yaml', None)
        assert 'must be exist' in str(exc_info.value)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_config_init_with_invalid_extension(self):
        """test _config_init with invalid file extension."""
        txt_path = os.path.join(self.temp_dir, 'config.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('test')

        with pytest.raises(ValueError) as exc_info:
            Trainer._config_init(txt_path, None)
        assert 'must be end with .yaml or .yml' in str(exc_info.value)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        if hasattr(cls, '_exit_stack'):
            cls._exit_stack.close()


class TestTrainerInitMethods(unittest.TestCase):
    """Test Trainer initialization related methods."""

    @classmethod
    def setUpClass(cls):
        cls._exit_stack = ExitStack()
        cls.temp_dir = cls._exit_stack.enter_context(tempfile.TemporaryDirectory())
        config_path = os.path.join(os.path.dirname(__file__), 'test_trainer_config.yaml')
        cls.config = MindFormerConfig(config_path)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_init_tokenizer(self):
        """test _init_tokenizer method."""
        trainer = Trainer(args=self.config, task='general', model_name='common')
        mock_tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
        trainer.tokenizer = mock_tokenizer
        trainer._init_tokenizer()
        # Verify tokenizer is set in config
        assert trainer.config.train_dataset.tokenizer == mock_tokenizer
        assert trainer.config.eval_dataset.tokenizer == mock_tokenizer

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_init_tokenizer_none(self):
        """test _init_tokenizer with None tokenizer."""
        trainer = Trainer(args=self.config, task='general', model_name='common')
        trainer.tokenizer = None
        trainer._init_tokenizer()
        # Should not raise error

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_init_dataset_with_train_path(self):
        """test _init_dataset with train dataset path."""
        trainer = Trainer(args=self.config, task='general', model_name='common')
        train_dir = os.path.join(self.temp_dir, 'train_data')
        os.makedirs(train_dir, exist_ok=True)
        trainer.train_dataset = train_dir
        trainer._init_dataset()
        assert trainer.config.train_dataset.data_loader.dataset_dir == train_dir
        assert trainer.train_dataset is None

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_init_dataset_with_eval_path(self):
        """test _init_dataset with eval dataset path."""
        trainer = Trainer(args=self.config, task='general', model_name='common')
        eval_dir = os.path.join(self.temp_dir, 'eval_data')
        os.makedirs(eval_dir, exist_ok=True)
        trainer.eval_dataset = eval_dir
        trainer._init_dataset()
        assert trainer.config.eval_dataset.data_loader.dataset_dir == eval_dir
        assert trainer.eval_dataset is None

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_init_dataset_invalid_train_path(self):
        """test _init_dataset with invalid train path."""
        trainer = Trainer(args=self.config, task='general', model_name='common')
        trainer.train_dataset = '/nonexistent/train/path'
        with pytest.raises(ValueError) as exc_info:
            trainer._init_dataset()
        assert 'must be exist' in str(exc_info.value)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_init_dataset_invalid_eval_path(self):
        """test _init_dataset with invalid eval path."""
        trainer = Trainer(args=self.config, task='general', model_name='common')
        trainer.eval_dataset = '/nonexistent/eval/path'
        with pytest.raises(ValueError) as exc_info:
            trainer._init_dataset()
        assert 'must be exist' in str(exc_info.value)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_build_profile_cb_enabled(self):
        """test _build_profile_cb when profile is enabled."""
        config = deepcopy(self.config)
        config.profile = True
        config.profile_start_step = 10
        config.profile_stop_step = 20
        config.init_start_profile = False
        config.profile_communication = False
        config.profile_memory = False
        config.profile_output = self.temp_dir
        config.profiler_level = 0
        config.with_stack = False
        config.data_simplification = True
        config.runner_config.sink_size = 1
        config.mstx = False
        config.runner_config.sink_mode = False
        trainer = Trainer(args=config, task='general', model_name='common')


        trainer._build_profile_cb()
        assert hasattr(trainer.config, 'profile_cb')
        assert trainer.config.profile_cb is not None
        assert trainer.config.auto_tune is False

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_build_profile_cb_disabled(self):
        """test _build_profile_cb when profile is disabled."""
        config = deepcopy(self.config)
        config.profile = False
        trainer = Trainer(args=config, task='general', model_name='common')
        trainer._build_profile_cb()
        assert not hasattr(trainer.config, 'profile_cb') or trainer.config.profile_cb is None

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        if hasattr(cls, '_exit_stack'):
            cls._exit_stack.close()


class TestTrainerSaveMethods(unittest.TestCase):
    """Test Trainer save related methods."""

    @classmethod
    def setUpClass(cls):
        cls._exit_stack = ExitStack()
        cls.temp_dir = cls._exit_stack.enter_context(tempfile.TemporaryDirectory())
        config_path = os.path.join(os.path.dirname(__file__), 'test_trainer_config.yaml')
        cls.config = MindFormerConfig(config_path)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_save_config_to_yaml_function(self):
        """test _save_config_to_yaml function."""
        save_path = os.path.join(self.temp_dir, 'test_config.yaml')
        config_dict = {'model': {'type': 'test'}, 'trainer': {'type': 'test'}}

        _save_config_to_yaml(save_path, config_dict)
        assert os.path.exists(save_path)

        # Verify content
        with open(save_path, 'r', encoding='utf-8') as f:
            loaded = yaml.safe_load(f)
        assert loaded['model']['type'] == 'test'

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_save_model_method(self):
        """test save_model method."""
        trainer = Trainer(args=self.config, task='general', model_name='common')
        save_dir = os.path.join(self.temp_dir, 'model_save')
        os.makedirs(save_dir, exist_ok=True)

        with patch.object(trainer, '_save') as mock_save:
            trainer.save_model(output_dir=save_dir)
            mock_save.assert_called_once()

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        if hasattr(cls, '_exit_stack'):
            cls._exit_stack.close()


class TestTrainerCheckMethods(unittest.TestCase):
    """Test Trainer check related methods."""

    @classmethod
    def setUpClass(cls):
        cls._exit_stack = ExitStack()
        cls.temp_dir = cls._exit_stack.enter_context(tempfile.TemporaryDirectory())
        config_path = os.path.join(os.path.dirname(__file__), 'test_trainer_config.yaml')
        cls.config = MindFormerConfig(config_path)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_check_args_task_and_model_general_common(self):
        """test _check_args_task_and_model with general task and common model."""
        trainer = Trainer(args=self.config, task='general', model_name='common')
        # Should not raise error
        trainer._check_args_task_and_model()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_check_args_task_and_model_invalid_model_name(self):
        """test _check_args_task_and_model with invalid model name for task."""
        with pytest.raises(ValueError) as exc_info:
            Trainer(task='text_generation', model_name='invalid_model')
        assert 'not support in task' in str(exc_info.value)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_load_model_checkpoint_none(self):
        """test _load_model_checkpoint with None checkpoint."""
        trainer = Trainer(args=self.config, task='general', model_name='common')
        trainer.config.model.model_config.checkpoint_name_or_path = None
        trainer._load_model_checkpoint()
        # Should not raise error

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_load_model_checkpoint_invalid_type(self):
        """test _load_model_checkpoint with invalid type."""
        trainer = Trainer(args=self.config, task='general', model_name='common')
        trainer.config.model.model_config.checkpoint_name_or_path = 123
        with pytest.raises(TypeError) as exc_info:
            trainer._load_model_checkpoint()
        assert 'type error' in str(exc_info.value)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        if hasattr(cls, '_exit_stack'):
            cls._exit_stack.close()


class TestResetConfigForSave(unittest.TestCase):
    """Test _reset_config_for_save function."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_reset_config_for_save_with_model(self):
        """test _reset_config_for_save with model config."""
        config = MindFormerConfig()
        config.model = MindFormerConfig({'type': 'test_model'})
        result = _reset_config_for_save(config)
        assert isinstance(result, OrderedDict)
        assert 'model' in result

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_reset_config_for_save_with_processor(self):
        """test _reset_config_for_save with processor config."""
        config = MindFormerConfig()
        config.processor = MindFormerConfig({'type': 'test_processor'})
        result = _reset_config_for_save(config)
        assert isinstance(result, OrderedDict)
        assert 'processor' in result

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_reset_config_for_save_with_datasets(self):
        """test _reset_config_for_save with dataset configs."""
        config = MindFormerConfig()
        config.train_dataset = MindFormerConfig({'type': 'train_dataset'})
        config.train_dataset_task = MindFormerConfig({'type': 'train_task'})
        config.eval_dataset = MindFormerConfig({'type': 'eval_dataset'})
        config.eval_dataset_task = MindFormerConfig({'type': 'eval_task'})
        result = _reset_config_for_save(config)
        assert 'train_dataset' in result
        assert 'train_dataset_task' in result
        assert 'eval_dataset' in result
        assert 'eval_dataset_task' in result

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_reset_config_for_save_with_parallel_configs(self):
        """test _reset_config_for_save with parallel configs."""
        config = MindFormerConfig()
        config.context = MindFormerConfig({'mode': 'GRAPH_MODE'})
        config.parallel = MindFormerConfig({'parallel_mode': 'semi_auto_parallel'})
        config.moe_config = MindFormerConfig({'expert_num': 4})
        config.recompute_config = MindFormerConfig({'recompute': True})
        config.parallel_config = MindFormerConfig({'data_parallel': 2})
        result = _reset_config_for_save(config)
        assert 'context' in result
        assert 'parallel' in result
        assert 'moe_config' in result
        assert 'recompute_config' in result
        assert 'parallel_config' in result

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_reset_config_for_save_with_optimizer_lr(self):
        """test _reset_config_for_save with optimizer and lr configs."""
        config = MindFormerConfig()
        config.optimizer = MindFormerConfig({'type': 'AdamW'})
        config.lr_schedule = MindFormerConfig({'type': 'cosine'})
        result = _reset_config_for_save(config)
        assert 'optimizer' in result
        assert 'lr_schedule' in result

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_reset_config_for_save_with_callbacks(self):
        """test _reset_config_for_save with callbacks config."""
        config = MindFormerConfig()
        config.callbacks = [{'type': 'CheckpointMonitor'}]
        result = _reset_config_for_save(config)
        assert 'callbacks' in result

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_reset_config_for_save_with_runner_config(self):
        """test _reset_config_for_save with runner config."""
        config = MindFormerConfig()
        config.runner_config = MindFormerConfig({'sink_mode': True})
        result = _reset_config_for_save(config)
        assert 'runner_config' in result

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_reset_config_for_save_with_runner_wrapper(self):
        """test _reset_config_for_save with runner wrapper config."""
        config = MindFormerConfig()
        config.runner_wrapper = MindFormerConfig({'type': 'TrainOneStepCell'})
        result = _reset_config_for_save(config)
        assert 'runner_wrapper' in result

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_reset_config_for_save_empty_config(self):
        """test _reset_config_for_save with empty config."""
        config = MindFormerConfig()
        result = _reset_config_for_save(config)
        assert isinstance(result, OrderedDict)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_reset_config_for_save_none_config(self):
        """test _reset_config_for_save with None."""
        result = _reset_config_for_save(None)
        assert isinstance(result, OrderedDict)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_reset_config_for_save_ordered_dict(self):
        """test _reset_config_for_save returns OrderedDict."""
        config = MindFormerConfig()
        config.model = MindFormerConfig({'type': 'test'})
        config.optimizer = MindFormerConfig({'type': 'AdamW'})
        result = _reset_config_for_save(config)
        assert isinstance(result, OrderedDict)
        # Check order: model should come before optimizer
        keys_list = list(result.keys())
        if 'model' in keys_list and 'optimizer' in keys_list:
            assert keys_list.index('model') < keys_list.index('optimizer')
