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
"""Unit tests for Trainer public interfaces: train() and get_batch()."""
# pylint: disable=protected-access
import os
import sys
from unittest.mock import Mock, patch

import pytest


# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, project_root)

# Mock mindspore modules FIRST before any mindformers imports
mindspore_mock = Mock()
mindspore_mock.nn = Mock()
mindspore_mock.dataset = Mock()
mindspore_mock.common = Mock()
mindspore_mock.common.tensor = Mock()
mindspore_mock.common.initializer = Mock()
mindspore_mock.common.parameter = Mock()
mindspore_mock.common.dtype = Mock()
mindspore_mock.ops = Mock()
mindspore_mock.context = Mock()

sys.modules['mindspore'] = mindspore_mock
sys.modules['mindspore.nn'] = mindspore_mock.nn
sys.modules['mindspore.dataset'] = mindspore_mock.dataset
sys.modules['mindspore.common'] = mindspore_mock.common
sys.modules['mindspore.common.tensor'] = mindspore_mock.common.tensor
sys.modules['mindspore.common.initializer'] = mindspore_mock.common.initializer
sys.modules['mindspore.common.parameter'] = mindspore_mock.common.parameter
sys.modules['mindspore.common.dtype'] = mindspore_mock.common.dtype
sys.modules['mindspore.ops'] = mindspore_mock.ops
sys.modules['mindspore.context'] = mindspore_mock.context
sys.modules['mindspore.parallel'] = Mock()
sys.modules['mindspore.train'] = Mock()

# Mock mindspore.communication
communication_mock = Mock()
communication_mock.management = Mock()
communication_mock.management.get_rank = Mock(return_value=0)
communication_mock.management.get_group_size = Mock(return_value=1)
sys.modules['mindspore.communication'] = communication_mock
sys.modules['mindspore.communication.management'] = communication_mock.management

sys.modules['mindspore._checkparam'] = Mock()
sys.modules['mindspore.amp'] = Mock()
sys.modules['mindspore._c_expression'] = Mock()

# Mock mindformers modules
modules_mock = Mock()
modules_mock.transformer = Mock()
modules_mock.__all__ = []
sys.modules['mindformers.modules'] = modules_mock
sys.modules['mindformers.modules.transformer'] = modules_mock.transformer

checkpoint_mock = Mock()
checkpoint_mock.__all__ = []
checkpoint_mock.checkpoint = Mock()
sys.modules['mindformers.checkpoint'] = checkpoint_mock
sys.modules['mindformers.checkpoint.checkpoint'] = checkpoint_mock.checkpoint

models_mock = Mock()
models_mock.llama = Mock()
models_mock.__all__ = []
sys.modules['mindformers.models'] = models_mock
sys.modules['mindformers.models.llama'] = models_mock.llama

dataset_mock = Mock()
dataset_mock.__all__ = []
sys.modules['mindformers.dataset'] = dataset_mock

sys.modules['mindformers.run_check'] = Mock()

core_mock = Mock()
core_mock.context = Mock()
core_mock.__all__ = []
sys.modules['mindformers.core'] = core_mock
sys.modules['mindformers.core.context'] = core_mock.context
sys.modules['mindformers.core.config_args'] = Mock()
sys.modules['mindformers.core.lr'] = Mock()
sys.modules['mindformers.core.optim'] = Mock()
sys.modules['mindformers.core.callback'] = Mock()
sys.modules['mindformers.core.callback_pynative'] = Mock()
sys.modules['mindformers.core.metric'] = Mock()

pet_mock = Mock()
pet_mock.__all__ = []
sys.modules['mindformers.pet'] = pet_mock

wrapper_mock = Mock()
wrapper_mock.__all__ = []
sys.modules['mindformers.wrapper'] = wrapper_mock

generation_mock = Mock()
generation_mock.__all__ = []
sys.modules['mindformers.generation'] = generation_mock

pipeline_mock = Mock()
pipeline_mock.__all__ = []
sys.modules['mindformers.pipeline'] = pipeline_mock

trainer_mock = Mock()
trainer_mock.__all__ = []
trainer_mock.training_args = Mock()
trainer_mock.optimizer_grouped_parameters = Mock()
sys.modules['mindformers.trainer'] = trainer_mock
sys.modules['mindformers.trainer.training_args'] = trainer_mock.training_args
sys.modules['mindformers.trainer.optimizer_grouped_parameters'] = trainer_mock.optimizer_grouped_parameters
sys.modules['mindformers.trainer.general_task_trainer'] = Mock()

sys.modules['mindformers.model_runner'] = Mock()

# Mock trainer.train_state
train_state_mock = Mock()
mock_trainer_state_class = Mock()
sys.modules['trainer'] = Mock()
sys.modules['trainer.train_state'] = train_state_mock
train_state_mock.TrainerState = mock_trainer_state_class

# Import real MindFormerConfig
from mindformers.tools.register import MindFormerConfig  # pylint: disable=wrong-import-position

# Now import Trainer
from mindformers.trainer_pynative.trainer import Trainer  # pylint: disable=wrong-import-position


class TestTrainerTrain:
    """Test cases for Trainer.train() interface."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock MindFormerConfig."""
        config = MindFormerConfig()
        config.max_steps = 100
        config.eval_steps = 20
        config.save_steps = 50
        config.global_batch_size = 32
        return config

    @pytest.fixture
    def mock_trainer_state(self):
        """Create a mock TrainerState."""
        state = Mock()
        state.global_step = 0
        state.epoch_step = 10
        state.max_steps = 100
        state.eval_steps = 20
        state.save_steps = 50
        state.global_batch_size = 32
        state.update_epoch = Mock()
        return state

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock()
        model.__call__ = Mock(return_value={'loss': 0.5})
        return model

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset."""
        dataset = Mock()
        dataset.__len__ = Mock(return_value=10)
        dataset.get_dataset_size = Mock(return_value=10)

        # Create mock iterator
        mock_iter = Mock()
        mock_iter.__next__ = Mock(side_effect=[
            {'input_ids': [1, 2, 3], 'labels': [2, 3, 4]},
            {'input_ids': [4, 5, 6], 'labels': [5, 6, 7]},
        ] * 100)  # Repeat to avoid StopIteration during tests

        dataset.create_dict_iterator = Mock(return_value=mock_iter)
        return dataset

    @pytest.fixture
    def mock_optimizer(self):
        """Create a mock optimizer."""
        optimizer = Mock()
        optimizer.step = Mock()
        return optimizer

    @pytest.fixture
    def mock_callback_handler(self):
        """Create a mock CallbackHandler."""
        handler = Mock()
        handler.on_train_begin = Mock()
        handler.on_train_end = Mock()
        handler.on_epoch_begin = Mock()
        handler.on_epoch_end = Mock()
        handler.on_step_begin = Mock()
        handler.on_step_end = Mock()
        return handler

    def test_train_pretrain_mode_success(
        self, mock_config, mock_model, mock_dataset,
        mock_optimizer, mock_callback_handler, mock_trainer_state
    ):
        """Test train() in pretrain mode executes successfully."""
        # Create trainer instance with mocked components
        trainer = Trainer.__new__(Trainer)
        trainer.config = mock_config
        trainer.model = mock_model
        trainer.train_dataset = mock_dataset
        trainer.eval_dataset = None
        trainer.optimizer = mock_optimizer
        trainer.lr_scheduler = Mock()
        trainer.callback_handler = mock_callback_handler
        trainer.compute_metrics = None
        trainer.compute_loss_func = None
        trainer.processing_class = None

        # Mock internal methods
        trainer._init_parallel_config = Mock()
        trainer._load_checkpoint = Mock()
        trainer._get_dataset_size = Mock(return_value=10)
        trainer._inner_train_loop = Mock()

        # Mock TrainerState
        with patch('trainer.train_state.TrainerState', return_value=mock_trainer_state):
            # Execute train
            trainer.train(checkpoint_path=None, mode="pretrain", do_eval=False)

        # Verify method calls
        trainer._init_parallel_config.assert_called_once()
        trainer._load_checkpoint.assert_not_called()  # No checkpoint in pretrain mode
        mock_callback_handler.on_train_begin.assert_called_once()
        mock_callback_handler.on_train_end.assert_called_once()
        trainer._inner_train_loop.assert_called_once_with(False)

    def test_train_finetune_mode_requires_checkpoint(
        self, mock_config, mock_model, mock_dataset,
        mock_optimizer, mock_callback_handler
    ):
        """Test train() in finetune mode raises error without checkpoint."""
        trainer = Trainer.__new__(Trainer)
        trainer.config = mock_config
        trainer.model = mock_model
        trainer.train_dataset = mock_dataset
        trainer.eval_dataset = None
        trainer.optimizer = mock_optimizer
        trainer.lr_scheduler = Mock()
        trainer.callback_handler = mock_callback_handler
        trainer.compute_metrics = None
        trainer.compute_loss_func = None
        trainer.processing_class = None

        trainer._init_parallel_config = Mock()

        # Should raise ValueError when checkpoint_path is None in finetune mode
        with pytest.raises(ValueError, match="checkpoint_path cannot be None"):
            trainer.train(checkpoint_path=None, mode="finetune", do_eval=False)

    def test_train_finetune_mode_with_checkpoint(
        self, mock_config, mock_model, mock_dataset,
        mock_optimizer, mock_callback_handler, mock_trainer_state
    ):
        """Test train() in finetune mode with checkpoint loads correctly."""
        trainer = Trainer.__new__(Trainer)
        trainer.config = mock_config
        trainer.model = mock_model
        trainer.train_dataset = mock_dataset
        trainer.eval_dataset = None
        trainer.optimizer = mock_optimizer
        trainer.lr_scheduler = Mock()
        trainer.callback_handler = mock_callback_handler
        trainer.compute_metrics = None
        trainer.compute_loss_func = None
        trainer.processing_class = None

        trainer._init_parallel_config = Mock()
        trainer._load_checkpoint = Mock()
        trainer._get_dataset_size = Mock(return_value=10)
        trainer._inner_train_loop = Mock()

        checkpoint_path = "/mock/checkpoint.ckpt"

        with patch('trainer.train_state.TrainerState', return_value=mock_trainer_state):
            with patch('os.path.exists', return_value=True):
                trainer.train(checkpoint_path=checkpoint_path, mode="finetune", do_eval=False)

        # Verify checkpoint loading
        trainer._load_checkpoint.assert_called_once_with(checkpoint_path, "finetune")

    def test_train_invalid_mode_raises_error(
        self, mock_config, mock_model, mock_dataset,
        mock_optimizer, mock_callback_handler
    ):
        """Test train() raises error with invalid mode."""
        trainer = Trainer.__new__(Trainer)
        trainer.config = mock_config
        trainer.model = mock_model
        trainer.train_dataset = mock_dataset
        trainer.optimizer = mock_optimizer
        trainer.callback_handler = mock_callback_handler

        trainer._init_parallel_config = Mock()

        with pytest.raises(ValueError, match="mode must be 'pretrain' or 'finetune'"):
            trainer.train(checkpoint_path=None, mode="invalid_mode", do_eval=False)

    def test_train_calls_callbacks_correctly(
        self, mock_config, mock_model, mock_dataset,
        mock_optimizer, mock_callback_handler, mock_trainer_state
    ):
        """Test train() calls all callback hooks in correct order."""
        trainer = Trainer.__new__(Trainer)
        trainer.config = mock_config
        trainer.model = mock_model
        trainer.train_dataset = mock_dataset
        trainer.optimizer = mock_optimizer
        trainer.lr_scheduler = Mock()
        trainer.callback_handler = mock_callback_handler
        trainer.compute_loss_func = None

        trainer._init_parallel_config = Mock()
        trainer._load_checkpoint = Mock()
        trainer._get_dataset_size = Mock(return_value=10)
        trainer._inner_train_loop = Mock()

        with patch('trainer.train_state.TrainerState', return_value=mock_trainer_state):
            trainer.train(checkpoint_path=None, mode="pretrain", do_eval=False)

        # Verify callback call order
        assert mock_callback_handler.on_train_begin.called
        assert mock_callback_handler.on_train_end.called
        # on_train_begin should be called before on_train_end
        call_order = [
            call for call in mock_callback_handler.method_calls
            if call[0] in ['on_train_begin', 'on_train_end']
        ]
        assert call_order[0][0] == 'on_train_begin'
        assert call_order[-1][0] == 'on_train_end'

    def test_train_with_do_eval_true(
        self, mock_config, mock_model, mock_dataset,
        mock_optimizer, mock_callback_handler, mock_trainer_state
    ):
        """Test train() with do_eval=True passes flag to inner loop."""
        trainer = Trainer.__new__(Trainer)
        trainer.config = mock_config
        trainer.model = mock_model
        trainer.train_dataset = mock_dataset
        trainer.optimizer = mock_optimizer
        trainer.lr_scheduler = Mock()
        trainer.callback_handler = mock_callback_handler

        trainer._init_parallel_config = Mock()
        trainer._get_dataset_size = Mock(return_value=10)
        trainer._inner_train_loop = Mock()

        with patch('trainer.train_state.TrainerState', return_value=mock_trainer_state):
            trainer.train(checkpoint_path=None, mode="pretrain", do_eval=True)

        # Verify do_eval flag is passed
        trainer._inner_train_loop.assert_called_once_with(True)


class TestTrainerGetBatch:
    """Test cases for Trainer.get_batch() interface."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        config = Mock()
        config.use_distribute_dataset = False
        config.use_remove_redundant_dataset = False
        return config

    @pytest.fixture
    def mock_dataset_iter(self):
        """Create a mock dataset iterator."""
        iterator = Mock()
        iterator.__next__ = Mock(return_value={'input_ids': [1, 2, 3], 'labels': [2, 3, 4]})
        return iterator

    def test_get_batch_naive_mode_returns_dict(self, mock_config, mock_dataset_iter):
        """Test get_batch() in naive mode returns dict data."""
        trainer = Trainer.__new__(Trainer)
        trainer.config = mock_config

        # Execute get_batch
        batch = trainer.get_batch(mock_dataset_iter)

        # Verify return type and content
        assert isinstance(batch, dict)
        assert 'input_ids' in batch
        assert batch['input_ids'] == [1, 2, 3]

    def test_get_batch_distributed_mode(self, mock_config, mock_dataset_iter):
        """Test get_batch() in distributed mode."""
        mock_config.use_distribute_dataset = True

        trainer = Trainer.__new__(Trainer)
        trainer.config = mock_config

        batch = trainer.get_batch(mock_dataset_iter)

        assert isinstance(batch, dict)
        assert 'input_ids' in batch

    def test_get_batch_remove_redundant_mode(self, mock_config, mock_dataset_iter):
        """Test get_batch() in remove redundant mode."""
        mock_config.use_remove_redundant_dataset = True

        trainer = Trainer.__new__(Trainer)
        trainer.config = mock_config

        batch = trainer.get_batch(mock_dataset_iter)

        assert isinstance(batch, dict)
        assert 'input_ids' in batch

    def test_get_batch_handles_tuple_data(self, mock_config):
        """Test get_batch() converts tuple data to dict."""
        trainer = Trainer.__new__(Trainer)
        trainer.config = mock_config

        # Mock iterator returning tuple
        mock_iter = Mock()
        mock_iter.__next__ = Mock(return_value=([1, 2, 3], [2, 3, 4]))

        batch = trainer.get_batch(mock_iter)

        # Should convert tuple to dict with 'input_ids' key
        assert isinstance(batch, dict)
        assert 'input_ids' in batch

    def test_get_batch_handles_list_data(self, mock_config):
        """Test get_batch() converts list data to dict."""
        trainer = Trainer.__new__(Trainer)
        trainer.config = mock_config

        # Mock iterator returning list
        mock_iter = Mock()
        mock_iter.__next__ = Mock(return_value=[[1, 2, 3], [2, 3, 4]])

        batch = trainer.get_batch(mock_iter)

        # Should convert list to dict with 'input_ids' key
        assert isinstance(batch, dict)
        assert 'input_ids' in batch

    def test_get_batch_handles_none_data(self, mock_config):
        """Test get_batch() handles None data gracefully."""
        trainer = Trainer.__new__(Trainer)
        trainer.config = mock_config

        # Mock iterator returning None
        mock_iter = Mock()
        mock_iter.__next__ = Mock(return_value=None)

        batch = trainer.get_batch(mock_iter)

        # Should return empty dict
        assert isinstance(batch, dict)
        assert len(batch) == 0

    def test_get_batch_calls_correct_internal_method(self, mock_config, mock_dataset_iter):
        """Test get_batch() calls the correct internal method based on config."""
        trainer = Trainer.__new__(Trainer)
        trainer.config = mock_config

        # Mock internal methods
        trainer._get_batch_naive = Mock(return_value={'input_ids': [1, 2, 3]})
        trainer._get_batch_distributed = Mock()
        trainer._get_batch_remove_redundant = Mock()

        # Test naive mode
        trainer.get_batch(mock_dataset_iter)
        trainer._get_batch_naive.assert_called_once()

        # Test distributed mode
        trainer.config.use_distribute_dataset = True
        trainer._get_batch_distributed.reset_mock()
        trainer.get_batch(mock_dataset_iter)
        trainer._get_batch_distributed.assert_called_once()

        # Test remove redundant mode
        trainer.config.use_distribute_dataset = False
        trainer.config.use_remove_redundant_dataset = True
        trainer._get_batch_remove_redundant.reset_mock()
        trainer.get_batch(mock_dataset_iter)
        trainer._get_batch_remove_redundant.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
