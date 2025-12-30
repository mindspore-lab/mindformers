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
"""Unit tests for CheckpointCallback using pytest."""
import os
import shutil
import sys
import tempfile
from unittest.mock import Mock, patch
import pytest

from mindformers.pynative.callback import CheckpointCallback


# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))


class TestCheckpointCallback:
    """Test cases for CheckpointCallback."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def save_dir(self, temp_dir):
        """Create save directory path."""
        return os.path.join(temp_dir, 'checkpoints')

    @pytest.fixture
    def callback(self, save_dir):
        """Create callback instance."""
        return CheckpointCallback(
            save_dir=save_dir,
            save_interval=100,
            save_optimizer=True,
            keep_checkpoint_max=3
        )

    @pytest.fixture
    def mock_args(self):
        """Create mock args."""
        return Mock()

    @pytest.fixture
    def mock_state(self):
        """Create mock state."""
        state = Mock()
        state.global_step = 0
        state.epoch = 0.0
        state.epoch_step = 100
        state.global_batch_size = 32
        return state

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        return Mock()

    @pytest.fixture
    def mock_optimizer(self):
        """Create mock optimizer."""
        return Mock()

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        cb = CheckpointCallback(save_dir='/path/to/ckpts')

        assert cb.save_dir == '/path/to/ckpts'
        assert cb.save_interval == 1000
        assert cb.save_optimizer is True
        assert cb.keep_checkpoint_max == 5
        assert cb.save_on_train_end is True
        assert cb.user_prefix == "checkpoint"
        assert cb.async_save is False
        assert cb.remove_redundancy is False
        assert cb.async_save_manager is None

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        cb = CheckpointCallback(
            save_dir='/path/to/ckpts',
            save_interval=500,
            save_optimizer=False,
            keep_checkpoint_max=10,
            save_on_train_end=False,
            user_prefix="my_ckpt",
            async_save=True,
            remove_redundancy=True
        )

        assert cb.save_dir == '/path/to/ckpts'
        assert cb.save_interval == 500
        assert cb.save_optimizer is False
        assert cb.keep_checkpoint_max == 10
        assert cb.save_on_train_end is False
        assert cb.user_prefix == "my_ckpt"
        assert cb.async_save is True
        assert cb.remove_redundancy is True

    @patch('mindformers.core.callback_pynative.checkpoint_callback.AsyncSaveManager')
    def test_init_with_async_save(self, mock_async_manager_class, save_dir):
        """Test initialization creates AsyncSaveManager when async_save is True."""
        mock_async_manager = Mock()
        mock_async_manager_class.return_value = mock_async_manager

        with patch('mindformers.core.callback_pynative.checkpoint_callback.logger'):
            cb = CheckpointCallback(
                save_dir=save_dir,
                async_save=True
            )

        # Verify AsyncSaveManager was created
        mock_async_manager_class.assert_called_once_with(async_save=True)
        assert cb.async_save_manager == mock_async_manager

    def test_on_train_begin_creates_directory(self, callback, save_dir, mock_args, mock_state):
        """Test on_train_begin creates save directory."""
        assert not os.path.exists(save_dir)

        with patch('mindformers.core.callback_pynative.checkpoint_callback.logger'):
            callback.on_train_begin(mock_args, mock_state)

        assert os.path.exists(save_dir)

    @patch('mindformers.core.callback_pynative.checkpoint_callback.save_checkpoint')
    def test_on_step_end_saves_at_interval(
        self, mock_save_checkpoint, callback, save_dir, mock_args, mock_state, mock_model, mock_optimizer
    ):
        """Test on_step_end saves checkpoint at interval."""
        os.makedirs(save_dir, exist_ok=True)

        # Step 50 - should not save
        mock_state.global_step = 50
        with patch('mindformers.core.callback_pynative.checkpoint_callback.logger'):
            callback.on_step_end(mock_args, mock_state, model=mock_model, optimizer=mock_optimizer)

        mock_save_checkpoint.assert_not_called()

        # Step 100 - should save
        mock_state.global_step = 100
        with patch('mindformers.core.callback_pynative.checkpoint_callback.logger'):
            callback.on_step_end(mock_args, mock_state, model=mock_model, optimizer=mock_optimizer)

        # Verify save_checkpoint was called
        assert mock_save_checkpoint.called

    @patch('mindformers.core.callback_pynative.checkpoint_callback.save_checkpoint')
    def test_save_checkpoint_with_common_info(
        self, mock_save_checkpoint, callback, save_dir, mock_args, mock_state, mock_model
    ):
        """Test checkpoint saves with CommonInfo (always enabled)."""
        os.makedirs(save_dir, exist_ok=True)
        mock_state.global_step = 100
        mock_state.epoch = 1.5
        mock_state.epoch_step = 100

        with patch('mindformers.core.callback_pynative.checkpoint_callback.logger'):
            callback.on_step_end(mock_args, mock_state, model=mock_model)

        # Verify common_info was passed
        call_args = mock_save_checkpoint.call_args
        assert 'common_info' in call_args[1]
        common_info = call_args[1]['common_info']
        assert common_info is not None
        assert common_info.global_step == 100
        assert common_info.epoch_num == 1

    @patch('mindformers.core.callback_pynative.checkpoint_callback.save_checkpoint')
    def test_save_checkpoint_with_async_save(
        self, mock_save_checkpoint, save_dir, mock_args, mock_state, mock_model
    ):
        """Test checkpoint saves with async save enabled."""
        mock_async_manager = Mock()

        with patch('mindformers.core.callback_pynative.checkpoint_callback.AsyncSaveManager') as mock_async_class:
            mock_async_class.return_value = mock_async_manager

            with patch('mindformers.core.callback_pynative.checkpoint_callback.logger'):
                cb = CheckpointCallback(
                    save_dir=save_dir,
                    save_interval=100,
                    async_save=True
                )

        os.makedirs(save_dir, exist_ok=True)
        mock_state.global_step = 100

        with patch('mindformers.core.callback_pynative.checkpoint_callback.logger'):
            cb.on_step_end(mock_args, mock_state, model=mock_model)

        # Verify prepare_before_save was called
        mock_async_manager.prepare_before_save.assert_called_once()

        # Verify async_save_manager was passed
        call_args = mock_save_checkpoint.call_args
        assert call_args[1]['async_save_manager'] == mock_async_manager

    @patch('mindformers.core.callback_pynative.checkpoint_callback.save_checkpoint')
    def test_save_checkpoint_with_remove_redundancy(
        self, mock_save_checkpoint, save_dir, mock_args, mock_state, mock_model
    ):
        """Test checkpoint saves with remove_redundancy enabled."""
        cb = CheckpointCallback(
            save_dir=save_dir,
            save_interval=100,
            remove_redundancy=True
        )

        os.makedirs(save_dir, exist_ok=True)
        mock_state.global_step = 100

        with patch('mindformers.core.callback_pynative.checkpoint_callback.logger'):
            cb.on_step_end(mock_args, mock_state, model=mock_model)

        # Verify remove_redundancy was passed
        call_args = mock_save_checkpoint.call_args
        assert call_args[1]['remove_redundancy'] is True

    @patch('mindformers.core.callback_pynative.checkpoint_callback.save_checkpoint')
    def test_save_checkpoint_all_parameters(
        self, mock_save_checkpoint, save_dir, mock_args, mock_state, mock_model, mock_optimizer
    ):
        """Test all parameters are passed correctly to save_checkpoint."""
        cb = CheckpointCallback(
            save_dir=save_dir,
            save_interval=100,
            save_optimizer=True,
            keep_checkpoint_max=5,
            user_prefix="test",
            async_save=False,
            remove_redundancy=True
        )

        os.makedirs(save_dir, exist_ok=True)
        mock_state.global_step = 200
        mock_state.epoch = 2.0
        mock_state.epoch_step = 100

        with patch('mindformers.core.callback_pynative.checkpoint_callback.logger'):
            cb.on_step_end(mock_args, mock_state, model=mock_model, optimizer=mock_optimizer)

        # Verify all expected parameters
        mock_save_checkpoint.assert_called_once()
        call_args = mock_save_checkpoint.call_args

        assert call_args[1]['iteration'] == 200
        assert call_args[1]['network'] == mock_model
        assert call_args[1]['optimizer'] == mock_optimizer
        assert call_args[1]['async_save_manager'] is None  # Not enabled
        assert call_args[1]['common_info'] is not None
        assert call_args[1]['keep_max_num'] == 5
        assert call_args[1]['user_prefix'] == "test"
        assert call_args[1]['save_checkpoint_path'] == save_dir
        assert call_args[1]['remove_redundancy'] is True

    @patch('mindformers.core.callback_pynative.checkpoint_callback.save_checkpoint')
    def test_create_common_info_from_state(
        self, mock_save_checkpoint, callback, save_dir, mock_args, mock_state, mock_model
    ):
        """Test CommonInfo creation from TrainerState."""
        os.makedirs(save_dir, exist_ok=True)
        mock_state.global_step = 200  # Changed to 200 (multiple of save_interval=100)
        mock_state.epoch = 2.0
        mock_state.epoch_step = 100
        mock_state.global_batch_size = 64

        with patch('mindformers.core.callback_pynative.checkpoint_callback.logger'):
            callback.on_step_end(mock_args, mock_state, model=mock_model)

        # Get common_info from call
        call_args = mock_save_checkpoint.call_args
        common_info = call_args[1]['common_info']

        # Verify CommonInfo fields
        assert common_info.epoch_num == 2
        assert common_info.global_step == 200
        assert common_info.step_num == 0  # 200 % 100 = 0
        assert common_info.global_batch_size == 64

    @patch('mindformers.core.callback_pynative.checkpoint_callback.save_checkpoint')
    def test_on_train_end_final_checkpoint(
        self, mock_save_checkpoint, callback, save_dir, mock_args, mock_state, mock_model
    ):
        """Test on_train_end saves final checkpoint with same user_prefix (no _final suffix)."""
        os.makedirs(save_dir, exist_ok=True)
        mock_state.global_step = 1000

        with patch('mindformers.core.callback_pynative.checkpoint_callback.logger'):
            callback.on_train_end(mock_args, mock_state, model=mock_model)

        # Verify save_checkpoint was called with same prefix (no _final)
        call_args = mock_save_checkpoint.call_args
        assert call_args[1]['user_prefix'] == "checkpoint"  # Default user_prefix
        assert call_args[1]['keep_max_num'] == 3  # Same as callback.keep_checkpoint_max

    @patch('mindformers.core.callback_pynative.checkpoint_callback.save_checkpoint')
    def test_error_handling(
        self, mock_save_checkpoint, callback, save_dir, mock_args, mock_state, mock_model
    ):
        """Test error handling during checkpoint save."""
        os.makedirs(save_dir, exist_ok=True)
        mock_state.global_step = 100

        # Mock save_checkpoint to raise error
        mock_save_checkpoint.side_effect = Exception("Save error")

        with patch('mindformers.core.callback_pynative.checkpoint_callback.logger') as mock_logger:
            callback.on_step_end(mock_args, mock_state, model=mock_model)

            # Should log error message
            mock_logger.error.assert_called_once()
            assert "Error saving checkpoint" in str(mock_logger.error.call_args)

    def test_no_model_warning(self, callback, mock_args, mock_state):
        """Test warning when no model is provided."""
        mock_state.global_step = 100

        with patch('mindformers.core.callback_pynative.checkpoint_callback.logger') as mock_logger:
            callback.on_step_end(mock_args, mock_state)

            # Should log warning
            mock_logger.warning.assert_called_once()
            assert "No model provided" in str(mock_logger.warning.call_args)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
