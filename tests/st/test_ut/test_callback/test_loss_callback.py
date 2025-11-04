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
"""Unit tests for LossCallback using pytest."""
import os
import sys
import time
from unittest.mock import Mock, patch

import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from mindformers.core.callback_pynative import LossCallback  # pylint: disable=wrong-import-position


class TestLossCallback:
    """Test cases for LossCallback."""

    @pytest.fixture
    def callback(self):
        """Create callback fixture."""
        return LossCallback(log_interval=1)

    @pytest.fixture
    def mock_args(self):
        """Create mock args."""
        return Mock()

    @pytest.fixture
    def mock_state(self):
        """Create mock state."""
        state = Mock()
        state.global_step = 0
        state.max_steps = 1000
        state.epoch = 0.0
        return state

    def test_init(self):
        """Test initialization."""
        cb = LossCallback(log_interval=10)
        assert cb.log_interval == 10

    def test_on_train_begin(self, callback, mock_args, mock_state):
        """Test on_train_begin resets state."""
        callback.on_train_begin(mock_args, mock_state)

        assert callback.step_time is not None
        assert callback.epoch_time is not None

    def test_on_epoch_begin(self, callback, mock_args, mock_state):
        """Test on_epoch_begin updates epoch time."""
        old_time = callback.epoch_time
        time.sleep(0.01)
        callback.on_epoch_begin(mock_args, mock_state)

        assert callback.epoch_time != old_time

    def test_on_step_begin(self, callback, mock_args, mock_state):
        """Test on_step_begin updates step time."""
        old_time = callback.step_time
        time.sleep(0.01)
        callback.on_step_begin(mock_args, mock_state)

        assert callback.step_time != old_time

    def test_on_step_end_with_no_loss(self, callback, mock_args, mock_state):
        """Test on_step_end when no loss provided."""
        # Should handle gracefully without error
        callback.on_step_end(mock_args, mock_state)

    def test_on_step_end_with_float_loss(self, callback, mock_args, mock_state):
        """Test on_step_end with float loss."""
        mock_state.global_step = 1
        mock_state.max_steps = 1000
        loss = 0.5

        with patch('mindformers.core.callback_pynative.loss_callback.logger') as mock_logger:
            callback.on_step_end(mock_args, mock_state, loss=loss)
            # Should log the loss in MFLossMonitor format
            mock_logger.info.assert_called()
            call_args = str(mock_logger.info.call_args)
            # Check for MFLossMonitor-style format: "step:[X/Y]"
            assert 'step:[' in call_args

    def test_on_step_end_with_tensor_loss(self, callback, mock_args, mock_state):
        """Test on_step_end with tensor-like loss."""
        mock_state.global_step = 1
        mock_state.max_steps = 1000

        # Mock tensor with asnumpy method
        loss = Mock()
        loss.asnumpy = Mock(return_value=0.5)

        with patch('mindformers.core.callback_pynative.loss_callback.logger') as mock_logger:
            callback.on_step_end(mock_args, mock_state, loss=loss)
            # Should log the loss
            mock_logger.info.assert_called()
            call_args = str(mock_logger.info.call_args)
            assert 'loss:' in call_args

    def test_on_step_end_logs_at_interval(self, mock_args, mock_state):
        """Test that logging happens at specified interval."""
        cb = LossCallback(log_interval=2)
        cb.on_train_begin(mock_args, mock_state)

        # Step 1 - should not log
        mock_state.global_step = 1
        mock_state.max_steps = 1000
        with patch('mindformers.core.callback_pynative.loss_callback.logger') as mock_logger:
            cb.on_step_end(mock_args, mock_state, loss=0.5)
            # Logger should not be called (step 1 % 2 != 0)
            mock_logger.info.assert_not_called()

        # Step 2 - should log
        mock_state.global_step = 2
        with patch('mindformers.core.callback_pynative.loss_callback.logger') as mock_logger:
            cb.on_step_end(mock_args, mock_state, loss=0.4)
            # Logger should be called
            mock_logger.info.assert_called()
            call_args = str(mock_logger.info.call_args)
            # Verify MFLossMonitor-style format
            assert 'per_step_time:' in call_args

    def test_on_step_end_with_lr_scheduler(self, callback, mock_args, mock_state):
        """Test on_step_end includes learning rate when available."""
        mock_state.global_step = 1
        mock_state.max_steps = 1000

        lr_scheduler = Mock()
        lr_scheduler.get_last_lr = Mock(return_value=0.001)

        with patch('mindformers.core.callback_pynative.loss_callback.logger') as mock_logger:
            callback.on_step_end(
                mock_args,
                mock_state,
                loss=0.5,
                lr_scheduler=lr_scheduler
            )
            # Should log with learning rate
            mock_logger.info.assert_called()
            # Note: LR extraction from lr_scheduler is TODO, so won't appear yet

    def test_on_step_end_multiple_steps(self, mock_args, mock_state):
        """Test that on_step_end works correctly over multiple steps."""
        cb = LossCallback(log_interval=1)
        cb.on_train_begin(mock_args, mock_state)

        # Add 3 losses
        for i in range(3):
            mock_state.global_step = i + 1
            mock_state.max_steps = 1000
            with patch('mindformers.core.callback_pynative.loss_callback.logger') as mock_logger:
                cb.on_step_end(mock_args, mock_state, loss=float(i + 1))
                # Should log each step
                mock_logger.info.assert_called()
                # Verify format includes step info
                call_args = str(mock_logger.info.call_args)
                assert f'{i+1:5d}/1000' in call_args or 'step:[' in call_args

    def test_on_epoch_end(self, callback, mock_args, mock_state):
        """Test on_epoch_end prints epoch info."""
        mock_state.epoch = 1.0

        with patch('mindformers.core.callback_pynative.loss_callback.logger') as mock_logger:
            callback.on_epoch_end(mock_args, mock_state)

            # Should log epoch info
            mock_logger.info.assert_called()
            call_args = str(mock_logger.info.call_args)
            assert 'Epoch' in call_args

    def test_on_epoch_end_always_logs(self, callback, mock_args, mock_state):
        """Test on_epoch_end always logs epoch info."""
        mock_state.epoch = 2.0

        with patch('mindformers.core.callback_pynative.loss_callback.logger') as mock_logger:
            callback.on_epoch_end(mock_args, mock_state)

            # Should always log epoch completion
            mock_logger.info.assert_called()

    def test_print_log_format(self, callback):
        """Test _print_log formats correctly in MFLossMonitor style."""
        log_info = {
            'cur_step': 100,
            'max_steps': 1000,
            'loss': 0.123456,
            'learning_rate': 0.001,
            'step_time': 123
        }

        mock_state = Mock()
        mock_state.global_step = 100
        mock_state.max_steps = 1000

        with patch('mindformers.core.callback_pynative.loss_callback.logger') as mock_logger:
            callback._print_log(log_info, mock_state)  # pylint: disable=protected-access

            # Check logger was called
            mock_logger.info.assert_called_once()
            logged_str = str(mock_logger.info.call_args[0][0])

            # Verify MFLossMonitor-style format
            assert 'step:[' in logged_str
            assert 'loss:' in logged_str
            assert 'per_step_time:' in logged_str
            assert 'lr:' in logged_str

    def test_print_log_with_list_lr(self, callback):
        """Test _print_log handles list learning rate."""
        log_info = {
            'cur_step': 100,
            'max_steps': 1000,
            'loss': 0.5,
            'step_time': 100,
            'learning_rate': [0.001, 0.002]  # List of LRs
        }

        mock_state = Mock()
        mock_state.global_step = 100
        mock_state.max_steps = 1000

        with patch('mindformers.core.callback_pynative.loss_callback.logger') as mock_logger:
            callback._print_log(log_info, mock_state)  # pylint: disable=protected-access

            mock_logger.info.assert_called_once()
            logged_str = str(mock_logger.info.call_args[0][0])
            # Should use first LR from list
            assert 'lr:' in logged_str
