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
"""Unit tests for callback module using pytest."""
import os
import sys
from unittest.mock import Mock

import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from mindformers.core.callback_pynative import TrainerCallback, CallbackHandler  # pylint: disable=wrong-import-position


class TestTrainerCallback:
    """Test cases for TrainerCallback base class."""

    @pytest.fixture
    def callback(self):
        """Create callback instance."""
        return TrainerCallback()

    @pytest.fixture
    def mock_args(self):
        """Create mock args."""
        return Mock()

    @pytest.fixture
    def mock_state(self):
        """Create mock state."""
        return Mock()

    def test_on_begin(self, callback, mock_args, mock_state):
        """Test on_begin method."""
        # Should not raise error (base implementation does nothing)
        callback.on_begin(mock_args, mock_state)

    def test_on_end(self, callback, mock_args, mock_state):
        """Test on_end method."""
        callback.on_end(mock_args, mock_state)

    def test_on_train_begin(self, callback, mock_args, mock_state):
        """Test on_train_begin method."""
        callback.on_train_begin(mock_args, mock_state)

    def test_on_train_end(self, callback, mock_args, mock_state):
        """Test on_train_end method."""
        callback.on_train_end(mock_args, mock_state)

    def test_on_epoch_begin(self, callback, mock_args, mock_state):
        """Test on_epoch_begin method."""
        callback.on_epoch_begin(mock_args, mock_state)

    def test_on_epoch_end(self, callback, mock_args, mock_state):
        """Test on_epoch_end method."""
        callback.on_epoch_end(mock_args, mock_state)

    def test_on_step_begin(self, callback, mock_args, mock_state):
        """Test on_step_begin method."""
        callback.on_step_begin(mock_args, mock_state)

    def test_on_step_end(self, callback, mock_args, mock_state):
        """Test on_step_end method."""
        callback.on_step_end(mock_args, mock_state)


class TestCallbackHandler:
    """Test cases for CallbackHandler class."""

    @pytest.fixture
    def model(self):
        """Create mock model."""
        return Mock()

    @pytest.fixture
    def train_dataset(self):
        """Create mock train dataset."""
        return Mock()

    @pytest.fixture
    def eval_dataset(self):
        """Create mock eval dataset."""
        return Mock()

    @pytest.fixture
    def optimizer(self):
        """Create mock optimizer."""
        return Mock()

    @pytest.fixture
    def lr_scheduler(self):
        """Create mock lr scheduler."""
        return Mock()

    def test_init_empty(self):
        """Test initialization with no callbacks."""
        handler = CallbackHandler()
        assert len(handler.callbacks) == 0
        assert handler.model is None
        assert handler.train_dataset is None

    def test_init_with_components(self, model, train_dataset, eval_dataset, optimizer, lr_scheduler):
        """Test initialization with model and datasets."""
        handler = CallbackHandler(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler
        )
        assert handler.model == model
        assert handler.train_dataset == train_dataset
        assert handler.eval_dataset == eval_dataset
        assert handler.optimizer == optimizer
        assert handler.lr_scheduler == lr_scheduler

    def test_add_callback_instance(self):
        """Test adding callback instance."""
        handler = CallbackHandler()
        cb = TrainerCallback()
        handler.add_callback(cb)
        assert len(handler.callbacks) == 1
        assert handler.callbacks[0] == cb

    def test_add_callback_class(self):
        """Test adding callback class (should instantiate)."""
        handler = CallbackHandler()
        handler.add_callback(TrainerCallback)
        assert len(handler.callbacks) == 1
        assert isinstance(handler.callbacks[0], TrainerCallback)

    def test_add_callback_duplicate_warning(self):
        """Test warning when adding duplicate callback type."""
        handler = CallbackHandler()
        cb1 = TrainerCallback()
        cb2 = TrainerCallback()

        handler.add_callback(cb1)
        handler.add_callback(cb2)

        # Both should still be added
        assert len(handler.callbacks) == 2

    def test_pop_callback_by_instance(self):
        """Test removing callback by instance."""
        handler = CallbackHandler()
        cb = TrainerCallback()
        handler.add_callback(cb)

        removed = handler.pop_callback(cb)
        assert removed == cb
        assert len(handler.callbacks) == 0

    def test_pop_callback_by_class(self):
        """Test removing callback by class."""
        handler = CallbackHandler()
        cb = TrainerCallback()
        handler.add_callback(cb)

        removed = handler.pop_callback(TrainerCallback)
        assert isinstance(removed, TrainerCallback)
        assert len(handler.callbacks) == 0

    def test_pop_callback_not_found(self):
        """Test popping non-existent callback."""
        handler = CallbackHandler()
        removed = handler.pop_callback(TrainerCallback)
        assert removed is None

    def test_remove_callback_by_instance(self):
        """Test removing callback by instance."""
        handler = CallbackHandler()
        cb = TrainerCallback()
        handler.add_callback(cb)

        handler.remove_callback(cb)
        assert len(handler.callbacks) == 0

    def test_remove_callback_by_class(self):
        """Test removing callback by class."""
        handler = CallbackHandler()
        cb = TrainerCallback()
        handler.add_callback(cb)

        handler.remove_callback(TrainerCallback)
        assert len(handler.callbacks) == 0

    def test_on_begin_calls_all_callbacks(self):
        """Test on_begin calls all registered callbacks."""
        handler = CallbackHandler()

        cb1 = Mock(spec=TrainerCallback)
        cb2 = Mock(spec=TrainerCallback)
        handler.callbacks = [cb1, cb2]

        args = Mock()
        state = Mock()

        handler.on_begin(args, state)

        cb1.on_begin.assert_called_once()
        cb2.on_begin.assert_called_once()

    def test_on_train_begin_calls_all_callbacks(self, model, optimizer):
        """Test on_train_begin calls all registered callbacks."""
        handler = CallbackHandler(
            model=model,
            optimizer=optimizer
        )

        cb = Mock(spec=TrainerCallback)
        handler.callbacks = [cb]

        args = Mock()
        state = Mock()

        handler.on_train_begin(args, state)

        cb.on_train_begin.assert_called_once()
        # Check that model and optimizer are passed
        call_kwargs = cb.on_train_begin.call_args[1]
        assert call_kwargs['model'] == model
        assert call_kwargs['optimizer'] == optimizer

    def test_on_step_end_calls_all_callbacks(self):
        """Test on_step_end calls all registered callbacks."""
        handler = CallbackHandler()

        cb1 = Mock(spec=TrainerCallback)
        cb2 = Mock(spec=TrainerCallback)
        handler.callbacks = [cb1, cb2]

        args = Mock()
        state = Mock()

        handler.on_step_end(args, state, loss=0.5)

        cb1.on_step_end.assert_called_once()
        cb2.on_step_end.assert_called_once()

        # Check loss is passed in kwargs
        call_kwargs = cb1.on_step_end.call_args[1]
        assert call_kwargs['loss'] == 0.5

    def test_call_event(self):
        """Test call_event method."""
        handler = CallbackHandler()

        cb = Mock(spec=TrainerCallback)
        cb.on_train_begin = Mock(return_value="result")
        handler.callbacks = [cb]

        args = Mock()
        state = Mock()

        result = handler.call_event("on_train_begin", args, state)

        cb.on_train_begin.assert_called_once()
        assert result == "result"

    def test_callback_list_property(self):
        """Test callback_list property returns string representation."""
        handler = CallbackHandler()

        class CustomCallback1(TrainerCallback):
            pass

        class CustomCallback2(TrainerCallback):
            pass

        handler.add_callback(CustomCallback1())
        handler.add_callback(CustomCallback2())

        callback_list = handler.callback_list
        assert "CustomCallback1" in callback_list
        assert "CustomCallback2" in callback_list

    def test_init_with_callback_list(self):
        """Test initialization with callback list."""
        cb1 = TrainerCallback()
        cb2 = TrainerCallback()

        handler = CallbackHandler(callbacks=[cb1, cb2])

        assert len(handler.callbacks) == 2
