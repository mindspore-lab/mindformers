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
"""Unit tests for TrainerState."""
import os
import sys

import pytest

from mindformers.pynative.trainer.train_state import TrainerState


# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))


class TestTrainerState:
    """Test cases for TrainerState."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        state = TrainerState()

        assert state.epoch == 0.0
        assert state.global_step == 0
        assert state.max_steps == 0
        assert state.eval_steps == 0
        assert state.save_steps == 0
        assert state.epoch_step == 0
        assert state.global_batch_size == 0
        assert state.best_metric is None
        assert state.best_model_checkpoint is None
        assert not state.is_train_begin
        assert not state.is_train_end

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        state = TrainerState(
            epoch=2.5,
            global_step=1000,
            max_steps=5000,
            eval_steps=100,
            save_steps=500,
            epoch_step=200,
            global_batch_size=64,
            best_metric=0.95
        )

        assert state.epoch == 2.5
        assert state.global_step == 1000
        assert state.max_steps == 5000
        assert state.eval_steps == 100
        assert state.save_steps == 500
        assert state.epoch_step == 200
        assert state.global_batch_size == 64
        assert state.best_metric == 0.95

    def test_update_epoch(self):
        """Test update_epoch method."""
        state = TrainerState(
            global_step=250,
            epoch_step=100
        )

        state.update_epoch()

        assert state.epoch == 2.5

    def test_update_epoch_zero_epoch_step(self):
        """Test update_epoch with zero epoch_step."""
        state = TrainerState(
            global_step=100,
            epoch_step=0
        )

        # Should not crash
        state.update_epoch()
        assert state.epoch == 0.0

    def test_save_to_dict(self):
        """Test save_to_dict method."""
        state = TrainerState(
            epoch=2.5,
            global_step=1000,
            max_steps=5000,
            eval_steps=100,
            save_steps=500,
            epoch_step=200,
            global_batch_size=64,
            best_metric=0.95,
            best_model_checkpoint="/path/to/ckpt"
        )

        state_dict = state.save_to_dict()

        assert state_dict["epoch"] == 2.5
        assert state_dict["global_step"] == 1000
        assert state_dict["max_steps"] == 5000
        assert state_dict["eval_steps"] == 100
        assert state_dict["save_steps"] == 500
        assert state_dict["epoch_step"] == 200
        assert state_dict["global_batch_size"] == 64
        assert state_dict["best_metric"] == 0.95
        assert state_dict["best_model_checkpoint"] == "/path/to/ckpt"

    def test_load_from_dict(self):
        """Test load_from_dict method."""
        state_dict = {
            "epoch": 3.0,
            "global_step": 1500,
            "max_steps": 6000,
            "eval_steps": 150,
            "save_steps": 600,
            "epoch_step": 300,
            "global_batch_size": 128,
            "best_metric": 0.98,
            "best_model_checkpoint": "/path/to/best"
        }

        state = TrainerState.load_from_dict(state_dict)

        assert state.epoch == 3.0
        assert state.global_step == 1500
        assert state.max_steps == 6000
        assert state.eval_steps == 150
        assert state.save_steps == 600
        assert state.epoch_step == 300
        assert state.global_batch_size == 128
        assert state.best_metric == 0.98
        assert state.best_model_checkpoint == "/path/to/best"

    def test_save_and_load_roundtrip(self):
        """Test saving and loading state."""
        original_state = TrainerState(
            epoch=4.2,
            global_step=2000,
            max_steps=8000,
            eval_steps=200,
            save_steps=800,
            epoch_step=400,
            global_batch_size=256,
            best_metric=0.99
        )

        # Save to dict
        state_dict = original_state.save_to_dict()

        # Load from dict
        loaded_state = TrainerState.load_from_dict(state_dict)

        # Verify all fields match
        assert loaded_state.epoch == original_state.epoch
        assert loaded_state.global_step == original_state.global_step
        assert loaded_state.max_steps == original_state.max_steps
        assert loaded_state.eval_steps == original_state.eval_steps
        assert loaded_state.save_steps == original_state.save_steps
        assert loaded_state.epoch_step == original_state.epoch_step
        assert loaded_state.global_batch_size == original_state.global_batch_size
        assert loaded_state.best_metric == original_state.best_metric

    def test_repr(self):
        """Test string representation."""
        state = TrainerState(
            epoch=2.5,
            global_step=1000,
            max_steps=5000
        )

        repr_str = repr(state)

        assert "TrainerState" in repr_str
        assert "epoch=2.5" in repr_str
        assert "global_step=1000" in repr_str
        assert "max_steps=5000" in repr_str

    def test_global_batch_size_in_save_to_dict(self):
        """Test that global_batch_size is included in save_to_dict."""
        state = TrainerState(global_batch_size=32)
        state_dict = state.save_to_dict()

        assert "global_batch_size" in state_dict
        assert state_dict["global_batch_size"] == 32

    def test_global_batch_size_in_load_from_dict(self):
        """Test that global_batch_size is loaded from dict."""
        state_dict = {
            "epoch": 0.0,
            "global_step": 0,
            "max_steps": 1000,
            "eval_steps": 100,
            "save_steps": 100,
            "epoch_step": 100,
            "global_batch_size": 64,
            "best_metric": None,
            "best_model_checkpoint": None
        }

        state = TrainerState.load_from_dict(state_dict)

        assert state.global_batch_size == 64


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
