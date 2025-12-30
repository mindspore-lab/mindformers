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
"""TrainerState for tracking training progress."""
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class TrainerState:
    """
    A class containing the state of the :class:`~Trainer` during training.

    Args:
        epoch (float):
            Current epoch number. Can be fractional for partial epochs.
        global_step (int):
            Current global training step.
        max_steps (int):
            Total number of training steps to perform.
        eval_steps (int):
            Number of steps between evaluations.
        save_steps (int):
            Number of steps between checkpoint saves.
        epoch_step (int):
            Number of steps in one epoch. Used to determine epoch boundaries.
        global_batch_size (int):
            Global batch size across all devices.
        best_metric (float):
            Best metric value achieved so far.
        best_model_checkpoint (str):
            Path to the best model checkpoint.
        is_train_begin (bool):
            Whether training has begun.
        is_train_end (bool):
            Whether training has ended.
    """

    epoch: float = 0.0
    global_step: int = 0
    max_steps: int = 0
    eval_steps: int = 0
    save_steps: int = 0
    epoch_step: int = 0
    global_batch_size: int = 0
    best_metric: Optional[float] = None
    best_model_checkpoint: Optional[str] = None
    is_train_begin: bool = False
    is_train_end: bool = False

    def update_epoch(self):
        """Update epoch based on current step and epoch_step."""
        if self.epoch_step > 0:
            self.epoch = self.global_step / self.epoch_step

    def save_to_dict(self) -> Dict[str, Any]:
        """
        Save the state to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing the state.
        """
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "max_steps": self.max_steps,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "epoch_step": self.epoch_step,
            "global_batch_size": self.global_batch_size,
            "best_metric": self.best_metric,
            "best_model_checkpoint": self.best_model_checkpoint,
        }

    @classmethod
    def load_from_dict(cls, state_dict: Dict[str, Any]) -> "TrainerState":
        """
        Load the state from a dictionary.

        Args:
            state_dict (Dict[str, Any]): Dictionary containing the state.

        Returns:
            TrainerState: The loaded state object.
        """
        return cls(**state_dict)

    def __repr__(self):
        """Return string representation of the state."""
        return (
            f"TrainerState(epoch={self.epoch}, "
            f"global_step={self.global_step}, "
            f"max_steps={self.max_steps})"
        )
