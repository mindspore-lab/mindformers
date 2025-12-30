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
"""Checkpoint callback for saving model checkpoints during training."""
import os
from mindformers.core.callback_pynative.callback import TrainerCallback
from mindformers.tools.logger import logger
from mindformers.checkpoint import save_checkpoint
from mindformers.checkpoint.checkpoint import CommonInfo, AsyncSaveManager


class CheckpointCallback(TrainerCallback):
    """
    Callback for saving model checkpoints during training.

    This callback saves model checkpoints at specified intervals and at the end of training.
    It can save both the model parameters and optimizer state.

    Args:
        save_dir (str): Directory where checkpoints will be saved
        save_interval (int): Number of steps between checkpoint saves. Default: 1000
        save_optimizer (bool): Whether to save optimizer state. Default: True
        keep_checkpoint_max (int): Maximum number of checkpoints to keep. Default: 5
        save_on_train_end (bool): Whether to save checkpoint at the end of training. Default: True
        user_prefix (str): Prefix for checkpoint file names. Default: "checkpoint"
        async_save (bool): Enable async save. Default: False
        remove_redundancy (bool): Whether to remove redundancy when saving. Default: False
    """

    def __init__(
        self,
        save_dir: str,
        save_interval: int = 1000,
        save_optimizer: bool = True,
        keep_checkpoint_max: int = 5,
        save_on_train_end: bool = True,
        user_prefix: str = "checkpoint",
        async_save: bool = False,
        remove_redundancy: bool = False
    ):
        """
        Initialize the CheckpointCallback.

        Args:
            save_dir: Directory path for saving checkpoints
            save_interval: Steps between checkpoint saves
            save_optimizer: Whether to save optimizer state
            keep_checkpoint_max: Maximum number of checkpoints to keep
            save_on_train_end: Whether to save at training end
            user_prefix: Prefix for checkpoint file names
            async_save: Enable async save
            remove_redundancy: Whether to remove redundancy when saving
        """
        super().__init__()
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.save_optimizer = save_optimizer
        self.keep_checkpoint_max = keep_checkpoint_max
        self.save_on_train_end = save_on_train_end
        self.user_prefix = user_prefix
        self.async_save = async_save
        self.remove_redundancy = remove_redundancy

        # Create async save manager if needed
        self.async_save_manager = None
        if self.async_save:
            self.async_save_manager = AsyncSaveManager(async_save=True)
            logger.info("AsyncSaveManager created")

    # pylint: disable=unused-argument
    def on_train_begin(self, args, state, **kwargs):
        """
        Called at the beginning of training.

        Creates the save directory if it doesn't exist.

        Args:
            args: Training arguments
            state: Trainer state
            **kwargs: Additional keyword arguments
        """
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
            logger.info(f"Created checkpoint directory: {self.save_dir}")

    def on_step_end(self, args, state, **kwargs):
        """
        Called at the end of each training step.

        Saves checkpoint if the current step matches the save interval.

        Args:
            args: Training arguments
            state: Trainer state
            **kwargs: Additional keyword arguments including:
                - model: The model to save
                - optimizer: The optimizer to save (if save_optimizer=True)
        """
        # Check if we should save at this step
        if state.global_step % self.save_interval != 0:
            return

        self._save_checkpoint(args, state, **kwargs)

    def on_train_end(self, args, state, **kwargs):
        """
        Called at the end of training.

        Saves a final checkpoint if save_on_train_end is True.

        Args:
            args: Training arguments
            state: Trainer state
            **kwargs: Additional keyword arguments
        """
        if self.save_on_train_end:
            self._save_checkpoint(args, state, is_final=True, **kwargs)
            logger.info("Training completed. Final checkpoint saved.")

    # pylint: disable=unused-argument
    def _save_checkpoint(self, args, state, is_final=False, **kwargs):
        """
        Save a checkpoint using mindformers.checkpoint.save_checkpoint.

        Args:
            args: Training arguments
            state: Trainer state
            is_final: Whether this is the final checkpoint
            **kwargs: Additional keyword arguments including model and optimizer
        """
        model = kwargs.get("model", None)
        optimizer = kwargs.get("optimizer", None)

        if model is None:
            logger.warning("No model provided to CheckpointCallback, skipping save.")
            return

        # Create CommonInfo from TrainerState (always required)
        common_info = self._create_common_info(state)

        try:
            # Prepare async save manager if needed
            if self.async_save_manager is not None:
                self.async_save_manager.prepare_before_save()

            # Call mindformers save_checkpoint with full parameters
            save_checkpoint(
                iteration=state.global_step,
                network=model,
                optimizer=optimizer if self.save_optimizer else None,
                async_save_manager=self.async_save_manager,
                common_info=common_info,
                keep_max_num=self.keep_checkpoint_max,
                user_prefix=self.user_prefix,
                save_checkpoint_path=self.save_dir,
                remove_redundancy=self.remove_redundancy
            )

            logger.info(
                f"Checkpoint saved at step {state.global_step} to {self.save_dir} "
                f"(async={self.async_save}, remove_redundancy={self.remove_redundancy})"
            )

        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")

    def _create_common_info(self, state) -> CommonInfo:
        """
        Create CommonInfo from TrainerState.

        Args:
            state: Trainer state containing training information

        Returns:
            CommonInfo instance
        """
        common_info = CommonInfo()

        # Extract information from state
        if hasattr(state, 'epoch'):
            common_info.epoch_num = int(state.epoch)

        if hasattr(state, 'global_step'):
            common_info.global_step = state.global_step

        if hasattr(state, 'epoch_step') and state.epoch_step > 0:
            # Calculate step_num within current epoch
            common_info.step_num = state.global_step % state.epoch_step

        # Try to get batch size if available
        if hasattr(state, 'global_batch_size'):
            common_info.global_batch_size = state.global_batch_size

        logger.debug(
            f"Created CommonInfo: epoch={common_info.epoch_num}, "
            f"step={common_info.step_num}, global_step={common_info.global_step}"
        )

        return common_info
