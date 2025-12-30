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
"""Loss callback for logging training loss."""
import time
from typing import Dict, Any
from mindformers.core.callback_pynative.callback import TrainerCallback
from mindformers.tools.logger import logger


class LossCallback(TrainerCallback):
    """
    Callback for logging loss information during training.

    This callback logs the training loss at the end of each training step.

    Compared to MFLossMonitor:
    - MFLossMonitor supports: pipeline parallel loss fixing, MoE/MTP separate loss,
      model FLOPs calculation, throughput computation, time remaining estimation,
      overflow/scaling_sens monitoring, global_norm logging, TensorBoard integration
    - LossCallback: Simplified version for basic loss logging only

    Args:
        log_interval (int): Number of steps between loss logging. Default: 1

    TODOs:
    - TODO: Support overflow and scaling_sens monitoring (similar to MFLossMonitor)
    - TODO: Support global_norm logging
    - TODO: Support MoE/MTP separate loss printing
    - TODO: Support throughput calculation
    - TODO: Support model FLOPs calculation
    - TODO: Support TensorBoard integration
    - TODO: Support pipeline parallel loss fixing
    """

    def __init__(self, log_interval: int = 1):
        """
        Initialize the LossCallback.

        Args:
            log_interval: How often to log loss (in steps)
        """
        super().__init__()
        self.log_interval = log_interval
        self.step_time = time.time()
        self.epoch_time = time.time()

    # pylint: disable=unused-argument
    def on_train_begin(self, args, state, **kwargs):
        """
        Called at the beginning of training.

        Args:
            args: Training arguments
            state: Trainer state
            **kwargs: Additional keyword arguments
        """
        self.step_time = time.time()
        self.epoch_time = time.time()

    # pylint: disable=unused-argument
    def on_epoch_begin(self, args, state, **kwargs):
        """
        Called at the beginning of each epoch.

        Args:
            args: Training arguments
            state: Trainer state
            **kwargs: Additional keyword arguments
        """
        self.epoch_time = time.time()

    # pylint: disable=unused-argument
    def on_step_begin(self, args, state, **kwargs):
        """
        Called at the beginning of each training step.

        Args:
            args: Training arguments
            state: Trainer state
            **kwargs: Additional keyword arguments
        """
        self.step_time = time.time()

    # pylint: disable=unused-argument
    def on_step_end(self, args, state, **kwargs):
        """
        Called at the end of each training step.

        Logs the loss value and optionally computes statistics.

        Args:
            args: Training arguments
            state: Trainer state
            **kwargs: Additional keyword arguments including:
                - loss: The current step loss value
        """
        loss = kwargs.get("loss", None)

        if loss is None:
            return

        # Convert loss to float if it's a tensor
        if hasattr(loss, "asnumpy"):
            loss_value = float(loss.asnumpy())
        elif hasattr(loss, "item"):
            loss_value = loss.item()
        else:
            loss_value = float(loss)

        # Log loss at specified intervals
        if state.global_step % self.log_interval == 0:
            cur_time = time.time()
            step_time_cost = (cur_time - self.step_time) * 1000  # Convert to milliseconds

            # Prepare log information
            log_info = {
                "loss": loss_value,
                "cur_step": state.global_step,
                "max_steps": state.max_steps,
                "step_time": step_time_cost,
            }

            # Implement later: Extract learning rate from optimizer or lr_scheduler
            # Similar to MFLossMonitor's approach:
            # - Get from kwargs['lr_scheduler'] if available
            # - Or get from args.optimizer.global_step + learning_rate_schedule

            # Print log information
            self._print_log(log_info)

    # pylint: disable=unused-argument
    def on_epoch_end(self, args, state, **kwargs):
        """
        Called at the end of each epoch.

        Args:
            args: Training arguments
            state: Trainer state
            **kwargs: Additional keyword arguments
        """
        epoch_time = time.time() - self.epoch_time
        logger.info(f"Epoch {state.epoch} finished. Time: {epoch_time:.2f}s")

    def _print_log(self, log_info: Dict[str, Any]):
        """
        Print formatted log information in MFLossMonitor style.

        Format similar to MFLossMonitor:
        "step:[cur_step/max_steps], loss: X.XXXXXX, per_step_time: Xms, lr: X.XXXe-XX"

        Args:
            log_info: Dictionary containing log information
            state: Trainer state for accessing additional info
        """
        cur_step = log_info.get('cur_step', 0)
        max_steps = log_info.get('max_steps', 0)
        loss = log_info.get('loss', 0)
        per_step_time = int(log_info.get('step_time', 0))

        # Build log string in MFLossMonitor format
        log_parts = [f"step:[{cur_step:5d}/{max_steps:5d}]"]
        log_parts.append(f"loss: {loss:.6f}")
        log_parts.append(f"per_step_time: {per_step_time}ms")

        if "learning_rate" in log_info:
            lr = log_info["learning_rate"]
            if isinstance(lr, (list, tuple)):
                lr = lr[0]
            log_parts.append(f"lr: {lr:.6e}")

        # Format: "{ step:[X/Y], loss: X.XXXXXX, per_step_time: Xms, lr: X.XXXe-XX }"
        logger.info("{ " + ", ".join(log_parts) + " }")
