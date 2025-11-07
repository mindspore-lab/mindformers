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
"""Callback base classes and handler for Trainer."""
import abc
from typing import List, Optional

from mindformers.tools.logger import logger


class TrainerCallback(metaclass=abc.ABCMeta):
    """
    Base class for callbacks that can be registered with the Trainer.

    A callback can execute custom code at various points during training,
    including at the beginning/end of training, epoch, and step.

    All callback methods receive the following parameters:
        args: Training arguments
        state: Current training state
        **kwargs: Additional keyword arguments including model, optimizer, etc.
    """

    def on_begin(self, args, state, **kwargs):
        """
        Event called at the beginning of a task.

        Args:
            args: Training arguments
            state: Current trainer state
            **kwargs: Additional keyword arguments
        """

    def on_end(self, args, state, **kwargs):
        """
        Event called at the end of a task.

        Args:
            args: Training arguments
            state: Current trainer state
            **kwargs: Additional keyword arguments
        """

    def on_train_begin(self, args, state, **kwargs):
        """
        Event called at the beginning of training.

        Args:
            args: Training arguments
            state: Current trainer state
            **kwargs: Additional keyword arguments
        """

    def on_train_end(self, args, state, **kwargs):
        """
        Event called at the end of training.

        Args:
            args: Training arguments
            state: Current trainer state
            **kwargs: Additional keyword arguments
        """

    def on_epoch_begin(self, args, state, **kwargs):
        """
        Event called at the beginning of an epoch.

        Args:
            args: Training arguments
            state: Current trainer state
            **kwargs: Additional keyword arguments
        """

    def on_epoch_end(self, args, state, **kwargs):
        """
        Event called at the end of an epoch.

        Args:
            args: Training arguments
            state: Current trainer state
            **kwargs: Additional keyword arguments
        """

    def on_step_begin(self, args, state, **kwargs):
        """
        Event called at the beginning of a training step.

        Args:
            args: Training arguments
            state: Current trainer state
            **kwargs: Additional keyword arguments
        """

    def on_step_end(self, args, state, **kwargs):
        """
        Event called at the end of a training step.

        Args:
            args: Training arguments
            state: Current trainer state
            **kwargs: Additional keyword arguments
        """


class CallbackHandler:
    """
    Internal class that manages and calls all registered callbacks.

    This class is responsible for:
    - Managing the list of callbacks
    - Adding/removing callbacks
    - Calling all callbacks at appropriate events

    Args:
        callbacks (List[TrainerCallback], optional):
            List of callbacks to register initially
        model: The model being trained
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        optimizer: Optimizer instance
        lr_scheduler: Learning rate scheduler instance
    """

    def __init__(
        self,
        callbacks: Optional[List[TrainerCallback]] = None,
        model=None,
        train_dataset=None,
        eval_dataset=None,
        optimizer=None,
        lr_scheduler=None
    ):
        """
        Initialize the callback handler.

        Args:
            callbacks: List of TrainerCallback instances
            model: Model instance
            train_dataset: Training dataset instance
            eval_dataset: Evaluation dataset instance
            optimizer: Optimizer instance
            lr_scheduler: Learning rate scheduler instance
        """
        self.callbacks = []
        if callbacks is not None:
            for cb in callbacks:
                self.add_callback(cb)

        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def add_callback(self, callback: TrainerCallback):
        """
        Add a callback to the handler.

        Args:
            callback: TrainerCallback instance or class to add
        """
        # If callback is a class, instantiate it
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__

        # Check if this type of callback already exists
        existing_callbacks = [c.__class__ for c in self.callbacks]
        if cb_class in existing_callbacks:
            logger.warning(
                f"You are adding a {cb_class.__name__} to the callbacks of this Trainer, "
                f"but there is already one. The current list of callbacks is: "
                f"{[c.__class__.__name__ for c in self.callbacks]}"
            )

        self.callbacks.append(cb)

    def pop_callback(self, callback):
        """
        Remove and return a callback from the handler.

        Args:
            callback: TrainerCallback instance or class to remove

        Returns:
            The removed callback instance, or None if not found
        """
        if isinstance(callback, type):
            # callback is a class, find instance of that class
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return cb
        else:
            # callback is an instance
            for cb in self.callbacks:
                if cb == callback:
                    self.callbacks.remove(cb)
                    return cb
        return None

    def remove_callback(self, callback):
        """
        Remove a callback from the handler without returning it.

        Args:
            callback: TrainerCallback instance or class to remove
        """
        if isinstance(callback, type):
            # callback is a class, remove all instances of that class
            for cb in self.callbacks[:]:  # Copy list to avoid modification during iteration
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return
        else:
            # callback is an instance
            if callback in self.callbacks:
                self.callbacks.remove(callback)

    def on_begin(self, args, state, **kwargs):
        """
        Call on_begin for all registered callbacks.

        Args:
            args: Training arguments
            state: Current trainer state
            **kwargs: Additional keyword arguments
        """
        return self.call_event("on_begin", args, state, **kwargs)

    def on_end(self, args, state, **kwargs):
        """
        Call on_end for all registered callbacks.

        Args:
            args: Training arguments
            state: Current trainer state
            **kwargs: Additional keyword arguments
        """
        return self.call_event("on_end", args, state, **kwargs)

    def on_train_begin(self, args, state, **kwargs):
        """
        Call on_train_begin for all registered callbacks.

        Args:
            args: Training arguments
            state: Current trainer state
            **kwargs: Additional keyword arguments
        """
        return self.call_event("on_train_begin", args, state, **kwargs)

    def on_train_end(self, args, state, **kwargs):
        """
        Call on_train_end for all registered callbacks.

        Args:
            args: Training arguments
            state: Current trainer state
            **kwargs: Additional keyword arguments
        """
        return self.call_event("on_train_end", args, state, **kwargs)

    def on_epoch_begin(self, args, state, **kwargs):
        """
        Call on_epoch_begin for all registered callbacks.

        Args:
            args: Training arguments
            state: Current trainer state
            **kwargs: Additional keyword arguments
        """
        return self.call_event("on_epoch_begin", args, state, **kwargs)

    def on_epoch_end(self, args, state, **kwargs):
        """
        Call on_epoch_end for all registered callbacks.

        Args:
            args: Training arguments
            state: Current trainer state
            **kwargs: Additional keyword arguments
        """
        return self.call_event("on_epoch_end", args, state, **kwargs)

    def on_step_begin(self, args, state, **kwargs):
        """
        Call on_step_begin for all registered callbacks.

        Args:
            args: Training arguments
            state: Current trainer state
            **kwargs: Additional keyword arguments
        """
        return self.call_event("on_step_begin", args, state, **kwargs)

    def on_step_end(self, args, state, **kwargs):
        """
        Call on_step_end for all registered callbacks.

        Args:
            args: Training arguments
            state: Current trainer state
            **kwargs: Additional keyword arguments
        """
        return self.call_event("on_step_end", args, state, **kwargs)

    def call_event(self, event: str, args, state, **kwargs):
        """
        Call a specific event on all registered callbacks.

        Args:
            event: Name of the event method to call
            args: Training arguments
            state: Current trainer state
            **kwargs: Additional keyword arguments

        Returns:
            Result from the last callback (if any)
        """
        result = None
        for callback in self.callbacks:
            result = getattr(callback, event)(
                args,
                state,
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                **kwargs,
            )
        return result

    @property
    def callback_list(self) -> str:
        """
        Get a string representation of all registered callbacks.

        Returns:
            String listing all callback class names
        """
        return "\n".join([cb.__class__.__name__ for cb in self.callbacks])
