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
"""Base Strategy Class."""

from abc import ABC, abstractmethod


class LoadStrategyBase(ABC):
    """
    Base class for defining load strategies.
    This class serves as an abstract base for implementing various strategies
    to load and apply parallelization for models or data. Subclasses should
    provide concrete implementations for the abstract methods.
    Methods:
        load(*args, **kwargs):
            Abstract method to define the logic for loading resources.
            Must be implemented by subclasses.
        apply_loading_parallelization(*args, **kwargs):
            Abstract method to define the logic for applying parallelization
    """
    @abstractmethod
    def load(self, *args, **kwargs):
        """
        Load the necessary resources or configurations.

        This method is intended to be overridden by subclasses to implement
        specific loading logic. It currently does nothing.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        # first api defines, no implement, need to disable pylint.
        # pylint: disable=unnecessary-pass
        pass

    @abstractmethod
    def apply_loading_parallelization(self, *args, **kwargs):
        """
        Apply loading parallelization strategy.

        This method is intended to be overridden by subclasses to implement
        specific strategies for parallelizing the loading of resources or data.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Note:
            This is a placeholder method and does not contain any implementation.
        """
        # first api defines, no implement, need to disable pylint.
        # pylint: disable=unnecessary-pass
        pass


class SaveStrategyBase:
    """
    Base class for defining save strategies.
    This abstract base class provides the interface for implementing
    custom save strategies. Subclasses must override the abstract
    methods to define specific saving behavior and parallelization
    logic.
    Methods:
        save(*args, **kwargs):
            Abstract method to save data or models. Must be implemented
            by subclasses.
        apply_saving_parallelization(*args, **kwargs):
            Abstract method to apply parallelization techniques during
            the saving process. Must be implemented by subclasses.
    """
    @abstractmethod
    def save(self, *args, **kwargs):
        """
        Save the current state or checkpoint.

        This method is intended to be overridden by subclasses to implement
        specific saving logic. It accepts arbitrary arguments and keyword
        arguments to provide flexibility for different saving strategies.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        # first api defines, no implement, need to disable pylint.
        # pylint: disable=unnecessary-pass
        pass

    @abstractmethod
    def apply_saving_parallelization(self, *args, **kwargs):
        """
        Abstract method to apply saving parallelization strategies.

        This method should be implemented by subclasses to define the logic
        for saving parallelized data or models. The implementation can vary
        depending on the specific parallelization strategy being used.

        Args:
            *args: Variable length argument list for additional parameters.
            **kwargs: Arbitrary keyword arguments for additional parameters.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        # first api defines, no implement, need to disable pylint.
        # pylint: disable=unnecessary-pass
        pass
