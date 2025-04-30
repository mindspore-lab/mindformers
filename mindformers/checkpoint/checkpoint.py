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
"""load/save checkpoint apis."""
from mindspore.nn import Cell
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore.nn.optim.optimizer import Optimizer


# first api defines, no implement, need to disable pylint.
# pylint: disable=unused-argument
def load_checkpoint(network: Cell, optimizer: Optimizer = None, scheduler: LearningRateSchedule = None):
    """load checkpoint into network or optimizer.

    Args:
        network (Cell): mindspore model obj.
        optimizer (Optimizer, optional): optimizer obj. Defaults to None.
        scheduler (LearningRateSchedule, optional): learning rate scheduler. Defaults to None.
    """
    if network is None:
        raise ValueError("The 'network' cannot be None.")
    # Placeholder logic to save the network.
    # first api defines, no implement, need to disable pylint.
    # pylint: disable=unnecessary-pass
    pass


# first api defines, no implement, need to disable pylint.
# pylint: disable=unused-argument
def save_checkpoint(iteration: int, network: Cell, optimizer: Optimizer = None, scheduler: LearningRateSchedule = None):
    """Saves the current state of the training process, including the model, optimizer,
    and learning rate scheduler, to a checkpoint file.
        iteration (int): The current training iteration step.
        network (Cell): The MindSpore model object to be saved.
        optimizer (Optimizer, optional): The optimizer object associated with the model. Defaults to None.
        scheduler (LearningRateSchedule, optional): The learning rate scheduler object. Defaults to None.
    Returns:
        None
    """
    # first api defines, no implement, need to disable pylint.
    # pylint: disable=unnecessary-pass
    pass


# first api defines, no implement, need to disable pylint.
# pylint: disable=unused-argument
def get_checkpoint_name(use_prefix: str, file_idx: int, total_file_num: int, file_type: str) -> str:
    """Generate a checkpoint name for model parameters or optimizer parameters.
        use_prefix (str): The prefix to use for the checkpoint file name.
        file_idx (int): The index of the current file.
        total_file_num (int): The total number of files.
        file_type (str): The type of the file (e.g., model parameters, optimizer parameters).
    Returns:
        str: The generated checkpoint file name.
    """
    # first api defines, no implement, need to disable pylint.
    # pylint: disable=unnecessary-pass
    pass


# first api defines, no implement, need to disable pylint.
# pylint: disable=unused-argument
def get_checkpoint_dir(checkpoints_path: str, iteration: int) -> str:
    """
    Get the directory path for a specific checkpoint iteration.

    Args:
        checkpoints_path (str): The base path where checkpoints are stored.
        iteration (int): The iteration number of the checkpoint.

    Returns:
        str: The full path to the directory for the specified checkpoint iteration.
    """
    # first api defines, no implement, need to disable pylint.
    # pylint: disable=unnecessary-pass
    pass


# first api defines, no implement, need to disable pylint.
# pylint: disable=unused-argument
def get_checkpoint_tracker_filename(checkpoints_path: str) -> str:
    """
    Generate the filename for the checkpoint tracker.

    Args:
        checkpoints_path (str): The path to the directory where checkpoints are stored.

    Returns:
        str: The filename for the checkpoint tracker.
    """
    # first api defines, no implement, need to disable pylint.
    # pylint: disable=unnecessary-pass
    pass


# first api defines, no implement, need to disable pylint.
# pylint: disable=unused-argument
def get_checkpoint_common_filename(checkpoints_path: str, iteration: int) -> str:
    """
    Generate a common filename for a checkpoint based on the given path and iteration.

    Args:
        checkpoints_path (str): The base directory or path where checkpoints are stored.
        iteration (int): The iteration number to include in the checkpoint filename.

    Returns:
        str: The generated checkpoint filename including the iteration number.
    """
    # first api defines, no implement, need to disable pylint.
    # pylint: disable=unnecessary-pass
    pass


# first api defines, no implement, need to disable pylint.
# pylint: disable=unused-argument
def get_metadata_filename(checkpoints_path: str, iteration: int) -> str:
    """
    Generate the metadata filename for a specific checkpoint iteration.

    Args:
        checkpoints_path (str): The base path where checkpoint files are stored.
        iteration (int): The iteration number of the checkpoint.

    Returns:
        str: The full path to the metadata file for the specified checkpoint iteration.
    """
    # first api defines, no implement, need to disable pylint.
    # pylint: disable=unnecessary-pass
    pass
