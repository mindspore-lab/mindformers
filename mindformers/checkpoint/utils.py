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
"""APIs of checkpoint utils."""

import os
import shutil
import re
from pathlib import Path

from mindformers.tools.logger import logger


PER_ITERATION_CKPT_DIR_PREFIX = "iteration_"


def check_checkpoints_dir_max_num(max_keep_num: int, checkpoints_root_path: str = None):
    """
    Monitor the maximum number of weights that can be stored.
    If the number of checkpoint directory greater than 'max_keep_num',
        the most recently saved weights file will be kept.
    max_keep_num (int): The maximum number of iterations to save weights.
    checkpoints_root_path (str): The root directory where weights are saved,
        including the weight directories of all iterations.
    """
    # Matches regular expressions to get numbers.
    pattern: str = PER_ITERATION_CKPT_DIR_PREFIX + r"(\d+)"

    # Get the root path information.
    root_path = Path(checkpoints_root_path)

    # Get all matching directory.
    matched_dirs = []
    for item in root_path.iterdir():
        if item.is_dir():
            match = re.fullmatch(pattern, item.name)
            if match:
                # Convert to integer comparison
                num = int(match.group(1))
                matched_dirs.append((num, item))

    # Sort by number in ascending order.
    matched_dirs.sort(key=lambda x: x[0])

    # If the quantity does not exceed the limit, no need to delete.
    if len(matched_dirs) <= max_keep_num:
        logger.info(f"The current number of weights is: {len(matched_dirs)}, "
                    f"no more than 'keep_checkpoint_max': {max_keep_num}. "
                    f"So no need to remove any checkpoints directory.")
        return

    # Calculate how many directory should be deleted.
    num_to_delete = len(matched_dirs) - max_keep_num
    logger.warning(f"The current number of weights is: {len(matched_dirs)}, "
                   f"more than 'keep_checkpoint_max': {max_keep_num}.")

    for i in range(num_to_delete):
        num, dir_path = matched_dirs[i]
        logger.warning(f"So, the oldest directory: '{dir_path.name}' (iteration: {num}) will be removed!")
        try:
            shutil.rmtree(dir_path)  # Delete entire directory
        except Exception as e:
            raise RuntimeError(f"Failed to delete folder '{dir_path}'") from e


def get_checkpoint_iter_dir(checkpoints_path: str, iteration: int) -> str:
    """
    Get the directory path for a specific checkpoint iteration.

    Args:
        checkpoints_path (str): The base path where checkpoints are stored.
        iteration (int): The iteration number of the checkpoint.

    Returns:
        str: The full path to the directory for the specified checkpoint iteration.
    """
    if not isinstance(iteration, int):
        raise ValueError(f"'iteration' must be an integer! But got '{type(iteration)}'.")

    directory = PER_ITERATION_CKPT_DIR_PREFIX + '{:08d}'.format(iteration)
    iter_dir = os.path.join(checkpoints_path, directory)

    return iter_dir


def get_checkpoint_tracker_filename(checkpoints_path: str) -> str:
    """
    Generate the filename for the checkpoint tracker.

    Args:
        checkpoints_path (str): The path to the directory where checkpoints are stored.

    Returns:
        str: The filename for the checkpoint tracker.
    """
    return os.path.join(checkpoints_path, 'latest_checkpointed_iteration.txt')


def get_common_filename(checkpoints_path: str, iteration: int) -> str:
    """
    Generate a common filename for a checkpoint based on the given path and iteration.

    Args:
        checkpoints_path (str): The base directory or path where checkpoints are stored.
        iteration (int): The iteration number to include in the checkpoint filename.

    Returns:
        str: The filename for the common data.
    """
    common_path = get_checkpoint_iter_dir(checkpoints_path, iteration)
    return os.path.join(common_path, 'common.json')


def get_metadata_filename(checkpoints_path: str, iteration: int) -> str:
    """
    Generate a metadata filename for a checkpoint based on the given path and iteration.

    Args:
        checkpoints_path (str): The base path where checkpoint files are stored.
        iteration (int): The iteration number of the checkpoint.

    Returns:
        str: The filename for the metadata.
    """
    metadata_path = get_checkpoint_iter_dir(checkpoints_path, iteration)
    return os.path.join(metadata_path, 'metadata.json')


def get_latest_iteration_from_tracker(checkpoints_path: str) -> bool:
    """
    Get the iteration tracker file content. Used in resume scene.

    Args:
        checkpoints_path (str): The base path where checkpoint files are stored.

    Returns:
        The value of the tracker file (latest_checkpointed_iteration.txt) recorded.

    Raises:
        ValueError: If the iteration number in the tracker file is invalid.
        FileNotFoundError: If the weight folder corresponding to the tracker file can not be found.
    """
    # Check the tracker file is exist, and it is a file.
    tracker_filename = get_checkpoint_tracker_filename(checkpoints_path)
    if not os.path.isfile(tracker_filename):
        raise FileNotFoundError(f"No tracker file found in load directory: '{tracker_filename}'.")

    # Get the latest iteration number from the tracker file.
    with open(tracker_filename, 'r') as f:
        iter_string = f.read().strip()
        try:
            iteration = int(iter_string)
        except Exception as e:
            raise ValueError(f"Invalid iteration num in {tracker_filename}") from e
    logger.info(f"The latest valid iteration is '{iteration}', get it from '{tracker_filename}'.")

    # Check that the corresponding weight path exists, and it is a directory.
    latest_inter_dir = get_checkpoint_iter_dir(checkpoints_path, iteration)
    if not os.path.isdir(latest_inter_dir):
        raise FileNotFoundError(f"Can not find the latest iteration weight path: '{latest_inter_dir}'. "
                                f"Or it does not a directory.")

    # Pass all check, return the iteration
    return iteration


def get_checkpoint_name(cur_iter_checkpoint_dir: str, user_prefix: str, file_idx: int, total_file_num: int,
                        file_type: str) -> str:
    """
    Generate a checkpoint name for model parameters or optimizer parameters.

    Args:
        cur_iter_checkpoint_dir (str): Currently iteration checkpoint path.
        user_prefix (str): The prefix to use for the checkpoint file name.
        file_idx (int): The index of the current file.
        total_file_num (int): The total number of files.
        file_type (str): The type of the file (e.g., model parameters, optimizer parameters).

    Returns:
        str: The generated checkpoint file name.
    """
    if file_type == "Model":
        type_prefix = 'model'
    elif file_type == "Optimizer":
        type_prefix = 'opt'
    else:
        raise TypeError(f"The type of safetensors file must be 'Model' or 'Optimizer', but got '{file_type}'.")

    if user_prefix is None:
        file_name = f'{type_prefix}-{file_idx:07d}-{total_file_num:07d}'
    else:
        file_name = f'{user_prefix}-{type_prefix}-{file_idx:07d}-{total_file_num:07d}'

    if cur_iter_checkpoint_dir is None:
        return file_name
    return os.path.join(cur_iter_checkpoint_dir, file_name)


def get_sharded_tensor_shard_id(param_name, global_offset):
    """
    Generate a unique identifier for a sharded tensor based on its parameter name and global offset.

    Args:
        param_name (str): The name of the parameter associated with the sharded tensor.
        global_offset (tuple): The global offset of the sharded tensor.

    Returns:
        str: A unique identifier for the sharded tensor.
    """
    return str(tuple((param_name, tuple(global_offset))))
