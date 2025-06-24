#  Copyright 2025 Huawei Technologies Co., Ltd
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ============================================================================
"""checkpoint utils apis."""
import os


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
        raise ValueError('iteration must be an integer!')
    directory = 'iter_{:08d}'.format(iteration)
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
        str: The filename for the common data
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
        str: The filename for the metadata
    """
    common_path = get_checkpoint_iter_dir(checkpoints_path, iteration)
    return os.path.join(common_path, 'metadata.json')


def check_iteration_path_exists(checkpoints_path: str) -> bool:
    """
    Check if the iteration path exists.

    Args:
        checkpoints_path (str): The base path where checkpoint files are stored.

    Returns:
        bool: True if the iteration path exists, False otherwise.

    Raises:
        FileNotFoundError: If no tracker file is found in the load directory.
        ValueError: If the iteration number in the tracker file is invalid.
    """
    if checkpoints_path is None:
        return False
    tracker_filename = get_checkpoint_tracker_filename(checkpoints_path)
    if not os.path.isfile(tracker_filename):
        raise FileNotFoundError("No tracker file found in load directory.")
    with open(tracker_filename, 'r') as f:
        iter_string = f.read().strip()
        try:
            iteration = int(iter_string)
        except ValueError:
            raise ValueError(f"Invalid iteration num in {tracker_filename}")
    inter_dir = get_checkpoint_iter_dir(checkpoints_path, iteration)
    return os.path.exists(inter_dir) and os.path.isdir(inter_dir)


def ensure_directory_exists(filename, check_parent=True):
    """
    Create filename's path if it does not already exist.

    Args:
        filename (str): The filename for which to ensure the directory exists.
        check_parent (bool): Whether to check the parent directory as well. Defaults to True.

    Returns:
        None
    """
    dirname = os.path.dirname(filename) if check_parent else filename
    os.makedirs(dirname, exist_ok=True)


def _check_checkpoint_path(path: str):
    """check checkpoint path."""
    if not isinstance(path, str) or isinstance(path, os.PathLike):
        raise ValueError(f"config.load_checkpoint must be a str, but got {path} as type {type(path)}.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"config.load_checkpoint {path} does not exist.")

    if path[-1] == '/':  # remove last '/' in path
        return path[:-1]
    return path
