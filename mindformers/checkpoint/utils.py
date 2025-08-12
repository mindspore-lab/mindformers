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
import re
import json
import time
import shutil
from enum import Enum
from glob import glob
from pathlib import Path
from typing import Optional
import numpy as np

import mindspore as ms
from mindspore import context
from mindspore.common import dtype as mstype
from mindformers.tools.logger import logger


PER_ITERATION_CKPT_DIR_PREFIX = "iteration_"

MS_TYPE_TO_SIZE = {
    "Int8": 8,
    "UInt8": 8,
    "Int16": 16,
    "UInt16": 16,
    "Int32": 32,
    "UInt32": 32,
    "Int64": 64,
    "UInt64": 64,
    "Float16": 16,
    "Float32": 32,
    "Float64": 64,
    "Bool": 1,
    "BFloat16": 16,
    "Int4": 4
}


class FileType(Enum):
    MODEL = "model"
    OPTIMIZER = "opt"


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
                        file_type: FileType) -> str:
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
    type_prefix = file_type.value

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


def _sharded_tensor_shard_id(param_name, global_offset):
    """
    Generate a unique identifier for a sharded tensor based on its parameter name and global offset.

    Args:
        param_name (str): The name of the parameter associated with the sharded tensor.
        global_offset (tuple): The global offset of the sharded tensor.

    Returns:
        str: A unique identifier for the sharded tensor.
    """
    return str(tuple((param_name, tuple(global_offset))))


def _reverse_sharded_tensor_shard_id(shard_id):
    """
    Reverse-engineer the original input parameters from the sharded tensor ID.

    Args:
        shard_id (str): Unique identifier generated by _sharded_tensor_shard_id.

    Returns:
        tuple: Original input parameters (param_name, global_offset).

    Raises:
        ValueError: If the shard_id format is invalid and cannot be parsed.

    Example:
        Input: "('model.layer.weight', (100, 200))"
        Output: ('model.layer.weight', (100, 200))
    """
    try:
        inner_str = shard_id.strip("()")
        param_part, offset_part = inner_str.split(",", 1)
        param_name = param_part.strip("'\" ").strip()
        offset_str = offset_part.strip()
        if not offset_str or offset_str == "()":
            global_offset = ()
        else:
            elements = offset_str.strip("()").split(",")
            global_offset = tuple(int(e.strip()) for e in elements if e.strip())
        return param_name, global_offset
    except Exception as e:
        raise ValueError(f"Failed to parse shard ID: '{shard_id}'") from e


def _get_shard_size(local_shape, dtype):
    """
    Calculate the size of a shard in bytes based on its local shape and data type.

    Args:
        local_shape (tuple): The local shape of the shard.
        dtype: The data type of the shard.

    Returns:
        int: The size of the shard in bytes.
    """
    type_size = MS_TYPE_TO_SIZE.get(str(dtype), 16)
    element_count = 1
    for size in local_shape:
        element_count *= size
    return element_count * type_size


def numpy_dtype_to_mindspore(numpy_dtype):
    """
    Convert NumPy dtype to MindSpore dtype

    Args:
        numpy_dtype: NumPy data type (e.g., np.float32, np.int64)

    Returns:
        mindspore.dtype: Corresponding MindSpore data type
    """
    # Mapping table for basic data types
    dtype_mapping = {
        np.bool_: mstype.bool_,
        np.int8: mstype.int8,
        np.int16: mstype.int16,
        np.int32: mstype.int32,
        np.int64: mstype.int64,
        np.uint8: mstype.uint8,
        np.uint16: mstype.uint16,
        np.uint32: mstype.uint32,
        np.uint64: mstype.uint64,
        np.float16: mstype.float16,
        np.float32: mstype.float32,
        np.float64: mstype.float64,
        np.complex64: mstype.complex64,
        np.complex128: mstype.complex128,
    }

    # Handle both dtype objects and direct type references
    if isinstance(numpy_dtype, np.dtype):
        numpy_dtype = numpy_dtype.type

    # Look up mapping and raise error if not found
    if numpy_dtype in dtype_mapping:
        return dtype_mapping[numpy_dtype]

    raise ValueError(f"Unsupported NumPy data type: {numpy_dtype}")


def verify_ckpt_valid(checkpoint_dir: str) -> Optional[str]:
    """
    Validates the integrity of a checkpoint directory by checking metadata and Safetensors file existence.

    Ensures the checkpoint directory contains a valid `metadata.json` (with intact storage data) and that
    all Safetensors files referenced in the metadata actually exist in the directory. If no metadata file
    exists, it falls back to checking for at least one Safetensors file (to support single-card checkpoints).

    Args:
        checkpoint_dir: Path to the checkpoint directory to validate.

    Returns:
        Optional[str]: `None` if validation passes..

    Raises:
        NotADirectoryError: If `checkpoint_dir` does not exist or is not a directory.
        FileNotFoundError: If:
            1. No `metadata.json` exists AND no Safetensors files (*.safetensors) are found in the directory.
            2. A Safetensors file referenced in `metadata.json` is missing from the directory.
        RuntimeError: If `metadata.json` exists but is not valid JSON, or if its "storage_data" is malformed.
    """
    if not os.path.isdir(checkpoint_dir):
        raise NotADirectoryError(
            f"Checkpoint directory validation failed: '{checkpoint_dir}' is not a valid directory."
        )

    metadata_path = os.path.join(checkpoint_dir, "metadata.json")
    # Find all Safetensors files in the directory (for fallback check)
    safetensor_files = glob(os.path.join(checkpoint_dir, "*.safetensors"))

    if not os.path.exists(metadata_path):
        # No metadata → at least one Safetensors file must exist (single-card checkpoint case)
        if not safetensor_files:
            raise FileNotFoundError(
                f"Checkpoint directory validation failed: No Safetensors files "
                f"found in '{checkpoint_dir}'. Ensure the directory contains valid checkpoint files."
            )
        return None

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Checkpoint metadata validation failed: 'metadata.json' in '{checkpoint_dir}' has invalid JSON format. "
            f"Error details: {str(e)}"
        ) from e

    storage_data = metadata["storage_data"]
    for param_name, storage_info_list in storage_data.items():
        for storage_dict in storage_info_list:
            # Get full path to the referenced Safetensors file
            file_name = storage_dict["file_name"]
            file_path = os.path.join(checkpoint_dir, file_name)

            # Verify the file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Checkpoint file validation failed: Safetensors file '{file_name}' "
                    f"(referenced in metadata for parameter '{param_name}') is missing from "
                    f"checkpoint directory '{checkpoint_dir}'. Full expected path: '{file_path}'."
                )
    return None


def compile_model(model, dataset, mode, sink_mode, epoch=1, sink_size=1, do_eval=False, do_predict=False):
    """
    Compiles the MindSpore model, generates parallel strategy files (if applicable), and validates runtime context.

    This function ensures the model is built with the correct parallel strategy for distributed training or inference.

    Args:
        model: MindSpore Model object to compile.
        dataset: Input dataset for model compilation.
        mode: MindSpore runtime mode (either `ms.GRAPH_MODE` or `ms.PYNATIVE_MODE`). Strategy files are only
            generated in `GRAPH_MODE`.
        sink_mode: Whether to enable data sinking.
        epoch: Number of training epochs for model compilation (used in `model.build()`). Defaults to 1.
        sink_size: Batch size for data sinking (controls how many batches are sunk at once). Defaults to 1.
        do_eval: Whether to compile the model for evaluation (uses `infer_predict_layout()`). Defaults to `False`.
        do_predict: Whether to compile the model for prediction (uses `infer_predict_layout()`). Defaults to `False`.

    Returns:
        None.

    Raises:
        ValueError: If `sink_mode` is `False`.
        RuntimeError: If `dataset` is `None`.
        AttributeError: If `model` lacks required methods (`build()` or `infer_predict_layout()`).
    """
    # Validate mutually exclusive flags (eval and predict cannot both be True)
    if do_eval and do_predict:
        raise ValueError("'do_eval' and 'do_predict' cannot both be True; choose one or neither.")

    # Get current parallel mode from MindSpore context
    parallel_mode = context.get_auto_parallel_context('parallel_mode')
    supported_parallel_modes = ('semi_auto_parallel', 'auto_parallel', 'hybrid_parallel')

    # Skip compilation if in PyNative mode (no strategy files generated)
    if mode == ms.PYNATIVE_MODE:
        logger.warning(
            "Current runtime mode is PyNative. The `compile_model` function will not generate strategy files."
        )
        return

    # Skip compilation if parallel mode is unsupported (no strategy files generated)
    if parallel_mode not in supported_parallel_modes:
        logger.warning(
            f"Current parallel mode '{parallel_mode}' is unsupported. "
            f"Strategy files are only generated for: {supported_parallel_modes}. Compilation skipped."
        )
        return

    # Enforce sink_mode=True for distributed sliced weight loading
    if not sink_mode:
        raise ValueError(
            "When loading distributed sliced weights, 'sink_mode' must be set to True. "
            "Please enable sink_mode and retry."
        )

    # Validate dataset exists for compilation
    if not dataset:
        raise RuntimeError("Dataset cannot be None when compiling.")

    # Compile model and measure build time
    build_time_start = time.time()
    try:
        logger.info(".........Compiling model.........")
        if do_eval or do_predict:
            # Compile for evaluation/prediction (infer layout)
            model.infer_predict_layout(*dataset)  # Unpack dataset if it's a tuple (e.g., (data, label))
        else:
            # Compile for training (build model with dataset/epoch/sink settings)
            model.build(train_dataset=dataset, epoch=epoch, sink_size=sink_size)
    except AttributeError as e:
        raise AttributeError(
            f"Model is missing a required method: {str(e)}. "
            "Ensure the model has 'build()' (for training) and 'infer_predict_layout()' (for eval/predict)."
        ) from e
    except Exception as e:
        raise RuntimeError(f"Failed to compile model: {str(e)}") from e

    build_time_end = time.time()
    build_duration = build_time_end - build_time_start
    logger.info(f"Time spent compiling the model: {build_duration:.2f} seconds")
