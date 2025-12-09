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
"""metadata apis."""
import os
import json
import tempfile
from glob import glob
from safetensors import safe_open

from mindspore.communication.management import get_group_size
from mindspore.common.dtype import all_types
from mindspore.parallel import Layout

from mindformers.tools.logger import logger
from mindformers.tools.utils import set_safe_mode_for_file_or_dir
from mindformers.checkpoint.sharded_tensor import build_sharded_tensor
from mindformers.checkpoint.utils import (
    get_checkpoint_name,
    get_sharded_tensor_shard_id,
    FileType
)

tensor_to_ms_type = {str(dtype).lower(): dtype for dtype in all_types}


def _serialize_sharded_tensor_layout(layout):
    """Serialize layout object to dictionary for storage."""
    layout_dict = layout.to_dict()
    return layout_dict


def _reverse_index_dict(tup):
    """Create reverse index mapping from tuple."""
    n = len(tup)
    return {n - i - 1: value for i, value in enumerate(tup)}


def _replace_numbers_in_tuple(tensor_map, alias_name):
    """Replace numeric indices in tensor_map with alias names recursively."""
    result = []
    for item in tensor_map:
        if isinstance(item, list):
            result.append(_replace_numbers_in_tuple(item, alias_name))
        elif isinstance(item, (int, float)):
            if item == -1:
                result.append("None")
            else:
                result.append(alias_name.get(item, item))
        else:
            result.append(item)

    return tuple(result)


def _deserialize_sharded_tensor_layout(layout_dict):
    """Reconstruct layout object from dictionary representation."""
    device_matrix, tensor_map, interleaved_parallel, alias_name, rank_list = layout_dict.values()

    layout = Layout(tuple(device_matrix), tuple(alias_name), rank_list)
    alias_name_map = _reverse_index_dict(alias_name)
    tensor_map = _replace_numbers_in_tuple(tensor_map, alias_name_map)

    layout0 = layout(*tensor_map)
    layout0.interleaved_parallel = interleaved_parallel

    return layout0


def save_metadata(sharded_tensor_metas, param_file_mappings, meta_data_path):
    """
    Save sharded tensor metadata into a JSON-formatted metadata file.

    This function processes sharded tensor metadata and parameter file mappings
    to create a comprehensive metadata file
    that enables proper reconstruction of distributed model checkpoints.

    The metadata includes tensor properties, layout information, chunk details, and storage mappings.

    Args:
        sharded_tensor_metas (List[List[ShardedTensor]]): Nested list of sharded tensor metadata objects.
            Each ShardedTensor contains information about a tensor chunk
            including its key, data type, shapes, offsets, and parallel configuration.

        param_file_mappings (List[Tuple[str, int, Tuple[str, Tuple]]]): List of parameter file mappings
            that specify the storage location and identification information
            for each parameter chunk in the distributed checkpoint system.
            Each tuple establishes the relationship between parameter chunks and their physical storage files.

        meta_data_path (str): Directory path where the metadata files are stored.
            The metadata file will be saved in this directory.

    Returns:
        None: The function writes metadata directly to a JSON file and does not return any value.

    Raises:
        OSError: If the metadata file cannot be written to the specified path
                 due to file system errors, permission issues, or disk space problems.
    """
    state_dict_metadata = {}
    sharded_tensor_list = []
    for _, cur_rank_sharded_tensor_metas in sharded_tensor_metas.items():
        for _, sharded_tensor in cur_rank_sharded_tensor_metas.items():
            if isinstance(sharded_tensor, list):
                sharded_tensor_list.extend(sharded_tensor)
            else:
                sharded_tensor_list.append(sharded_tensor)

    for sharded_tensor in sharded_tensor_list:
        param_name = sharded_tensor.key
        new_chunk = {
            "global_offset": sharded_tensor.global_offset,
            "local_shape": sharded_tensor.local_shape
        }
        if param_name not in state_dict_metadata:
            state_dict_metadata[param_name] = {
                "properties": {
                    "dtype": str(sharded_tensor.dtype),
                    "replica_id": sharded_tensor.replica_id,
                    "allow_shape_mismatch": sharded_tensor.allow_shape_mismatch,
                    "allow_to_save": sharded_tensor.allow_to_save
                },
                "global_shape": sharded_tensor.global_shape,
                "axis_fragmentations": sharded_tensor.axis_fragmentations,
                "layout": _serialize_sharded_tensor_layout(sharded_tensor.layout),
                "chunk": [new_chunk]
            }
        elif param_name in state_dict_metadata:
            existing_chunks = state_dict_metadata[param_name]["chunk"]
            if not any(
                    chunk["global_offset"] == new_chunk["global_offset"] and
                    chunk["local_shape"] == new_chunk["local_shape"]
                    for chunk in existing_chunks
            ):
                state_dict_metadata[param_name]["chunk"].append(new_chunk)

    storage_data = {}
    for param_file_mapping in param_file_mappings:
        file_name = param_file_mapping[0]
        storage_rank = param_file_mapping[1]
        param_id = get_sharded_tensor_shard_id(param_file_mapping[2][0], param_file_mapping[2][1])
        if param_id not in storage_data:
            storage_data[param_id] = [{"file_name": file_name, "storage_rank": storage_rank}]
        else:
            storage_data[param_id].append({"file_name": file_name, "storage_rank": storage_rank})

    metadata = {"state_dict_metadata": state_dict_metadata, "storage_data": storage_data}

    metadata_str = json.dumps(metadata, ensure_ascii=False, indent=4)
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=os.path.dirname(meta_data_path), delete=False) as tmp_file:
            tmp_file.write(metadata_str)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())  # Ensure data is written to disk
            temp_filename = tmp_file.name
        os.replace(temp_filename, meta_data_path)
        set_safe_mode_for_file_or_dir(meta_data_path)
    except OSError as e:
        raise OSError(f"Failed to write metadata file '{meta_data_path}': {str(e)}") from e


def load_metadata(metadata_file: str):
    """
    Load sharded tensor metadata and parameter file mappings from a metadata.json file.

    This function reads and parses a metadata.json file to reconstruct sharded tensor
    metadata and parameter file mapping information. It deserializes the stored metadata
    back into ShardedTensor objects and parameter-to-file mappings that can be used
    for checkpoint loading operations.

    Args:
        metadata_file (str): Path to the metadata.json under checkpoint directory.

    Returns:
        A tuple containing two dictionaries:
          - First dictionary: Maps parameter_name to ShardedTensor instances.
            Each ShardedTensor contains complete metadata about a tensor chunk
            including its shape, layout, and positioning information.

          - Second dictionary: Maps parameter IDs to lists of storage information dictionaries.
            Each storage info dict contains:
              - "file_name": Name of the file storing the parameter data.
              - "storage_rank": Rank of the device that stored this parameter.

    Raises:
        RuntimeError: If metadata file has invalid JSON format or cannot be parsed.
        KeyError: If required metadata fields are missing from the JSON structure.
    """
    logger.info("..........Load Metadata from Metadata Json..........")

    # Load and parse metadata JSON file
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON format in metadata file: {metadata_file}") from e

    # Extract required metadata components
    state_dict_metadata = metadata.get("state_dict_metadata")
    storage_data = metadata.get("storage_data")

    sharded_tensor_metas = {}

    # Process each parameter's metadata
    for param_name, meta in state_dict_metadata.items():
        # Extract core metadata properties
        properties = meta.get("properties")
        global_shape = meta.get("global_shape")
        axis_fragmentations = meta.get("axis_fragmentations")
        layout = _deserialize_sharded_tensor_layout(meta.get("layout"))

        # Convert dtype string to appropriate type
        dtype_str = properties.get("dtype")
        dtype = tensor_to_ms_type.get(dtype_str.lower())

        # Process each chunk of the sharded tensor
        for chunk in meta.get("chunk"):
            global_offset = chunk.get("global_offset")
            local_shape = chunk.get("local_shape")

            # Create ShardedTensor instance with chunk-specific metadata
            sharded_tensor = build_sharded_tensor(
                param_name=param_name,
                param_dtype=dtype,
                local_shape=local_shape,
                global_shape=global_shape,
                global_offset=global_offset,
                axis_fragmentations=axis_fragmentations,
                replica_id=properties.get("replica_id"),
                allow_shape_mismatch=properties.get("allow_shape_mismatch"),
                allow_to_save=properties.get("allow_to_save"),
                layout=layout
            )

            # Add to collection (initialize list if first entry)
            if param_name in sharded_tensor_metas:
                sharded_tensor_metas[param_name].append(sharded_tensor)
            else:
                sharded_tensor_metas[param_name] = [sharded_tensor]

    return sharded_tensor_metas, storage_data


def generate_default_metadata_from_checkpoint(checkpoint_dir: str) -> tuple[dict, dict]:
    """
    Loads metadata from safetensors checkpoint files in the specified directory.

    Extracts tensor information from all .safetensors files in the checkpoint directory
    and constructs metadata objects for each tensor, including shape, dtype, and
    storage information.

    Args:
        checkpoint_dir: Path to the directory containing checkpoint files

    Returns:
        A tuple containing two dictionaries:
        - sharded_tensor_metas: Maps parameter names to lists of ShardedTensor objects
        - param_file_mappings: Maps parameter names with storage rank to file information

    Raises:
        NotADirectoryError: If the specified checkpoint directory does not exist
        FileNotFoundError: If no safetensors files are found in the directory
        RuntimeError: If there are duplicate parameter names or tensor loading fails
    """
    if not os.path.isdir(checkpoint_dir):
        raise NotADirectoryError(
            f"Checkpoint directory '{checkpoint_dir}' does not exist or is not a directory.")

    logger.info("..........Load Metadata from Checkpoint Files..........")

    # Find all safetensor files in the checkpoint directory
    safetensor_pattern = os.path.join(checkpoint_dir, "*.safetensors")
    safetensor_files = glob(safetensor_pattern)

    # Verify we found safetensor files
    if not safetensor_files:
        raise FileNotFoundError(
            f"No Safetensors files found in directory '{checkpoint_dir}'. "
            f"Ensure files match pattern: {safetensor_pattern}."
        )

    # Initialize metadata storage structures
    sharded_tensor_metas: dict = {}
    param_file_mappings: dict = {}

    # Process each safetensor file
    for safetensor_file in safetensor_files:
        file_basename = os.path.basename(safetensor_file)
        logger.info(f"Extracting metadata from Safetensors file: {file_basename}")

        # Open the safetensor file and process each tensor
        with safe_open(safetensor_file, framework="np", device="cpu") as f:
            for param_name in f.keys():
                try:
                    # Load the tensor to extract its properties
                    tensor = f.get_tensor(param_name)  # Load as numpy tensor
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load tensor '{param_name}' from file '{file_basename}'"
                    ) from e

                # Extract tensor properties
                tensor_shape = tensor.shape
                ms_dtype = tensor_to_ms_type.get(str(tensor.dtype))
                global_offset = (0,)
                axis_fragmentations = (1,) * len(tensor_shape)

                # Create sharded tensor metadata object
                sharded_tensor = build_sharded_tensor(
                    param_name=param_name,
                    param_dtype=ms_dtype,
                    local_shape=tensor_shape,
                    global_shape=tensor_shape,
                    global_offset=global_offset,
                    axis_fragmentations=axis_fragmentations,
                    layout=None
                )

                # Check for duplicate parameters
                if param_name in sharded_tensor_metas:
                    raise RuntimeError(f"Duplicate parameter_name found: {param_name}.")

                # Store metadata with fixed storage rank 0
                sharded_tensor_metas[param_name] = [sharded_tensor]
                param_file_mappings[str((param_name, (0,)))] = [
                    {"file_name": file_basename, "storage_rank": 0}
                ]

    return sharded_tensor_metas, param_file_mappings


def get_total_params_file_mapping_info(sharded_tensor_metas, user_prefix, model_keys):
    """Get all shard metadata file mappings list."""
    if sharded_tensor_metas is None:
        return None

    npu_nums = get_group_size()
    param_file_mappings = []
    for cur_npu_rank, cur_rank_sharded_tensors in sharded_tensor_metas.items():
        # Get mappings of parameter file of current rank.
        for sharded_tensor in cur_rank_sharded_tensors.values():
            if model_keys and sharded_tensor.key not in list(model_keys):
                ckpt_name = get_checkpoint_name(None, user_prefix, cur_npu_rank, npu_nums, FileType.OPTIMIZER)
            else:
                ckpt_name = get_checkpoint_name(None, user_prefix, cur_npu_rank, npu_nums, FileType.MODEL)

            param_file_mappings.append(
                (ckpt_name + '.safetensors', cur_npu_rank, (sharded_tensor.key, sharded_tensor.global_offset))
            )

    return param_file_mappings
