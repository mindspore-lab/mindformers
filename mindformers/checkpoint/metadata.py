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

from mindspore.communication.management import get_group_size
from mindspore.common.dtype import all_types
from mindspore.parallel import Layout

from mindformers.tools.utils import set_safe_mode_for_file_or_dir
from mindformers.checkpoint.sharded_tensor import ShardedTensor, get_sharded_tensor_list_from_strategy_metadata
from mindformers.checkpoint.utils import (
    get_checkpoint_name,
    get_metadata_filename,
    get_sharded_tensor_shard_id
)

tensor_to_ms_type = {str(dtype): dtype for dtype in all_types}


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
    sharded_tensor_metas = [
        item
        for sublist in sharded_tensor_metas
        for item in sublist
    ]
    for _, tensor_meta in enumerate(sharded_tensor_metas):
        param_name = tensor_meta.key
        new_chunk = {
            "global_offset": tensor_meta.global_offset,
            "local_shape": tensor_meta.local_shape
        }
        if param_name not in state_dict_metadata:
            state_dict_metadata[param_name] = {
                "properties": {
                    "dtype": str(tensor_meta.dtype),
                    "replica_id": tensor_meta.replica_id,
                    "allow_shape_mismatch": tensor_meta.allow_shape_mismatch,
                    "allow_to_save": tensor_meta.allow_to_save
                },
                "global_shape": tensor_meta.global_shape,
                "axis_fragmentations": tensor_meta.axis_fragmentations,
                "layout": _serialize_sharded_tensor_layout(tensor_meta.layout),
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



def load_metadata(checkpoints_path, iteration):
    """
    Load sharded tensor metadata and parameter file mappings from a metadata.json file.

    This function reads and parses a metadata.json file to reconstruct sharded tensor
    metadata and parameter file mapping information. It deserializes the stored metadata
    back into ShardedTensor objects and parameter-to-file mappings that can be used
    for checkpoint loading operations.

    Args:
        checkpoints_path (str): Path to the directory containing checkpoint files and metadata.
            The function will look for a metadata.json file in this directory.

        iteration (int): Training iteration number used to locate the specific metadata file.
            This corresponds to the iteration number used when the metadata was saved.

    Returns:
        A tuple containing two dictionaries:
          - First dictionary: Maps (parameter_name, global_offset) tuples to ShardedTensor instances.
            Each ShardedTensor contains complete metadata about a tensor chunk
            including its shape, layout, and positioning information.

          - Second dictionary: Maps parameter IDs to lists of storage information dictionaries.
            Each storage info dict contains:
              - "file_name": Name of the file storing the parameter data.
              - "storage_rank": Rank of the device that stored this parameter.

    Raises:
        FileNotFoundError: If the metadata.json file cannot be found at the specified path.

        FormatError: If the metadata file contains invalid JSON format or cannot be parsed properly.

        KeyError: If required metadata fields are missing from the JSON structure.
    """
    metadata_file = get_metadata_filename(checkpoints_path, iteration)

    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except json.JSONDecodeError as e:
        raise FormatError(f"Invalid JSON format in metadata file: {metadata_file}") from e

    state_dict_metadata = metadata.get("state_dict_metadata")
    storage_data = metadata.get("storage_data")

    sharded_tensor_metas = {}
    for param_name, meta in state_dict_metadata.items():
        properties = meta.get("properties")

        global_shape = meta.get("global_shape")
        axis_fragmentations = meta.get("axis_fragmentations")
        layout = _deserialize_sharded_tensor_layout(meta.get("layout"))
        dtype_str = properties.get("dtype")
        dtype = tensor_to_ms_type.get(dtype_str)

        chunks = meta.get("chunk")
        for chunk in chunks:
            global_offset = chunk.get("global_offset")
            local_shape = chunk.get("local_shape")

            sharded_tensor = ShardedTensor(
                key=param_name,
                dtype=dtype,
                local_shape=tuple(local_shape),
                global_shape=tuple(global_shape),
                global_offset=tuple(global_offset),
                axis_fragmentations=tuple(axis_fragmentations),
                replica_id=properties.get("replica_id"),
                allow_shape_mismatch=properties.get("allow_shape_mismatch"),
                allow_to_save=properties.get("allow_to_save"),
                layout=layout
            )

            sharded_tensor_metas[(param_name, tuple(global_offset))] = sharded_tensor

    param_file_mappings = {}
    for param_id, data in storage_data.items():
        param_file_mappings[param_id] = data

    return sharded_tensor_metas, param_file_mappings


def get_total_shard_metadata(global_strategy_info, filter_func):
    """Get all shard metadata."""
    npu_nums = get_group_size()
    sharded_tensor_metas = list()

    for cur_npu_rank in range(0, npu_nums):
        org_cur_rank_strategy_layout = global_strategy_info[cur_npu_rank]
        cur_rank_strategy_layout = [
            dict([item])
            for item in org_cur_rank_strategy_layout.items()
        ]

        # Get Sharded tensors from strategy metadata of current rank.
        cur_rank_sharded_tensors = get_sharded_tensor_list_from_strategy_metadata(
            param_infos=cur_rank_strategy_layout,
            cur_npu_rank=cur_npu_rank,
            filter_func=filter_func
        )

        sharded_tensor_metas.append(cur_rank_sharded_tensors)

    return sharded_tensor_metas


def get_total_params_file_mapping_info(sharded_tensor_metas, user_prefix, model_keys):
    """Get all shard metadata file mappings list."""
    if sharded_tensor_metas is None:
        return None

    npu_nums = get_group_size()
    param_file_mappings = list()
    for cur_rank_sharded_tensor_list in sharded_tensor_metas:
        # Get mappings of parameter file of current rank.
        for cur_npu_rank, sharded_tensor in enumerate(cur_rank_sharded_tensor_list):
            if model_keys and sharded_tensor.key not in list(model_keys):
                ckpt_name = get_checkpoint_name(None, user_prefix, cur_npu_rank, npu_nums, 'Optimizer')
            else:
                ckpt_name = get_checkpoint_name(None, user_prefix, cur_npu_rank, npu_nums, 'Model')

            param_file_mappings.append(
                (ckpt_name + '.safetensors', cur_npu_rank, (sharded_tensor.key, sharded_tensor.global_offset))
            )

    return param_file_mappings
