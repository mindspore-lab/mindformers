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
import os
import json
import tempfile
from time import time
from typing import Callable, Union, Dict, Optional, Tuple, Any, List
from dataclasses import dataclass

import threading
from multiprocessing import active_children
from safetensors import safe_open

import mindspore as ms
from mindspore import Tensor, Parameter, load_param_into_net
from mindspore.common import dtype as mstype
from mindspore.nn import Cell
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.communication.management import get_rank, get_group_size
from mindspore.communication import comm_func
from mindspore import save_checkpoint as ms_save_checkpoint

from mindformers.tools.logger import logger
from mindformers.checkpoint.reshard import ReshardHandler
from mindformers.utils.file_utils import is_publicly_accessible_path
from mindformers.utils.parallel_utils import barrier_world
from mindformers.tools.utils import (
    get_output_subpath,
    get_real_rank,
    set_safe_mode_for_file_or_dir,
    get_real_group_size
)
from mindformers.checkpoint.utils import (
    get_checkpoint_iter_dir,
    get_checkpoint_name,
    get_checkpoint_tracker_filename,
    get_latest_iteration_from_tracker,
    get_common_filename,
    check_checkpoints_dir_max_num,
    get_metadata_filename,
    verify_ckpt_valid,
    FileType
)
from mindformers.checkpoint.fully_parallel import BalancedSaveStrategy, apply_balance_shard_strategy
from mindformers.checkpoint.metadata import (
    save_metadata,
    load_metadata,
    generate_default_metadata_from_checkpoint,
    get_total_params_file_mapping_info,
)
from mindformers.checkpoint.sharded_tensor import (
    get_strategy_info_from_sharded_tensor,
    ShardedTensor,
    get_sharded_tensor_from_cell,
    get_cur_sharded_tensor,
    get_cur_sharded_tensor_after_balanced,
    get_param_redundancy_after_balanced
)
from mindformers.checkpoint.broadcast import single_parameter_broadcast


@dataclass
class CommonInfo:
    """
    Save/load common info for checkpoint.
    """
    epoch_num: int = None
    """The number of training epochs."""

    step_num: int = None
    """Training step number in current epoch."""

    global_step: int = None
    """Training step number in global epochs."""

    loss_scale: float = None
    """Magnification factor of gradients."""

    global_batch_size: int = None
    """The total batch size during multi-NPU training."""

    def save_common(self, common_filename: str):
        """
        Save common info to 'common.json'.

        Args:
            common_filename (str): The file path of 'common.json' to save.
        """
        logger.info(f"Saving common info to '{common_filename}'.")

        common_info_str = json.dumps(self.__dict__, ensure_ascii=False, indent=4)
        with tempfile.NamedTemporaryFile(mode='w', dir=os.path.dirname(common_filename), delete=False) as tmp_file:
            tmp_file.write(common_info_str)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())  # Ensure data is written to disk
            temp_filename = tmp_file.name
        os.replace(temp_filename, common_filename)
        set_safe_mode_for_file_or_dir(common_filename)

        logger.info(f"'common.json' successfully saved at: '{common_filename}'.")

    @classmethod
    def load_common(cls, common_filename: str):
        """
        Load common info from 'common.json'.

        Args:
            common_filename(str): The file path of 'common.json' to load.
        """
        logger.info(f"Loading common info from '{common_filename}'.")

        try:
            with open(common_filename, 'r', encoding='utf-8') as f:
                common_data = json.load(f)
            logger.info(f"'common.json' successfully loaded as:\n{common_data}")

            return cls(**common_data)

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Can not find 'common.json' file at: '{common_filename}'.") from e

        except json.JSONDecodeError as e:
            raise ValueError(f"JSON format failed: {e}") from e


class AsyncSaveManager:
    """
    Manager async save checkpoint process.
        1. Sync save process in all ranks and execute finalize functions before next save action.
        2. Check save process and execute finalize functions at the beginning of each step.
    """

    def __init__(self, async_save: Union[bool, str]):
        """
        Args:
            async_save (Union[bool, str]): Can be False, True(default 'thread'), 'thread', 'process'
        """
        self.async_save = async_save
        self.idx = 0
        self.finalize_fns = []
        self.is_finalized = True
        self.start_time = 0

    def add_finalize_fn(self, finalize_fn: Callable) -> None:
        """
        Adds a new finalize function to the manager.
        Finalize functions will be executed once after current save action.
        Finalize functions are reset when prepare_before_save is called.

        Args:
            finalize_fn (Callable): Function to add to the manager.
                This function will be called once after current save action.
        """
        logger.info(f"(idx:{self.idx})add finalize function")
        self.finalize_fns.append(finalize_fn)

    def prepare_before_save(self) -> None:
        """
        Prepare before a new save checkpoint action.
            1. Wait save process in all ranks and execute finalize functions
            2. Reset flags and finalize functions
        """
        logger.info(f"(idx:{self.idx})prepare before save")
        if not self.is_finalized:
            logger.info(f"(idx:{self.idx})previous save action is not finalized, wait finish synchronized...")
            self.maybe_finalize(wait_finish=True)

        self.is_finalized = False
        self.idx = self.idx + 1
        self.finalize_fns = []
        self.start_time = time()
        logger.info(f"(idx:{self.idx})prepare before save done")

    def maybe_finalize(self, wait_finish: bool = False) -> None:
        """
        Execute finalize functions if all ranks finish async save.

        Args:
            wait_finish (bool): If True, wait all async save process finish.
        """
        logger.info(f"(idx:{self.idx})self.is_finalized: {self.is_finalized}")
        if not self.is_finalized:
            start_time = time()
            is_alive = self.check_async_save_alive(wait_finish)
            logger.info(f"(idx:{self.idx})async_save: {self.async_save}, is_alive: {is_alive}, "
                        f"check is_alive cost time: {time() - start_time:.3f}s")
            start_time = time()
            is_all_done = self.sync_all_async_save_status(is_alive)
            logger.info(f"(idx:{self.idx})after all_reduce, is_all_done:{is_all_done}, "
                        f"cost time: {time() - start_time:.3f}s")

            if is_all_done:
                logger.info(f"(idx:{self.idx})execute finalize functions!")
                start_time = time()
                # Execute finalize functions
                for finalize_fn in self.finalize_fns:
                    finalize_fn()
                self.is_finalized = True
                logger.info(f"(idx:{self.idx})finalize functions done, cost time: {time() - start_time:.3f}s")
                logger.info(f"(idx:{self.idx})async save total time: {time() - self.start_time:.3f}s")

    def check_async_save_alive(self, wait_finish: bool = False) -> bool:
        """
        Check if current async save action is still running.

        Args:
            wait_finish (bool): If True, wait all async save process finish.

        Returns:
            A bool flag. True if current async save action is still running, False if it is finished.
        """
        if self.async_save is False:
            return False

        # Async process
        if self.async_save == "process":
            for process in active_children():
                if process.name == "asyn_save_ckpt":
                    if wait_finish:
                        process.join()
                        return False
                    return True
            return False

        # Async thread
        for thread in threading.enumerate():
            if thread.name == "asyn_save_ckpt":
                if wait_finish:
                    thread.join()
                    return False
                return True
        return False

    def sync_all_async_save_status(self, is_alive: int) -> bool:
        """Check if all ranks have completed async save checkpoint

        Args:
            is_alive (bool): if True, the current async save action is not completed

        Returns:
            A bool flag. True if all ranks are done, False if at least one rank is not completed.
        """
        if self.async_save is False:
            return True
        if get_group_size() == 1:
            return not is_alive

        ten = Tensor([is_alive], dtype=mstype.int8)
        ten, _ = comm_func.all_reduce(ten)

        return ten[0] == 0


def save_checkpoint(iteration: int, network: Cell, optimizer: Optimizer = None,
                    async_save_manager: AsyncSaveManager = None, common_info: CommonInfo = None,
                    keep_max_num: int = 5, user_prefix: str = None, save_checkpoint_path: str = None,
                    sharded_tensor_metas: Dict = None, remove_redundancy: bool = False):
    """
    Saves the current state of the training process,
        including the model, optimizer, and learning rate scheduler, to a checkpoint file.

    Args:
        iteration (int): The current training iteration step.
        network (Cell): The MindSpore model object to be saved.
        optimizer (Optimizer, optional): The optimizer object associated with the model. Defaults to None.
        async_save_manager (AsyncSaveManager, optional): The manager of async save if save weight in async way.
        common_info (CommonInfo): The instance of common info to save step_num, epoch_num, global_step and so on.
        keep_max_num (int): The maximum number of weights that can be stored.
        user_prefix (str): The prefix of user assign to use for the checkpoint file name.
        save_checkpoint_path (str): The user can specify the path to save the weights.
            If None, the default path is 'output_dir/checkpoint'.
            And 'output_dir' is configured in yaml and defaults to './output' in the execution script path.
        sharded_tensor_metas (Dict): The ShardedTensor metas of this network.
        remove_redundancy (bool): Whether to remove redundancy of saving checkpoint.
    """
    logger.info('....... Start to save checkpoint as new format .......')

    # Get the root path of all checkpoints to save.
    if save_checkpoint_path:
        checkpoints_root_path = os.path.realpath(save_checkpoint_path)
    else:
        checkpoints_root_path = get_output_subpath("checkpoint", append_rank=False)

    if not is_publicly_accessible_path(checkpoints_root_path):
        raise RuntimeError("The 'save_checkpoint_megatron_format' feature is not currently supported "
                           "in 'non-shared storage environments' with multiple hosts.")
    logger.info(f"The root path of saved checkpoints is: '{checkpoints_root_path}'.")

    # Generate current iteration saving path.
    cur_iter_checkpoint_dir = get_checkpoint_iter_dir(checkpoints_root_path, iteration)
    logger.info(f"At current iteration '{iteration}', the weight will be saved in: '{cur_iter_checkpoint_dir}'.")

    # Whether to use async save.
    use_async_save = async_save_manager is not None

    if get_rank() == 0:
        os.makedirs(cur_iter_checkpoint_dir, exist_ok=True)
        set_safe_mode_for_file_or_dir(checkpoints_root_path)
        set_safe_mode_for_file_or_dir(cur_iter_checkpoint_dir)
    barrier_world(f"Rank_0 to ensure path '{cur_iter_checkpoint_dir}' is exists.")
    # Fix cache coherency issues with shared storage.
    # Force refresh the disk cache of the current node to ensure that the path can be accessed correctly.
    os.listdir(os.path.dirname(checkpoints_root_path))

    # Prepare async save manager before save.
    def iter_finalize_func():
        """Save checkpoint finalize function."""
        tracker_filename = get_checkpoint_tracker_filename(checkpoints_root_path)
        logger.info(f"save checkpoint tracker file to {tracker_filename}")
        with open(tracker_filename, "w", encoding='utf-8') as f:
            f.write(str(iteration))
        set_safe_mode_for_file_or_dir(tracker_filename)
        if use_async_save:
            logger.info(f"successfully async saved checkpoint from step '{iteration}' to '{checkpoints_root_path}'.")
        else:
            logger.info(f"successfully sync saved checkpoint from step '{iteration}' to '{checkpoints_root_path}'.")

    if use_async_save:
        async_save_manager.prepare_before_save()
        if get_rank() == 0:
            async_save_manager.add_finalize_fn(iter_finalize_func)

    # Check if the number of saved folders has exceeded, and delete the oldest one.
    if get_rank() == 0:
        # NOTE: Currently only supports shared storage scenarios.
        check_checkpoints_dir_max_num(keep_max_num, checkpoints_root_path)
        # If the current iteration checkpoint directory be removed, raise an error to remind user
        # to check whether the file path for saving checkpoints is configured correctly.
        if not os.path.exists(cur_iter_checkpoint_dir):
            raise FileNotFoundError(f"Can not find current iteration checkpoint directory: "
                                    f"'{cur_iter_checkpoint_dir}'. Please check your configuration item "
                                    f"'save_checkpoint_path' under the 'CheckpointMonitor' in yaml, "
                                    f"to ensure that there is no weight left by other tasks under the path.")
    barrier_world("Rank_0 checking saved weights iteration num...")

    # Save model weight.
    logger.info("....... Start to save model weight .......")
    model_keys = network.parameters_dict().keys()
    start_save_ckpt_time = time()

    if remove_redundancy and sharded_tensor_metas is not None:
        remove_model_redundancy = BalancedSaveStrategy(
            network,
            user_prefix=user_prefix,
            checkpoint_path=checkpoints_root_path,
            filter_func=lambda x: x in list(model_keys),
            file_type=FileType.MODEL
        )
        remove_model_redundancy.save(iteration)
    else:
        model_ckpt_filename = get_checkpoint_name(
            cur_iter_checkpoint_dir, user_prefix, get_rank(), get_group_size(), FileType.MODEL
        )
        ms_save_checkpoint(
            network,
            model_ckpt_filename,
            async_save=use_async_save,
            format="safetensors"
        )
        logger.info(f"Model checkpoint successfully saved at '{model_ckpt_filename}.safetensors'.")

    # Save optimizer weight.
    if optimizer is not None:
        if remove_redundancy and sharded_tensor_metas is not None:
            # Optimizer weight remove redundancy.
            remove_optimizer_redundancy = BalancedSaveStrategy(
                optimizer,
                user_prefix=user_prefix,
                checkpoint_path=checkpoints_root_path,
                filter_func=lambda x: x not in list(model_keys),
                file_type=FileType.OPTIMIZER
            )
            remove_optimizer_redundancy.save(iteration)
        else:
            # Optimizer weight has redundancy.
            logger.warning("....... Start to save optimizer weight .......")
            optimizer_ckpt_filename = get_checkpoint_name(
                cur_iter_checkpoint_dir, user_prefix, get_rank(), get_group_size(), FileType.OPTIMIZER
            )
            ms_save_checkpoint(
                optimizer,
                optimizer_ckpt_filename,
                async_save=use_async_save,
                format="safetensors",
                choice_func=lambda x: x not in list(model_keys)
            )
            logger.info(f"Optimizer checkpoint successfully saved at '{optimizer_ckpt_filename}.safetensors'.")
    else:
        logger.warning("Optimizer weight will not be save!")

    # Save 'common.json'.
    if get_rank() == 0:
        logger.info("...... Start saving common info ......")
        start_save_common_info_time = time()

        common_filename = get_common_filename(checkpoints_root_path, iteration)
        common_info.save_common(common_filename)

        logger.info(f"The 'common.json' is saved at '{common_filename}'.")
        logger.info(f"Save common info cost time: {time() - start_save_common_info_time:.3f}s.")

    # Save 'metadata.json'.
    if not remove_redundancy:
        metadata_file_path = get_metadata_filename(checkpoints_root_path, iteration)
        save_metadata_json(sharded_tensor_metas, model_keys, user_prefix, metadata_file_path)

    # Save tracker file in sync save process.
    if not use_async_save:
        barrier_world("All ranks for sync save checkpoint.")
        logger.info("Rank_0 execute finalize func.")
        if get_rank() == 0:
            iter_finalize_func()
        logger.info(f"Save checkpoint cost time: {time() - start_save_ckpt_time:.3f}s.")


def save_metadata_json(sharded_tensor_metas, model_keys, user_prefix, metadata_file_path):
    """Saving metadata.json used `get_strategy_metadata` API."""
    if sharded_tensor_metas is not None:
        logger.info("...... Start saving metadata ......")
        if get_rank() == 0:
            param_file_mappings = get_total_params_file_mapping_info(sharded_tensor_metas, user_prefix, model_keys)
            save_metadata(sharded_tensor_metas, param_file_mappings, metadata_file_path)

        # Barrier here to ensure 'metadata.json' saved, then continue training.
        barrier_world("Rank_0 is saving 'metadata.json' ...")
        logger.info(f"The 'metadata.json' saved successfully at '{metadata_file_path}'.")
    else:
        logger.info("No need to save metadata.json for single card.")


def load_safetensor(
        checkpoint_path: str,
        param_name: Optional[Union[str, List[str]]] = None,
        index_tuple: Optional[Union[Tuple[Tuple[int, int], ...], List[Tuple[Tuple[int, int], ...]]]] = None,
        dtype: Optional[Union[Any, List[Any]]] = mstype.float32
) -> Dict[str, Parameter]:
    """
    Loads tensors from a Safetensors file into MindSpore Parameters.

    This function reads a Safetensors file and converts specified tensors into MindSpore
    Parameter objects with the specified data type. It can load either specific tensors
    (with optional slicing) or all tensors from the file.

    Args:
        checkpoint_path: Path to the Safetensors file to load
        param_name: Optional name or list of names of specific tensors to load.
                    If None, loads all tensors from the file.
        index_tuple: Optional slicing indices for tensors. Can be a single tuple of (start, end)
                    index pairs or a list of such tuples. Must match the dimension of the
                    corresponding tensor if provided.
        dtype: Target data type(s) for the loaded tensors. Can be a single type or a list of types.
               Defaults to mstype.float32.

    Returns:
        Dictionary mapping tensor names to MindSpore Parameter objects

    Raises:
        ValueError: If the file doesn't exist, if index_tuple dimension doesn't match
                   tensor dimension, or if parameter lists have mismatched lengths
        KeyError: If a specified parameter name doesn't exist in the file
    """
    # Validate file existence
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Safetensors file not found at: {checkpoint_path}")

    # Warn about unused index_tuple when no parameter name is specified
    if param_name is None and index_tuple is not None:
        logger.warning("index_tuple is ignored when param_name is None (loading all parameters)")

    def _convert_to_list(param: Optional[Union[object, List[object]]]) -> Optional[List[object]]:
        """Convert parameter to list if it's not already a list"""
        if param is None:
            return None
        return [param] if not isinstance(param, list) else param

    def _align_list_length(param_list: List[object], target_length: int) -> List[object]:
        """Align list length to match target length, supporting broadcasting of single-element lists"""
        if len(param_list) == target_length:
            return param_list
        if len(param_list) == 1:
            return param_list * target_length
        raise ValueError(f"List length {len(param_list)} cannot be aligned to target length {target_length}")

    # Unify parameters to list format for consistent processing
    param_name_list: Optional[List[str]] = _convert_to_list(param_name)
    index_tuple_list: Optional[List[Tuple[Tuple[int, int], ...]]] = _convert_to_list(index_tuple)
    dtype_list: Optional[List[Any]] = _convert_to_list(dtype)

    # Validate parameter list length consistency
    if param_name_list is not None:
        # Validate index list length
        if index_tuple_list is not None and len(index_tuple_list) != len(param_name_list):
            raise ValueError(
                f"Length of index_tuple ({len(index_tuple_list)}) must match "
                f"length of param_name ({len(param_name_list)})"
            )

        # Validate data type list length
        if dtype_list is not None:
            dtype_list = _align_list_length(dtype_list, len(param_name_list))

    weights: Dict[str, Parameter] = {}

    # Load data from Safetensors file
    with safe_open(checkpoint_path, framework="np", device="cpu") as f:
        if param_name_list:
            # Load specified parameters
            for idx, param_name_ in enumerate(param_name_list):
                # Get data type for current parameter
                cur_dtype = dtype_list[idx] if dtype_list and dtype_list[idx] else mstype.float32

                # Load tensor from file
                try:
                    tensor_np = f.get_tensor(param_name_)
                except KeyError as e:
                    raise KeyError(f"Parameter '{param_name_}' not found in Safetensors file") from e

                # Apply slicing if specified
                if index_tuple_list is not None:
                    index_tuple = index_tuple_list[idx]
                    if len(index_tuple) != tensor_np.ndim:
                        raise ValueError(
                            f"Index tuple dimension ({len(index_tuple)}) does not match "
                            f"parameter '{param_name_}' dimension ({tensor_np.ndim})"
                        )
                    # Create slice objects and apply
                    slices = tuple(slice(start, end) for start, end in index_tuple)
                    tensor_np = tensor_np[slices]

                # Convert to MindSpore Parameter
                weights[param_name_] = Parameter(
                    ms.from_numpy(tensor_np).astype(cur_dtype), name=param_name_, requires_grad=False
                )
        else:
            # Load all parameters
            cur_dtype = dtype if not isinstance(dtype, list) else dtype[0]
            for key in f.keys():
                tensor_np = f.get_tensor(key)
                weights[key] = Parameter(
                    ms.from_numpy(tensor_np).astype(cur_dtype), name=key, requires_grad=False
                )

    return weights


def load_tensor_by_offset(
        all_offset: Dict[int, Tuple[Tuple[int, int], ...]],
        param_name: str,
        checkpoint_dir: str,
        src_sharded_tensor_metas: Dict[str, List[ShardedTensor]],
        param_file_mappings: Dict[str, List[Dict[str, Any]]],
        key_mapping: Dict[str, str],
) -> Dict[int, Parameter]:
    """
    Loads specific tensor slices from checkpoint files based on offset information.

    Retrieves the appropriate segments of a tensor from checkpoint files according to
    the provided offset information. Handles storage rank mapping and potential resharding
    to ensure the correct tensor slices are loaded for each rank.

    Args:
        all_offset: Dictionary mapping ranks to their respective tensor slice offsets
        param_name: Name of the parameter/tensor to load
        checkpoint_dir: Directory containing the checkpoint files
        src_sharded_tensor_metas: Metadata for source sharded tensors
        param_file_mappings: Mapping of parameters to their storage files
        key_mapping: Mapping of `original key` in checkpoint to `param key` in network.

    Returns:
        Dictionary mapping ranks to their corresponding loaded Parameter objects
    """

    def _get_storage_info_of_sharded_tensor(
            sharded_tensor: ShardedTensor,
            param_file_mappings: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Retrieves storage information for a specific sharded tensor."""
        param_key = str((sharded_tensor.org_key, sharded_tensor.global_offset))
        return param_file_mappings[param_key]

    def _get_storage_rank_dict_of_param(
            sharded_tensor_metas: Dict[str, List[ShardedTensor]],
            param_file_mappings: Dict[str, List[Dict[str, Any]]],
            param_name: str
    ) -> Dict[int, Tuple[str, Any]]:
        """Creates a dictionary mapping storage ranks to their file and dtype information."""
        storage_rank_dict: Dict[int, Tuple[str, Any]] = {}
        if param_name not in sharded_tensor_metas:
            param_name = key_mapping[param_name]

        for sharded_tensor in sharded_tensor_metas[param_name]:
            storage_info_list = _get_storage_info_of_sharded_tensor(sharded_tensor, param_file_mappings)
            for storage_info in storage_info_list:
                storage_rank = storage_info["storage_rank"]
                storage_rank_dict[storage_rank] = (storage_info["file_name"], sharded_tensor.dtype)
        return storage_rank_dict

    # Get storage rank information for the parameter
    storage_rank_dict = _get_storage_rank_dict_of_param(
        src_sharded_tensor_metas, param_file_mappings, param_name)

    # Map storage ranks to source ranks and adjust offsets
    storage_to_src_rank_mapping: Dict[int, int] = {}
    for search_rank in list(all_offset.keys()):  # Iterate over copy of keys to allow modification
        if search_rank not in storage_rank_dict:
            # Get first source sharded tensor for this parameter
            src_sharded_tensor = next(iter(src_sharded_tensor_metas[param_name]))

            find_storage_rank = False
            # Find matching storage rank using reshard handler
            for storage_rank in storage_rank_dict:
                reshard_handler = ReshardHandler(
                    param_name=param_name,
                    full_shape=src_sharded_tensor.global_shape,
                    from_layout=src_sharded_tensor.layout,
                    to_layout=src_sharded_tensor.layout,  # No actual layout change
                    to_rank_id=storage_rank
                )

                # Get source rank from reshard handler
                src_rank = next(iter(reshard_handler.infer_all_tensor_offset().keys()))

                if src_rank == search_rank:
                    # Update offset mapping and record rank correspondence
                    all_offset[storage_rank] = all_offset.pop(search_rank)
                    storage_to_src_rank_mapping[storage_rank] = src_rank
                    find_storage_rank = True
                    break

            if not find_storage_rank:
                raise RuntimeError("Failed to find matching storage rank for the parameter")

    # Load tensor slices for each rank
    from_tensor_map: Dict[int, Parameter] = {}
    for search_rank, param_slice in all_offset.items():
        # Get file information and load the specific tensor slice
        param_file_name, param_dtype = storage_rank_dict[search_rank]
        param_file_path = os.path.join(checkpoint_dir, param_file_name)

        # Load the specific slice from the safetensor file
        loaded_weights = load_safetensor(param_file_path, param_name, param_slice, param_dtype)

        # Use original source rank if mapping exists
        mapped_rank = storage_to_src_rank_mapping.get(search_rank, search_rank)
        from_tensor_map[mapped_rank] = loaded_weights[param_name]

    return from_tensor_map


def categorize_params(
        dst_sharded_tensor_metas: Dict[str, ShardedTensor],
        src_sharded_tensor_metas: Dict[str, List[ShardedTensor]],
        param_file_mappings: Dict[str, List[Dict[str, Any]]]
) -> Tuple[List[str], Dict[str, Dict[str, List[Any]]], Dict[str, Dict[str, List[Any]]], Dict[str, List[Any]]]:
    """
    Categorizes parameters based on comparison of source and destination sharding strategies.

    Analyzes parameters from destination and source sharded tensor metadata to classify them into:
    - Special parameters: Missing from source metadata
    - No-shard parameters: Matching sharding strategies and offsets
    - Online-shard parameters: Different sharding strategies requiring resharding

    Args:
        dst_sharded_tensor_metas: Metadata for destination sharded tensors
        src_sharded_tensor_metas: Metadata for source sharded tensors
        param_file_mappings: Mapping of parameters to their storage files

    Returns:
        Tuple containing three collections:
        - special_params: List of parameter names missing from source
        - no_shard_params: Dict mapping filenames to params that don't need resharding
        - online_shard_params: Dict of params that need resharding with their details

    Raises:
        ValueError: If a parameter exists in source metadata but has an empty list of ShardedTensor instances
        RuntimeError: If global shapes of source and destination tensors for a parameter do not match
        RuntimeError: If sharding strategies match but no corresponding parameter offset is found
    """
    # Initialize categorization collections
    not_mapping_params: List[str] = []
    need_concat_params: Dict[str, Dict[str, List[Any]]] = {}
    no_shard_params: Dict[str, Dict[str, List[Any]]] = {}
    no_shard_params_list: List[str] = []
    online_shard_params: Dict[str, List[Any]] = {}

    rank_id = get_real_rank()

    # Analyze each parameter in destination metadata
    for param_name in dst_sharded_tensor_metas:
        # Handle parameters missing from source metadata
        if param_name not in src_sharded_tensor_metas:
            not_mapping_params.append(param_name)
            continue

        # Get destination tensor strategy information
        dst_sharded_tensor = dst_sharded_tensor_metas[param_name]
        dst_global_shape, dst_axis_fragmentations, dst_global_offset = get_strategy_info_from_sharded_tensor(
            dst_sharded_tensor)

        src_sharded_tensor_list = src_sharded_tensor_metas[param_name]
        if not src_sharded_tensor_list:
            raise ValueError(
                f"Source metadata for parameter '{param_name}' contains an empty list of ShardedTensor instances. "
                "Valid source metadata requires at least one ShardedTensor entry."
            )

        # Get parameters info which need to concat
        if param_name != src_sharded_tensor_list[0].key:
            concat_infos = []
            reshard_infos = []
            for src_sharded_tensor in src_sharded_tensor_list:
                param_key = str((src_sharded_tensor.org_key, src_sharded_tensor.global_offset))
                concat_infos.append(
                    {
                        'sub_name': src_sharded_tensor.org_key,
                        'file_name': param_file_mappings[param_key][0]["file_name"],
                        'param_dtype': src_sharded_tensor.dtype,
                    }
                )

            if dst_axis_fragmentations != src_sharded_tensor_list[0].axis_fragmentations:
                # `reshard_infos` contains `full_shape, from_layout, to_layout, to_rank_id`
                reshard_infos = [dst_global_shape, None, dst_sharded_tensor.layout, rank_id]
            need_concat_params[param_name] = (concat_infos, reshard_infos)
            continue

        param_key: Optional[str] = None
        strategy_is_same = False

        # Compare with each source tensor strategy
        for src_sharded_tensor in src_sharded_tensor_list:
            src_global_shape, src_axis_fragmentations, src_global_offset = \
                get_strategy_info_from_sharded_tensor(src_sharded_tensor)

            # Validate global shape compatibility
            if src_global_shape != dst_global_shape:
                raise RuntimeError("Global shapes of source and destination tensors do not match")

            # Check if sharding strategies differ
            if src_axis_fragmentations != dst_axis_fragmentations:
                break  # Strategies differ, no need to check further

            strategy_is_same = True

            # Check if offsets match for direct mapping
            if src_global_offset == dst_global_offset:
                param_key = str((src_sharded_tensor.org_key, src_global_offset))
                break  # Found matching parameter

        # Validate strategy consistency
        if strategy_is_same and param_key is None:
            raise RuntimeError("Matching strategy found but no corresponding parameter offset")

        src_sharded_tensor = src_sharded_tensor_list[0]

        # Categorize based on strategy comparison
        if strategy_is_same:
            # Parameters that don't need resharding
            file_name = param_file_mappings[param_key][0]["file_name"]

            # Initialize entry if new file
            if file_name not in no_shard_params:
                no_shard_params[file_name] = {
                    "param_name_list": [src_sharded_tensor.org_key],
                    "param_dtype_list": [src_sharded_tensor.dtype],
                }
            else:
                # Add to existing file entry
                no_shard_params[file_name]["param_name_list"].append(src_sharded_tensor.org_key)
                no_shard_params[file_name]["param_dtype_list"].append(src_sharded_tensor.dtype)

            no_shard_params_list.append(src_sharded_tensor.org_key)
        else:
            # Parameters that need online resharding
            online_shard_params[src_sharded_tensor.org_key] = [
                dst_global_shape, src_sharded_tensor.layout, dst_sharded_tensor.layout, rank_id
            ]
    # Parameters to be processed for categorized logging
    logger.debug(f"Params not mapping: {not_mapping_params}")
    logger.debug(f"Params needing transformation: {need_concat_params}")
    logger.debug(f"Params no need reshard: {no_shard_params_list}")
    logger.debug(f"Params need reshard: {list(online_shard_params.keys())}")

    return not_mapping_params, need_concat_params, no_shard_params, online_shard_params


def get_metadata_of_checkpoint(checkpoint_dir: str) -> tuple[dict, dict]:
    """
    Retrieves metadata from checkpoint directory, either from an existing metadata file
     or by parsing checkpoint files.

    First checks for a pre-existing 'metadata.json' file in the checkpoint directory. If found,
    it loads metadata from this file using load_metadata(). If not found, it generates metadata
    by parsing the checkpoint files directly using load_metadata_from_checkpoint().

    Args:
        checkpoint_dir: Path to the directory containing the checkpoint files.
        network: The target core network (Cell) which has method `convert_name` to convert Hugging Face weight.

    Returns:
        A tuple containing two dictionaries:
        - sharded_tensor_metas: Metadata about sharded tensors
        - param_file_mappings: Mapping of parameters to their storage files
    """
    logger.info("..........Load Metadata of Checkpoint..........")

    # Construct path to metadata file
    metadata_path = os.path.join(checkpoint_dir, "metadata.json")

    # Load from existing metadata file if available
    if os.path.exists(metadata_path):
        sharded_tensor_metas, param_file_mappings = load_metadata(metadata_path)
    # Otherwise generate metadata from checkpoint files
    else:
        sharded_tensor_metas, param_file_mappings = generate_default_metadata_from_checkpoint(checkpoint_dir)

    # if the content or format of metadata_path and checkpoint_dir are invalid, the return value of
    # sharded_tensor_metas and param_file_mappings may be empty or None,
    # and it may cause an error in subsequent loading process.
    return sharded_tensor_metas, param_file_mappings


def params_key_mapping(
        sharded_tensor_metas: Dict[str, List[ShardedTensor]],
        network: Cell
) -> tuple[dict, dict]:
    """
    Mapping Hugging Face checkpoint keys to MindSpore Transformers.

    Args:
        sharded_tensor_metas: Metadata about sharded tensors.
        network: The network (Cell) which has method `convert_name` to convert Hugging Face weight.

    Returns:
        A dictionary after mapping about sharded tensor metas.
    """
    # The key of `mapped_sharded_tensor_metas` is in the network,
    # such as { qkv: [ShardedTensor, ShardedTensor, ShardedTensor], ... }
    mapped_sharded_tensor_metas = {}
    # The key of `key_mapping` is {'weight_key': 'mapping_key'},
    # and the `mapping_key` may not have the same name as the parameter in the network,
    # it could be an intermediate form,
    # such as { 'q_proj': 'linear_q', 'k_proj': 'linear_k', 'v_proj': 'linear_v', ... }
    key_mapping = {}

    for param_name in sharded_tensor_metas:
        param_name_converted = network.convert_name(param_name)
        sharded_tensor_list = sharded_tensor_metas.get(param_name)

        for sharded_tensor in sharded_tensor_list:
            sharded_tensor.key = param_name_converted
            sharded_tensor.org_key = param_name

        key_mapping[param_name] = param_name_converted
        param_name_converted_concat = network.convert_concat_name(param_name_converted)
        mapped_sharded_tensor_metas.setdefault(param_name_converted_concat, []).extend(sharded_tensor_list)

    return mapped_sharded_tensor_metas, key_mapping


# pylint: disable=W0212
def get_core_network(network):
    """Get the core network that has `convert_name` method."""
    if hasattr(network, '_backbone'):
        return get_core_network(network._backbone)
    if hasattr(network, 'network'):
        return get_core_network(network.network)
    return network


def load_checkpoint(
        checkpoint: str,
        network: Cell,
        optimizer: Optional[Optimizer] = None,
        global_step: Optional[int] = None,
        balanced_load: bool = False
) -> None:
    """
    Loads a checkpoint into a network and optional optimizer.

    This function handles checkpoint validation, metadata extraction, parameter categorization,
    and parameter loading with optional resharding. It manages both network parameters and
    optimizer states, providing detailed logging of the loading process.

    Args:
        checkpoint: Path to the checkpoint file or directory containing checkpoint files
        network: The target network (Cell) to load parameters into (cannot be None)
        optimizer: Optional optimizer (Cell) to load optimizer states into
        global_step: Optional initial global step value if not found in checkpoint

    Raises:
        ValueError: If the input `network` is None
        (Other exceptions may be raised by dependent functions for checkpoint validation/loading)
    """
    # Validate mandatory network parameter
    check_the_param_for_load_ckpt(checkpoint, network)

    # Determine checkpoint directory path
    checkpoint_dir = get_checkpoint_path(checkpoint)

    logger.info("..........Start Load Checkpoint..........")

    # Retrieve metadata from checkpoint files
    src_sharded_tensor_metas, param_file_mappings = get_metadata_of_checkpoint(checkpoint_dir)

    # Get the core network and check the convert method is illegal
    network = get_core_network(network)
    # Mapping the weight keys, which is used to determine whether to load the Hugging Face weights.
    src_sharded_tensor_metas, key_mapping = params_key_mapping(src_sharded_tensor_metas, network)

    if not src_sharded_tensor_metas or not param_file_mappings:
        raise RuntimeError(
            f"Failed to load valid metadata from checkpoint directory: `{checkpoint_dir}`. "
            "Metadata must include both sharded tensor information and parameter-file mappings."
        )

    # Define parameter filtering function
    def filter_func(param_name: str) -> bool:
        if optimizer:
            return "accu_grads" not in param_name
        return param_name in list(network.parameters_dict().keys())

    param_redundancy = None
    if balanced_load:
        rank_id_to_sharded_tensors = apply_balance_shard_strategy(network, filter_func)
        dst_sharded_tensor_metas = get_cur_sharded_tensor_after_balanced(rank_id_to_sharded_tensors)
        param_redundancy = get_param_redundancy_after_balanced(rank_id_to_sharded_tensors)
    else:
        dst_sharded_tensor_metas = get_cur_sharded_tensor(network, filter_func) \
            if get_real_group_size() > 1 else get_sharded_tensor_from_cell(network, optimizer)

    # Categorize parameters based on sharding strategies
    _, need_concat_params, no_shard_params, online_shard_params = categorize_params(
        dst_sharded_tensor_metas, src_sharded_tensor_metas, param_file_mappings
    )

    # Process Weight
    state_dict: Dict[str, Parameter] = {}

    # Concat parameters
    concat_params(checkpoint_dir, network, key_mapping, need_concat_params, state_dict)

    # Load parameters that don't require resharding
    for file_name, param_info in no_shard_params.items():
        param_name_list = param_info["param_name_list"]
        param_dtype_list = param_info["param_dtype_list"]
        no_reshard_state_dict = load_safetensor(
            os.path.join(checkpoint_dir, file_name), param_name_list, dtype=param_dtype_list
        )

        state_dict.update({
            key_mapping[param_name]: value
            for param_name, value in no_reshard_state_dict.items()
        })

    # Load and reshard parameters that require online resharding
    for param_name, (full_shape, from_layout, to_layout, to_rank_id) in online_shard_params.items():
        reshard_handler = ReshardHandler(param_name, full_shape, from_layout, to_layout, to_rank_id)
        all_offset = reshard_handler.infer_all_tensor_offset()
        from_tensor_map = load_tensor_by_offset(
            all_offset, param_name, checkpoint_dir, src_sharded_tensor_metas, param_file_mappings, key_mapping
        )
        target_weight = reshard_handler.get_real_tensor(from_tensor_map)
        param_name = key_mapping[param_name]
        state_dict[param_name] = Parameter(target_weight, name=param_name, requires_grad=False)

    # Handle global_step for optimizer if needed
    if optimizer and "global_step" not in state_dict:
        # Initialize global_step with default or from common.json
        if not global_step:
            common_file = os.path.join(checkpoint_dir, 'common.json')
            global_step = 0 if not os.path.exists(common_file) else CommonInfo.load_common(common_file).global_step

        state_dict["global_step"] = Parameter(
            Tensor([global_step], mstype.int32), name="global_step", requires_grad=False
        )

    # Load state dictionary into network and optimizer
    load_parameters(
        network,
        state_dict,
        optimizer,
        balanced_load=balanced_load,
        param_redundancy=param_redundancy
    )


def concat_params(checkpoint_dir: str, network: Cell, key_mapping: dict, need_concat_params, state_dict: dict):
    """Concat the need_concat_params dict in checkpoint."""
    if need_concat_params and not hasattr(network, 'convert_hf_weight'):
        raise NotImplementedError("The `convert_hf_weight` method of network is not implemented.")

    for param_name, concat_info in need_concat_params.items():
        sharded_tensor_list, reshard_info = concat_info
        org_weight_dict = {}
        # Get all the params need to concat into `org_weight_dict`.
        for sharded_tensor in sharded_tensor_list:
            org_weight_dict.update(
                load_safetensor(
                    checkpoint_path=os.path.join(checkpoint_dir, sharded_tensor['file_name']),
                    param_name=sharded_tensor['sub_name'],
                    dtype=sharded_tensor['param_dtype']
                )
            )
        # Mapping the weight key to MCore key into `concat_dict`.
        concat_dict = {
            key_mapping[k]: v
            for k, v in org_weight_dict.items()
        }
        # Concat the weight.
        concated_weight = network.convert_hf_weight(concat_dict)

        if reshard_info:
            # Get the offset of the Tensor to reshard.
            full_shape, from_layout, to_layout, to_rank_id = reshard_info
            reshard_handler = ReshardHandler(param_name, full_shape, from_layout, to_layout, to_rank_id)
            all_offset = reshard_handler.infer_all_tensor_offset()
            # Get the slice to reshard the Tensor.
            slices = tuple(slice(start, end) for start, end in all_offset[0])
            target_weight = concated_weight[param_name][slices]
            # Update to `state_dict` to load into the network.
            state_dict[param_name] = Parameter(target_weight, name=param_name, requires_grad=False)
        else:
            state_dict[param_name] = Parameter(concated_weight[param_name], name=param_name, requires_grad=False)


def check_the_param_for_load_ckpt(checkpoint: str, network: Cell):
    """Check the params passing in `load_checkpoint` method is legal."""
    if network is None:
        raise ValueError("The 'network' cannot be None - a target network is required for loading.")

    if not os.path.exists(checkpoint):
        raise ValueError(f"Checkpoint does not exist: {checkpoint}")


def load_parameters(
        network: Cell,
        state_dict: Dict[str, Parameter],
        optimizer: Optional[Cell] = None,
        state_dict_opt: Optional[Dict[str, Parameter]] = None,
        balanced_load: Optional[bool] = False,
        param_redundancy: Optional[Dict[Tuple, str]] = None
):
    """
    Loads parameters into a MindSpore network and optional optimizer, with support for redundant parameter handling.

    This function separates network-specific and optimizer-specific parameters from input state dictionaries,
    loads them into their respective components, and provides detailed logging of unloaded parameters. When
    `balanced_load` is enabled, it leverages shard balancing and parameter broadcasting to eliminate redundant
    parameter storage across ranks, improving memory efficiency in distributed training scenarios.

    Core workflow:
    1. Initialize optimizer state dictionary if not provided.
    2. (If balanced load enabled) Generate parameter redundancy map via shard balancing if not explicitly provided.
    3. Separate parameters from the main state dict into network-specific and optimizer-specific (state_dict_opt).
    4. Load network parameters, track unloaded parameters, and filter out cache-related entries from unloaded logs.
    5. (If balanced load enabled) Broadcast redundant parameters across ranks to ensure consistency.
    6. Load optimizer parameters (if optimizer and state_dict_opt are provided) and apply balanced load if enabled.
    7. Log detailed information about loaded/unloaded parameters for both network and optimizer.

    Args:
        network (Cell): Target MindSpore Network Cell to load parameters into. Must be a valid Cell instance.
        state_dict (Dict[str, Parameter]): Dictionary containing network parameters to load. Keys must match
            parameter names in the network (or optimizer, for parameters to be redirected).
        optimizer (Optional[Cell]): Optional MindSpore Optimizer Cell to load optimizer-specific parameters into.
            If provided, must be a valid Cell instance.
        state_dict_opt (Optional[Dict[str, Parameter]]): Optional dictionary containing optimizer parameters to load.
            Initialized as an empty dict if not provided.
        balanced_load (Optional[bool]): Whether to enable balanced loading with redundant parameter elimination.
            When True, uses `apply_balance_shard_strategy` to identify redundant parameters and
            `single_parameter_broadcast` to synchronize values across ranks. Defaults to False.
        param_redundancy (Optional[Dict[Tuple[int, ...], List[str]]]): Precomputed mapping of redundant rank groups
            (tuples of rank IDs) to lists of parameter keys. Only used if `balanced_load` is True; if not provided,
            generated dynamically via `apply_balance_shard_strategy`. Defaults to None.

    Raises:
        ValueError: If `network` is not a valid MindSpore Cell, `state_dict` is invalid (e.g., not a dict),
            `state_dict_opt` is provided but not a dict, or `optimizer` is provided but not a valid Cell.
        RuntimeError: If parameter loading fails due to mismatched keys or invalid parameter types (propagated from
            `load_param_into_net`).
    """
    def split_state_dict(network, state_dict, optimizer, state_dict_opt):
        """split state dict"""
        network_param_names = set(network.parameters_dict().keys())
        optimizer_param_names = set(optimizer.parameters_dict().keys()) if optimizer else set()
        for param_name in list(state_dict.keys()):
            if param_name not in network_param_names and param_name in optimizer_param_names and \
                param_name not in state_dict_opt:
                state_dict_opt[param_name] = state_dict.pop(param_name)
        return network_param_names, optimizer_param_names, state_dict, state_dict_opt

    def print_not_load_info(param_list: List, param_info: str):
        if not param_list:
            logger.info(f"All {param_info} are loaded.")
            return

        logger.info(f"{param_info} not loaded:")
        for p in param_list:
            logger.info(f"  - {p}")

    state_dict_opt: Dict[str, Parameter] = {} if not state_dict_opt else state_dict_opt

    # Separate network and optimizer parameters
    if balanced_load and param_redundancy is None:
        rank_id_to_sharded_tensors = apply_balance_shard_strategy(network)
        param_redundancy = get_param_redundancy_after_balanced(rank_id_to_sharded_tensors)

    network_param_names, _, state_dict, state_dict_opt = \
        split_state_dict(network, state_dict, optimizer, state_dict_opt)

    # Load parameters into network
    logger.debug(f"Network state_dict keys: {list(state_dict.keys())}")
    param_not_load, ckpt_not_load = load_param_into_net(network, state_dict, strict_load=True)
    if balanced_load:
        param_loaded = {param_name for param_name in state_dict if param_name not in ckpt_not_load}
        single_parameter_broadcast(network, param_redundancy, param_not_load, param_loaded)
    # Filter out cache and optimizer parameters from unloaded list
    param_not_load = [p for p in param_not_load if "key_cache" not in p and "value_cache" not in p]
    print_not_load_info(param_not_load, "Network parameters")
    print_not_load_info(ckpt_not_load, "Checkpoint weights")

    # Load parameters into optimizer if available
    if optimizer and state_dict_opt:
        logger.debug(f"Optimizer state_dict keys: {list(state_dict_opt.keys())}")
        param_not_load_opt, ckpt_not_load_opt = load_param_into_net(optimizer, state_dict_opt, strict_load=True)
        if balanced_load:
            param_loaded_opt = {param_name for param_name in state_dict_opt if param_name not in ckpt_not_load_opt}
            single_parameter_broadcast(optimizer, param_redundancy, param_not_load_opt, param_loaded_opt)

        param_not_load_opt = [p for p in param_not_load_opt if p not in network_param_names]
        print_not_load_info(param_not_load_opt, "Optimizer parameters")
        print_not_load_info(ckpt_not_load_opt, "Optimizer weights")


def get_checkpoint_path(checkpoint: str) -> str:
    """
    Retrieve a valid checkpoint directory.

    This function locates the latest checkpoint iteration from a training checkpoint
    directory, validates its existence and suitability, and returns the path to be used.

    Args:
        checkpoint: Base directory containing training checkpoints

    Returns:
        str: Path to the valid checkpoint directory

    Raises:
        ValueError: If the base checkpoint directory doesn't exist or the found checkpoint
                   is invalid.
    """
    if not checkpoint:
        return ""

    if not os.path.exists(checkpoint):
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint}")

    if not os.path.isdir(checkpoint):
        raise ValueError(f"Checkpoint path is not a directory: {checkpoint}")

    # Check all need checkpoint files if load Hugging Face checkpoint
    hf_index_json = os.path.join(checkpoint, "model.safetensors.index.json")
    if os.path.exists(hf_index_json):
        with open(hf_index_json, 'r', encoding='utf-8') as f:
            index_json = json.load(f)
        if isinstance(index_json, dict):
            weight_map = index_json['weight_map'] if 'weight_map' in index_json else index_json
        else:
            raise ValueError(f"Format of '{hf_index_json}' is illegal!")

        sf_file_list = set(weight_map.values())
        not_exist_file = [
            f
            for f in sf_file_list
            if not os.path.isfile(os.path.join(checkpoint, f))
        ]
        not_exist_file.sort()
        if not_exist_file:
            raise ValueError(f"The files '{not_exist_file}' do not exist in `{checkpoint}`.")
        return checkpoint

    tracker_filename = get_checkpoint_tracker_filename(checkpoint)
    if os.path.exists(tracker_filename):
        iteration = get_latest_iteration_from_tracker(checkpoint)
        checkpoint = get_checkpoint_iter_dir(checkpoint, iteration)

    verify_ckpt_valid(checkpoint)
    logger.info(f"Get checkpoint: {checkpoint}")

    return checkpoint
