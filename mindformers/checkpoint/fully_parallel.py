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
"""save / load parallelization strategy."""
import os
from collections import defaultdict
from typing import Callable

from mindspore import save_checkpoint
from mindspore.nn import Cell

from mindformers.checkpoint.sharded_tensor import get_all_sharded_tensor
from mindformers.tools.logger import logger
from mindformers.tools.utils import get_real_rank
from mindformers.checkpoint.metadata import save_metadata, load_metadata
from mindformers.checkpoint.utils import (
    _reverse_sharded_tensor_shard_id,
    get_checkpoint_iter_dir,
    get_metadata_filename,
    get_checkpoint_name,
    FileType,
    _get_shard_size,
    sharded_tensor_shard_id
)


class BalancedSaveStrategy():
    """
    A class that implements a balanced saving strategy for model checkpoints in a distributed training environment.

    This strategy aims to evenly distribute the saving of model parameters across multiple ranks to optimize
    the checkpointing process. It takes into account the shared distribution of parameters among ranks and
    ensures that each rank saves only the parameters it is responsible for. Additionally, it provides options
    for caching the distribution information and saving metadata about the checkpoint files.

    Attributes:
        network: The neural network model to be saved.
        user_prefix (str): A user-defined prefix that can be used to customize the naming of checkpoint files.
            Defaults to an empty string.
        do_cache_distribution (bool): A flag indicating whether to cache the shared distribution information.
            Caching can improve performance if the distribution remains the same across multiple checkpoint saves.
            Defaults to False.
        cached_distribution (dict or None): The cached shared distribution information, if caching is enabled.
            Initially set to None.
        checkpoint_path (str or None): The directory path where the checkpoint files will be saved.
            Defaults to None.
        file_type (str): Specific file types corresponding to shard weights.
    """

    def __init__(self, network, user_prefix=None, do_cache_distribution=False, checkpoint_path=None,
                 filter_func=None, file_type=FileType.MODEL):
        """
        Initialize the BalancedSaveStrategy object.
        """
        super().__init__()
        self.user_prefix = user_prefix
        self.do_cache_distribution = do_cache_distribution
        self.total_files_num = None
        self.cur_rank_file_id = None
        self.cached_distribution = None
        self.rank_id = get_real_rank()
        self.checkpoint_path = checkpoint_path
        self.network = network
        self.ckpt_format = "safetensors"
        self.filter_func = filter_func
        self.file_type = file_type

    def get_total_files(self):
        """
        Get the total number of checkpoint files required for all ranks.

        If the total number of files has not been calculated yet, this method will calculate it based on the
        shared distribution of parameters among ranks.

        Returns:
            The total number of checkpoint files.
        """
        if self.total_files_num is None:
            shared_distribution = self.apply_saving_parallelization()
            rank_params_mappings = self._get_rank_params_mappings(shared_distribution)
            self.total_files_num = self._get_total_files_num(rank_params_mappings)

        return self.total_files_num

    def get_cur_rank_file_id(self):
        """
        Get the identifier for the current rank's checkpoint file.

        If the identifier has not been calculated yet, this method will calculate it based on the shared
        distribution of parameters among ranks.

        Returns:
            The identifier for the current rank's checkpoint file.
        """
        if self.cur_rank_file_id is None:
            shared_distribution = self.apply_saving_parallelization()
            rank_params_mappings = self._get_rank_params_mappings(shared_distribution)
            self.cur_rank_file_id = self._get_cur_rank_file_id(rank_params_mappings)

        return self.cur_rank_file_id

    def save(self, iteration):
        """
        Save the model checkpoint using the balanced saving strategy.

        This method determines which parameters should be saved by the current rank based on the shared distribution,
            generates the appropriate checkpoint file name,
            and saves the selected parameters in the specified format.
        It also saves metadata about the checkpoint files if the current rank is 0.
        """
        shared_distribution = self.apply_saving_parallelization()
        rank_params_mappings = self._get_rank_params_mappings(shared_distribution)

        if self.total_files_num is None:
            self.total_files_num = self._get_total_files_num(rank_params_mappings)
        if self.cur_rank_file_id is None:
            self.cur_rank_file_id = self._get_cur_rank_file_id(rank_params_mappings)

        save_ckpt_path = get_checkpoint_iter_dir(self.checkpoint_path, iteration)
        save_file_name = os.path.join(
            save_ckpt_path,
            get_checkpoint_name(None, self.user_prefix, self.cur_rank_file_id, self.total_files_num, self.file_type)
        )

        def choice_func(param_name):
            if param_name in rank_params_mappings[self.rank_id]:
                return True
            return False

        save_checkpoint(
            self.network,
            save_file_name,
            format=self.ckpt_format,
            choice_func=choice_func,
            integrated_save=False
        )
        logger.info(
            f"Non-redundancy {self.file_type.value} checkpoint successfully saved at '{save_file_name}.safetensors'."
        )
        self._save_metadata(shared_distribution, iteration)

    def apply_saving_parallelization(self):
        """
        Get the shared distribution of parameters among ranks.

        If caching is enabled and the distribution has been cached,
            this method will return the cached distribution.
        Otherwise, it will retrieve the current distribution and cache it if caching is enabled.

        Returns:
            shared_distribution (Dict[int, Dict[str, Tuple]]): A nested dictionary where:
            - Outer keys: Target rank IDs (int) in the parallel group.
            - Outer values: Dictionaries mapping unique shard IDs (str) to tuples containing:
            1. Corresponding `ShardedTensor` object with complete shard metadata (shape, dtype, global offset, etc.).
            2. Rank group (tuple of ints): Ranks that have redundant copies of the shard (supports fault tolerance or
                parallel access).
        """
        if self.do_cache_distribution and self.cached_distribution is not None:
            shared_distribution = self.cached_distribution
        else:
            rank_id_to_sharded_tensors = apply_balance_shard_strategy(self.network, self.filter_func)
            shared_distribution = rank_id_to_sharded_tensors

        if self.do_cache_distribution:
            self.cached_distribution = shared_distribution

        return shared_distribution

    def _get_cur_rank_file_id(self, rank_params_mappings):
        """
        Calculate the identifier for the current rank's checkpoint file based on the rank-parameter mappings.

        Args:
            rank_params_mappings (dict): A dictionary where keys are rank IDs and values are lists of parameter names
                associated with that rank.

        Returns:
            The identifier for the current rank's checkpoint file.
        """
        total_files_num = 0

        for rank_id, params in rank_params_mappings.items():
            if rank_id == self.rank_id:
                return total_files_num
            if params:
                total_files_num += 1

        return None

    def _get_total_files_num(self, rank_params_mappings):
        """
        Calculate the total number of files required for saving checkpoints based on the rank-parameter mappings.

        Args:
            rank_params_mappings (dict): A dictionary where keys are rank IDs and values are lists of parameter names
                associated with that rank.

        Returns:
            The total number of files needed for checkpoint saving.
        """
        total_files_num = 0

        for _, params in rank_params_mappings.items():
            if params:
                total_files_num += 1

        return total_files_num

    def _save_metadata(self, shared_distribution, iteration):
        """
        Save metadata about the checkpoint files if the current rank is 0.

        This method creates a mapping between parameter IDs, rank IDs, and checkpoint file names,
            and saves this mapping along with the shard metadata to the specified checkpoint path.

        Args:
            shared_distribution (Dict[int, Dict[str, Tuple]]): A nested dictionary where:
            - Outer keys: Target rank IDs (int) in the parallel group.
            - Outer values: Dictionaries mapping unique shard IDs (str) to tuples containing:
            1. Corresponding `ShardedTensor` object with complete shard metadata (shape, dtype, global offset, etc.).
            2. Rank group (tuple of ints): Ranks that have redundant copies of the shard (supports fault tolerance or
                parallel access).
            iteration (int): The current iteration number.
        """
        if self.rank_id == 0:
            param_file_mapping = []
            cur_rank_id = 0
            rank_param_ids_mappings = self._get_rank_param_ids_mappings(shared_distribution)

            for rank_id, params in rank_param_ids_mappings.items():
                if params:
                    save_file_name = get_checkpoint_name(
                        None, self.user_prefix, cur_rank_id, self.total_files_num, self.file_type
                    )
                    for param_id in params:
                        param_file_mapping.append(
                            (save_file_name + ".safetensors", rank_id, _reverse_sharded_tensor_shard_id(param_id)))
                    cur_rank_id += 1

            sharded_tensor_metas = get_all_sharded_tensor(self.network, self.filter_func)
            origin_metadata_file = get_metadata_filename(self.checkpoint_path, iteration)

            if os.path.exists(origin_metadata_file):
                origin_shard_metadata, origin_param_file_mapping = load_metadata(
                    get_metadata_filename(self.checkpoint_path, iteration))
                sharded_tensor_metas.update({"origin": origin_shard_metadata})
                for param_id, storage in origin_param_file_mapping.items():
                    for storage_item in storage:
                        param_file_mapping.append((
                            storage_item["file_name"],
                            storage_item["storage_rank"],
                            _reverse_sharded_tensor_shard_id(param_id)
                        ))

            metadata_file_path = get_metadata_filename(self.checkpoint_path, iteration)
            save_metadata(sharded_tensor_metas, param_file_mapping, metadata_file_path)
            if self.rank_id == 0:
                logger.info(
                    f"The 'metadata.json' of non-redundancy weight saved successfully at '{metadata_file_path}'."
                )

    def _get_rank_params_mappings(self, shared_distribution):
        """
        Create a mapping from rank IDs to lists of parameter names based on the shared distribution.

        Args:
            shared_distribution (Dict[int, Dict[str, Tuple]]): A nested dictionary where:
            - Outer keys: Target rank IDs (int) in the parallel group.
            - Outer values: Dictionaries mapping unique shard IDs (str) to tuples containing:
            1. Corresponding `ShardedTensor` object with complete shard metadata (shape, dtype, global offset, etc.).
            2. Rank group (tuple of ints): Ranks that have redundant copies of the shard (supports fault tolerance or
                parallel access).

        Returns:
            Dict[int, Optional[List[str]]]: A sorted dictionary where:
            - Outer keys: Rank IDs (int) sorted in ascending numerical order.
            - Outer values: List of param name (str) assigned to the rank.
        """
        rank_params_mappings = {}
        for rank_id, sharded_tensors in shared_distribution.items():
            rank_params_mappings[rank_id] = []
            for _, shard_id_info in sharded_tensors.items():
                sharded_tensor, _ = shard_id_info
                param_name = sharded_tensor.key
                rank_params_mappings[rank_id].append(param_name)

        sorted_rank_params_mappings = {
            k: rank_params_mappings.get(k, None)
            for k in sorted(rank_params_mappings)
        }
        return sorted_rank_params_mappings

    def _get_rank_param_ids_mappings(self, shared_distribution):
        """
        Create a mapping from rank IDs to lists of parameter IDs based on the shared distribution.

        Args:
            shared_distribution (Dict[int, Dict[str, Tuple]]): A nested dictionary where:
            - Outer keys: Target rank IDs (int) in the parallel group.
            - Outer values: Dictionaries mapping unique shard IDs (str) to tuples containing:
            1. Corresponding `ShardedTensor` object with complete shard metadata (shape, dtype, global offset, etc.).
            2. Rank group (tuple of ints): Ranks that have redundant copies of the shard (supports fault tolerance or
                parallel access).

        Returns:
            Dict[int, Optional[List[str]]]: A sorted dictionary where:
            - Outer keys: Rank IDs (int) sorted in ascending numerical order.
            - Outer values: List of parameter IDs (str) assigned to the rank.
        """
        rank_params_mappings = {rank_id: list(sharded_tensors.keys()) \
                                for rank_id, sharded_tensors in shared_distribution.items()}

        sorted_rank_params_mappings = {
            k: rank_params_mappings.get(k, None)
            for k in sorted(rank_params_mappings)
        }

        return sorted_rank_params_mappings


def distribute_shards(shard_coverage, shard_sizes, total_ranks):
    """
    Distribute shards to ranks using a greedy algorithm based on the following priority:
    1. Shards with fewer covering ranks are assigned first.
    2. For shards with the same number of covering ranks, larger shards are assigned first.
    3. For shards with the same size, the shard ID is used as a tiebreaker.

    Args:
        shard_coverage (dict): A dictionary mapping shard IDs to a list of ranks that contain the shard.
        shard_sizes (dict): A dictionary mapping shard IDs to their size in bytes.
        total_ranks (int): The total number of ranks.

    Returns:
        Dict[str, Tuple[int, Tuple[int, ...]]]: A dictionary where each key is a unique shard ID,
        and the corresponding value is a 2-element tuple:
        1. Selected target rank (int): The rank assigned to store the shard (chosen to minimize current load).
        2. Rank group (Tuple[int, ...]): Ranks that originally contain the shard (from `shard_coverage`),
            representing redundant copies for fault tolerance or parallel access.
    """
    coverage_map = {
        k: tuple(sorted(v))
        for k, v in shard_coverage.items()
    }
    rank_loads = {
        rank: 0
        for rank in range(total_ranks)
    }
    shard_assignment = {}
    sorted_shards = sorted(
        coverage_map.items(),
        key=lambda item: (len(item[1]), -shard_sizes[item[0]], item[0])
    )

    for shard_id, available_ranks in sorted_shards:
        selected_rank = min(available_ranks, key=lambda rank: rank_loads[rank])
        shard_assignment[shard_id] = (selected_rank, available_ranks)
        rank_loads[selected_rank] += shard_sizes[shard_id]

    return shard_assignment


def apply_balance_shard_strategy(network: Cell, filter_func: Callable[[str], bool] = None):
    """
    Distributes and balances sharded tensor storage across ranks in a parallel group,
    generating rank-specific shard assignments.

    Collects sharded tensor metadata from the input MindSpore Network Cell (filtered by an optional function),
    computes unique shard identifiers and their sizes, and distributes shards to ranks using a load-balanced strategy.
    The result maps each target rank to its assigned shards along with the group of ranks that share redundant copies
    of those shards.

    Core Workflow:
    1. Extract all sharded tensor metadata from the network using `get_all_sharded_tensor`, applying the `filter_func`
       to select target tensors (e.g., exclude non-trainable parameters).
    2. Generate unique shard IDs for each tensor shard (via `sharded_tensor_shard_id`) by combining the tensor key
       and global offset, then track which ranks originally own each shard.
    3. Calculate the byte size of each unique shard using its local shape and data type (via `_get_shard_size`),
       avoiding redundant size computations for identical shards.
    4. Distribute shards to ranks for storage using the `distribute_shards` function, which implements a load-balanced
       algorithm to evenly distribute the storage load across the parallel group.
    5. Compile a rank-to-shard mapping: for each rank, store its assigned shards and the corresponding rank group
       (ranks with redundant copies of the same shard).

    Args:
        network (Cell): A MindSpore Network Cell containing parameters and their associated sharding metadata.
        filter_func (Optional[Callable[[str], bool]]): An optional filtering function that takes a tensor key (str)
            and returns a boolean. Only tensors for which the function returns `True` are included in the shard
            distribution. Defaults to `None` (all sharded tensors in the network are included).

    Returns:
        Dict[int, Dict[str, Tuple]]: A nested dictionary where:
        - Outer keys: Target rank IDs (int) in the parallel group.
        - Outer values: Dictionaries mapping unique shard IDs (str) to tuples containing:
        1. Corresponding `ShardedTensor` object with complete shard metadata (local shape, dtype, global offset, etc.).
        2. Rank group (tuple of ints): Ranks that have redundant copies of the shard (supports fault tolerance or
            parallel access).
    """
    total_shard_metadata = get_all_sharded_tensor(network, filter_func)
    shard_id_to_ranks = defaultdict(list)
    shard_to_size = {}
    shards_in_this_parallelization_group = set()
    shard_id_to_tensor = {}

    for rank, sharded_tensor_metas in total_shard_metadata.items():
        for sharded_tensor in sharded_tensor_metas.values():
            shard_id = sharded_tensor_shard_id(sharded_tensor.key, sharded_tensor.global_offset)
            shard_id_to_ranks[shard_id].append(rank)

            if shard_id not in shard_to_size:
                shard_to_size[shard_id] = _get_shard_size(sharded_tensor.local_shape, sharded_tensor.dtype)
                shard_id_to_tensor[shard_id] = sharded_tensor
            shards_in_this_parallelization_group.add(shard_id)

    shard_id_to_ranks = {
        k: v
        for k, v in shard_id_to_ranks.items()
        if k in shards_in_this_parallelization_group
    }

    shard_to_saving_rank = distribute_shards(
        shard_id_to_ranks, shard_to_size, len(total_shard_metadata)
    )

    rank_id_to_sharded_tensors = {}
    for shard_id, rank_info in shard_to_saving_rank.items():
        selected_rank_id, rank_group = rank_info
        sharded_tensor = shard_id_to_tensor[shard_id]
        if selected_rank_id in rank_id_to_sharded_tensors:
            rank_id_to_sharded_tensors[selected_rank_id][shard_id] = (sharded_tensor, rank_group)
        else:
            rank_id_to_sharded_tensors[selected_rank_id] = {shard_id: (sharded_tensor, rank_group)}

    return rank_id_to_sharded_tensors
