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

from mindspore import save_checkpoint
from mindspore.communication import get_rank
from mindspore.nn import Cell

from mindformers.checkpoint.sharded_tensor import get_all_sharded_tensor
from mindformers.tools.logger import logger
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
from mindformers.tools.utils import get_real_local_rank


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
        self.rank_id = get_rank()
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
            shared_distribution, id_to_tensor = self.apply_saving_parallelization()
            rank_params_mappings = self._get_rank_params_mappings(shared_distribution, id_to_tensor)
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
            shared_distribution, id_to_tensor = self.apply_saving_parallelization()
            rank_params_mappings = self._get_rank_params_mappings(shared_distribution, id_to_tensor)
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
        shared_distribution, id_to_tensor = self.apply_saving_parallelization()
        rank_params_mappings = self._get_rank_params_mappings(shared_distribution, id_to_tensor)

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
            A tuple containing the shared distribution dictionary and the shard-to-name mapping dictionary.
        """
        if self.do_cache_distribution and self.cached_distribution is not None:
            shared_distribution = self.cached_distribution
        else:
            shard_id_to_ranks, shard_id_to_tensor, _ = apply_balance_shard_strategy(self.network, self.filter_func)
            shared_distribution = (shard_id_to_ranks, shard_id_to_tensor)

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
            shared_distribution (dict): A dictionary where keys are parameter IDs and values are rank IDs indicating
                which rank is responsible for a particular parameter.
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

            shard_to_metadata = get_all_sharded_tensor(self.network, self.filter_func)
            origin_metadata_file = get_metadata_filename(self.checkpoint_path, iteration)

            if os.path.exists(origin_metadata_file):
                origin_shard_metadata, origin_param_file_mapping = load_metadata(
                    get_metadata_filename(self.checkpoint_path, iteration))
                shard_to_metadata.extend(list(origin_shard_metadata.values()))
                for param_id, storage in origin_param_file_mapping.items():
                    for storage_item in storage:
                        param_file_mapping.append((
                            storage_item["file_name"],
                            storage_item["storage_rank"],
                            _reverse_sharded_tensor_shard_id(param_id)
                        ))

            metadata_file_path = get_metadata_filename(self.checkpoint_path, iteration)
            save_metadata(shard_to_metadata, param_file_mapping, metadata_file_path)
            if self.rank_id == 0:
                logger.info(
                    f"The 'metadata.json' of non-redundancy weight saved successfully at '{metadata_file_path}'."
                )

    def _get_rank_params_mappings(self, shared_distribution, id_to_tensor):
        """
        Create a mapping from rank IDs to lists of parameter names based on the shared distribution and
        shard-to-name mapping.

        Args:
            shared_distribution (dict): A dictionary where keys are parameter IDs and values are rank IDs indicating
                which rank is responsible for a particular parameter.
            id_to_tensor (dict): A dictionary that maps parameter IDs to their corresponding ShardTensor.

        Returns:
            A dictionary where keys are rank IDs and values are lists of parameter names assigned to that rank.
        """
        rank_params_mappings = {}
        for param_id, rank_id in shared_distribution.items():
            if rank_id not in rank_params_mappings:
                rank_params_mappings[rank_id] = [id_to_tensor[param_id].key]
            else:
                rank_params_mappings[rank_id].append(id_to_tensor[param_id].key)
        sorted_rank_params_mappings = {
            k: rank_params_mappings.get(k, None)
            for k in sorted(rank_params_mappings)
        }
        return sorted_rank_params_mappings

    def _get_rank_param_ids_mappings(self, shared_distribution):
        """
        Create a mapping from rank IDs to lists of parameter IDs based on the shared distribution.

        Args:
            shared_distribution (dict): A dictionary where keys are parameter IDs and values are rank IDs indicating
                which rank is responsible for a particular parameter.

        Returns:
            A dictionary where keys are rank IDs and values are lists of parameter IDs assigned to that rank.
        """
        rank_params_mappings = {}
        for param_id, rank_id in shared_distribution.items():
            if rank_id not in rank_params_mappings:
                rank_params_mappings[rank_id] = [param_id]
            else:
                rank_params_mappings[rank_id].append(param_id)

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
        A dictionary mapping shard IDs to the rank that will save the shard.
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
        shard_assignment[shard_id] = selected_rank
        rank_loads[selected_rank] += shard_sizes[shard_id]

    return shard_assignment


def apply_balance_shard_strategy(network: Cell, filter_func):
    """
    Process and balance sharded tensor metadata across all ranks.

    This function retrieves strategy metadata from the network (and optimizer if provided),
    processes sharding information, and distributes shards across ranks to generate balanced
    sharded tensor metadata. If no strategy metadata exists, it falls back to directly extracting
    sharded tensors from the network and optimizer.

    Args:
        network (Cell): The MindSpore network cell containing parameters and sharding strategies.
        optimizer (Optional[Optimizer]): Optional optimizer instance (if provided, filters out
            accumulator gradient parameters from sharding metadata).

    Returns:
        list: Balanced sharded tensor metadata for the current rank, either derived from
            strategy metadata distribution or directly extracted from the network/optimizer.

    Notes:
        - Relies on MindSpore's `get_strategy_metadata` for strategy-based sharding info.
        - Filters out "accu_grads" parameters when an optimizer is provided to avoid redundant sharding.
        - Falls back to direct tensor extraction if no strategy metadata is available.
    """
    total_shard_metadata = get_all_sharded_tensor(network, filter_func)
    shard_id_to_ranks = defaultdict(list)
    shard_to_size = {}
    shards_in_this_parallelization_group = set()
    shard_id_to_tensor = {}

    for rank, sharded_tensor_metas in enumerate(total_shard_metadata):
        for tensor_meta in sharded_tensor_metas:
            shard_id = sharded_tensor_shard_id(tensor_meta.key, tensor_meta.global_offset)
            shard_id_to_ranks[shard_id].append(rank)

            if shard_id not in shard_to_size:
                shard_to_size[shard_id] = _get_shard_size(tensor_meta.local_shape, tensor_meta.dtype)
                shard_id_to_tensor[shard_id] = tensor_meta
            shards_in_this_parallelization_group.add(shard_id)

    shard_id_to_ranks = {
        k: v
        for k, v in shard_id_to_ranks.items()
        if k in shards_in_this_parallelization_group
    }

    shard_to_saving_rank = distribute_shards(
        shard_id_to_ranks, shard_to_size, len(total_shard_metadata)
    )

    dst_sharded_tensor_metas = {}  # {shard_name: ShardTensor}
    local_rank = get_real_local_rank()
    for shard_id, rank_id in shard_to_saving_rank.items():
        if rank_id == local_rank:
            dst_sharded_tensor_metas[_reverse_sharded_tensor_shard_id(shard_id)[0]] = shard_id_to_tensor[shard_id]
    return shard_to_saving_rank, shard_id_to_tensor, dst_sharded_tensor_metas
