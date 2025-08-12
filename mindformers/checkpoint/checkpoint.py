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
from typing import Callable, Union, List, Dict
from dataclasses import dataclass

import threading
from multiprocessing import active_children

from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.nn import Cell
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.communication.management import get_rank, get_group_size
import mindspore.communication.comm_func as comm_func
from mindspore import save_checkpoint as ms_save_checkpoint

from mindformers.tools.logger import logger
from mindformers.tools.utils import (
    barrier_world,
    get_output_subpath,
    set_safe_mode_for_file_or_dir
)
from mindformers.checkpoint.utils import (
    get_checkpoint_iter_dir,
    get_checkpoint_name,
    get_checkpoint_tracker_filename,
    get_common_filename,
    check_checkpoints_dir_max_num,
    get_metadata_filename
)
from mindformers.checkpoint.sharded_tensor import get_sharded_tensor_list_from_strategy_metadata
from mindformers.checkpoint.metadata import save_metadata


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
            if thread.getName() == "asyn_save_ckpt":
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
                    global_strategy_info: List[Dict] = None):
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
        global_strategy_info (List[Dict]): The strategy info of this network.
    """
    logger.info('....... Start to save checkpoint as new format .......')

    # Get the root path of all checkpoints to save.
    if save_checkpoint_path:
        checkpoints_root_path = os.path.realpath(save_checkpoint_path)
    else:
        checkpoints_root_path = get_output_subpath("checkpoint", append_rank=False)
    logger.info(f"The root path of saved checkpoints is: '{checkpoints_root_path}'.")

    # Generate current iteration saving path.
    cur_iter_checkpoint_dir = get_checkpoint_iter_dir(checkpoints_root_path, iteration)
    logger.info(f"At current iteration '{iteration}', the weight will be saved in: '{cur_iter_checkpoint_dir}'.")

    # Whether to use async save.
    use_async_save = async_save_manager is not None

    if get_rank() == 0:
        os.makedirs(cur_iter_checkpoint_dir, exist_ok=True)
        set_safe_mode_for_file_or_dir(cur_iter_checkpoint_dir)
    barrier_world(f"Rank_0 to ensure path '{cur_iter_checkpoint_dir}' is exists.")

    # Prepare async save manager before save.
    def iter_finalize_func():
        """Save checkpoint finalize function."""
        tracker_filename = get_checkpoint_tracker_filename(checkpoints_root_path)
        logger.info(f"save checkpoint tracker file to {tracker_filename}")
        with open(tracker_filename, "w") as f:
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
    start_save_ckpt_time = time()
    model_ckpt_filename = get_checkpoint_name(
        cur_iter_checkpoint_dir, user_prefix, get_rank(), get_group_size(), 'Model'
    )
    ms_save_checkpoint(
        network,
        model_ckpt_filename,
        async_save=use_async_save,
        format="safetensors"
    )
    logger.info(f"Model checkpoint successfully saved at '{model_ckpt_filename}.safetensors'.")

    model_keys = network.parameters_dict().keys()

    # Save optimizer weight.
    if optimizer is not None:
        logger.warning("....... Start to save optimizer weight .......")
        optimizer_ckpt_filename = get_checkpoint_name(
            cur_iter_checkpoint_dir, user_prefix, get_rank(), get_group_size(), 'Optimizer'
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
    if global_strategy_info is not None:
        metadata_file_path = get_metadata_filename(checkpoints_root_path, iteration)
        save_metadata_json(
            global_strategy_info=global_strategy_info,
            model_keys=model_keys if optimizer is None else None,
            user_prefix=user_prefix,
            metadata_file_path=metadata_file_path,
            save_optimizer=optimizer is not None
        )
    else:
        logger.info("No need to save metadata.json for single card.")

    # Save tracker file in sync save process.
    if not use_async_save:
        barrier_world("All ranks for sync save checkpoint.")
        logger.info("Rank_0 execute finalize func.")
        if get_rank() == 0:
            iter_finalize_func()
        logger.info(f"Save checkpoint cost time: {time() - start_save_ckpt_time:.3f}s.")


def save_metadata_json(global_strategy_info, model_keys, user_prefix, metadata_file_path, save_optimizer):
    """Saving metadata.json used `get_strategy_metadata` API."""
    logger.info("...... Start saving metadata ......")

    if get_rank() == 0:
        npu_nums = get_group_size()
        sharded_tensor_metas = list()
        param_file_mappings = list()

        for cur_npu_rank in range(0, npu_nums):
            org_cur_rank_strategy_layout = global_strategy_info[cur_npu_rank]
            cur_rank_strategy_layout = [
                dict([item])
                for item in org_cur_rank_strategy_layout.items()
            ]

            # Get Sharded tensors from strategy metadata of current rank.
            cur_rank_sharded_tensors = get_sharded_tensor_list_from_strategy_metadata(
                param_infos=cur_rank_strategy_layout,
                model_keys=model_keys,
                cur_npu_rank=cur_npu_rank,
                save_optimizer=save_optimizer
            )

            # Get mappings of parameter file of current rank.
            for sharded_tensor in cur_rank_sharded_tensors:
                if model_keys and sharded_tensor.key not in list(model_keys):
                    ckpt_name = get_checkpoint_name(None, user_prefix, cur_npu_rank, npu_nums, 'Optimizer')
                else:
                    ckpt_name = get_checkpoint_name(None, user_prefix, cur_npu_rank, npu_nums, 'Model')
                param_file_mappings.append(
                    (ckpt_name + '.safetensors', cur_npu_rank, (sharded_tensor.key, sharded_tensor.global_offset))
                )

            sharded_tensor_metas.append(cur_rank_sharded_tensors)

        save_metadata(sharded_tensor_metas, param_file_mappings, metadata_file_path)

    # Barrier here to ensure 'metadata.json' saved, then continue training.
    barrier_world("Rank_0 is saving 'metadata.json' ...")
    logger.info(f"The 'metadata.json' saved successfully at '{metadata_file_path}'.")
