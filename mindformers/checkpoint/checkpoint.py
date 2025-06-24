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
from multiprocessing import active_children
import threading
import os
import json
import tempfile
from typing import Callable, Union
from time import time

from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.nn import Cell
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.communication.management import get_rank, get_group_size
import mindspore.communication.comm_func as comm_func
from mindspore import save_checkpoint as ms_save_checkpoint
from mindformers.tools.logger import logger
from mindformers.tools.utils import get_output_subpath
from mindformers.checkpoint import utils


class CommonInfo:
    """Save/load common info for checkpoint."""
    def __init__(self):
        self.common_info = {
            "epoch_num": None,
            "step_num": None,
            "global_step": None,
            "loss_scale": None,
            "global_batch_size": None,
        }

    def __getitem__(self, key):
        if key not in self.common_info:
            raise KeyError(f"{key} is not in common_info")
        return self.common_info[key]

    def __setitem__(self, key, value):
        self.common_info[key] = value

    def save_common(self, common_filename: str):
        """ Save common info to common.json."""
        logger.info(f"save common info to {common_filename}")
        with tempfile.NamedTemporaryFile(mode='w', dir=os.path.dirname(common_filename), delete=False) as tmp_file:
            json.dump(self.common_info, tmp_file)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
            temp_filename = tmp_file.name
        os.replace(temp_filename, common_filename)

    def load_common(self, common_filename: str):
        """Load common info from common.json."""
        logger.info(f"load common info from {common_filename}")
        with open(common_filename, "r") as fp:
            loaded = json.load(fp)
            self.common_info.update(loaded)

        logger.info(f"epoch_num: {self.common_info['epoch_num']}")
        logger.info(f"step_num: {self.common_info['step_num']}")
        logger.info(f"global_step: {self.common_info['global_step']}")
        logger.info(f"loss_scale: {self.common_info['loss_scale']}")
        logger.info(f"global_batch_size: {self.common_info['global_batch_size']}")

        return self.common_info


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
        Adds a new finalize function to the manager. Finalize functions will be execeted once after current
        save action.
        Finalize functions are reset when prepare_before_save is called.

        Args:
            finalize_fn (Callable): function to add to the manager. This function
                will be called once after current save action.

        Returns:
            None
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
        Execete finilize functions if all ranks finish async save.
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
            bool: True if current async save action is still running, False if it is finished.
        """
        if self.async_save is False:
            return False
        # async process
        if self.async_save == "process":
            for process in active_children():
                if process.name == "asyn_save_ckpt":
                    if wait_finish:
                        process.join()
                        return False
                    return True
            return False
        # async thread
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
            bool: True if all ranks are done, False if at least one rank is not completed.
        """
        if self.async_save is False:
            return True
        if get_group_size() == 1:
            return not is_alive
        ten = Tensor([is_alive], dtype=mstype.int8)
        ten, _ = comm_func.all_reduce(ten)
        return ten[0] == 0


def save_checkpoint_from_callback(cb: "CheckpointMonitor", cb_params):
    """Save checkpoint from CheckpointMonitor."""
    logger.info(f"---save checkpoint from callback---")

    # get callback parameters from cb_params
    network = cb_params.network  # TODO: need to be fixed, cb_params.train_network ?
    optimizer = cb_params.optimizer if cb_params.optimizer is not None else cb_params.network.optimizer
    step_num = cb._append_step_num + cb_params.cur_step_num  # pylint: disable=protected-access
    epoch_num = cb_params.cur_epoch_num
    global_step = int(optimizer.global_step)
    loss_scale = None
    if isinstance(cb_params.net_outputs, (tuple, list)) and len(cb_params.net_outputs) >= 3:
        loss_scale = float(cb_params.net_outputs[2])
    global_batch_size = int(cb.global_batch_size)

    async_save = cb._config.async_save  # pylint: disable=protected-access
    if async_save:
        async_save_manager: AsyncSaveManager = cb.async_save_manager

    # create checkpoint directory
    # NOTE: new checkpoints are saved to new_checkpoint directory, can be modified later
    checkpoint_path = get_output_subpath("new_checkpoint", append_rank=False)
    logger.info(f"checkpoint_path: {checkpoint_path}")
    checkpoint_iter_dir = utils.get_checkpoint_iter_dir(checkpoint_path, step_num)
    logger.info(f"checkpoint_iter_dir: {checkpoint_iter_dir}")
    if get_rank() == 0:
        utils.ensure_directory_exists(checkpoint_iter_dir, check_parent=False)
    if get_group_size() > 1:
        comm_func.barrier()  # barrier here to ensure directory exists

    # save common.json
    if get_rank() == 0:
        logger.info("start save common info")
        start_time = time()
        common_info = CommonInfo()
        common_info['step_num'] = step_num
        common_info['epoch_num'] = epoch_num
        common_info['global_step'] = global_step
        common_info['loss_scale'] = loss_scale
        common_info['global_batch_size'] = global_batch_size
        common_filename = utils.get_common_filename(checkpoint_path, step_num)
        common_info.save_common(common_filename)
        logger.info(f"save common info cost time: {time() - start_time:.3f}s")

    # prepare async save manager before save
    def iter_finalize_func():
        """Save checkpoint finalize function."""
        tracker_filename = utils.get_checkpoint_tracker_filename(checkpoint_path)
        logger.info(f"save checkpoint tracker file to {tracker_filename}")
        with open(tracker_filename, "w") as f:
            f.write(str(step_num))
        if async_save:
            logger.info(f"successfully async saved checkpoint from step {step_num} to {checkpoint_path}")
        else:
            logger.info(f"successfully sync saved checkpoint from step {step_num} to {checkpoint_path}")

    if async_save:
        async_save_manager.prepare_before_save()
        if get_rank() == 0:
            async_save_manager.add_finalize_fn(iter_finalize_func)

    # save model
    # NOTE: only basic save_checkpoint is called here, no optimizer or remove_redundancy is called, remain to be updated
    ckpt_filename = os.path.join(checkpoint_iter_dir, get_checkpoint_name(None, get_rank(), get_group_size(), 'Model'))
    logger.info(f"save checkpoint to {ckpt_filename}")
    ms_save_checkpoint(network, ckpt_filename, async_save=async_save, format="safetensors")

    # save tracker file in sync save process
    if not async_save:
        logger.info("barrier all ranks for sync save ckpt")
        if get_group_size() > 1:
            comm_func.barrier()
        logger.info("rank0 execute finalize func")
        if get_rank() == 0:
            iter_finalize_func()


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
    """
    Generate a checkpoint name for model parameters or optimizer parameters.
    Args:
        use_prefix (str): The prefix to use for the checkpoint file name.
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

    if use_prefix is None:
        file_name = f'{type_prefix}-{file_idx:07d}-{total_file_num:07d}'
    else:
        file_name = f'{use_prefix}-{type_prefix}-{file_idx:07d}-{total_file_num:07d}'
    return file_name
