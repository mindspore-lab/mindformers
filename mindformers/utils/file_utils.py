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
""" File utils."""
import os
import shutil

from mindspore import context
from mindspore._checkparam import args_type_check

from mindformers.tools.logger import logger
from mindformers.tools.utils import (
    Validator,
    FILE_PERMISSION,
    check_in_modelarts,
    get_real_rank,
    set_safe_mode_for_file_or_dir,
    get_output_root_path,
    get_real_group_size,
    get_device_num_per_node,
    get_remote_save_url,
    check_shared_disk,
    is_main_rank
)

if check_in_modelarts():
    import moxing as mox
else:
    mox = None


@args_type_check(path=str)
def set_output_path(path):
    """set output path"""
    if path is None:
        path = './output'
    expanduser_path = os.path.expanduser(path)
    os.environ['LOCAL_DEFAULT_PATH'] = os.path.realpath(expanduser_path)
    logger.info(f"set output path to '{os.path.realpath(expanduser_path)}'")


def set_strategy_save_path(config):
    """set strategy path"""
    rank_id = get_real_rank()
    strategy_ckpt_save_dir = os.path.join(get_output_root_path(), "strategy")
    os.makedirs(strategy_ckpt_save_dir, exist_ok=True)
    set_safe_mode_for_file_or_dir(strategy_ckpt_save_dir)

    strategy_ckpt_save_file = config.get('strategy_ckpt_save_file', "ckpt_strategy.ckpt")
    if not strategy_ckpt_save_file.endswith(f"_rank_{rank_id}.ckpt"):
        strategy_name = os.path.basename(strategy_ckpt_save_file).replace(".ckpt", f"_rank_{rank_id}.ckpt")
        config['strategy_ckpt_save_file'] = os.path.join(strategy_ckpt_save_dir, strategy_name)
        context.set_auto_parallel_context(strategy_ckpt_save_file=config['strategy_ckpt_save_file'])
        logger.info(f"set strategy path to '{config['strategy_ckpt_save_file']}'")


def set_checkpoint_save_path():
    """set checkpoint save path"""
    checkpoint_save_path = os.path.join(get_output_root_path(), "checkpoint")
    os.makedirs(checkpoint_save_path, exist_ok=True)
    set_safe_mode_for_file_or_dir(checkpoint_save_path)
    logger.info(f"set checkpoint save path to `{checkpoint_save_path}`")


def is_publicly_accessible_path(path):
    """Check a path is accessible by all rank."""
    if get_real_group_size() <= get_device_num_per_node():
        return True

    if check_in_modelarts():
        return True

    if check_shared_disk(path):
        return True

    # For example, SHARED_PATHS="/mnt/shared1,/mnt/shared2", which will be split by "/mnt/shared1" and "/mnt/shared2".
    shared_paths = os.getenv("SHARED_PATHS", "").split(',')
    path = os.path.realpath(path)
    for shared_path in shared_paths:
        if not shared_path:
            continue
        shared_path = os.path.realpath(shared_path)
        if path.startswith(shared_path):
            return True
    logger.info("System can not identify if given path is shared disk. "
                "If it is, Please set env 'SHARED_PATHS' to given path.")
    return False


def remake_folder(folder_path, permissions=None):
    """make folder"""
    rank_id = get_real_rank()
    logger.info("Remake %s...", folder_path)
    if Validator.is_obs_url(folder_path):
        if not check_in_modelarts():
            raise ValueError(f"When remaking {folder_path}, \
            it is detected that it is not in the ModelArts platform.")
        if not rank_id:
            if mox.file.exists(folder_path):
                mox.file.remove(folder_path, recursive=True)
            mox.file.make_dirs(folder_path)
            logger.info("OBS: Folder %s is remaked.", folder_path)
    else:
        if is_main_rank():
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
            os.makedirs(folder_path, exist_ok=True)
            os.chmod(folder_path, permissions)
            logger.info("Folder %s is remaked.", folder_path)


def remove_folder(folder_path, rank_id=None):
    """delete folder"""
    rank_id = rank_id or get_real_rank()
    logger.info("Remove %s...", folder_path)
    if Validator.is_obs_url(folder_path):
        if not check_in_modelarts():
            raise ValueError(f"When removing {folder_path}, \
            it is detected that it is not in the ModelArts platform.")
        if mox.file.exists(folder_path) and not rank_id:
            mox.file.remove(folder_path, recursive=True)
            logger.info("OBS: Folder %s is removed.", folder_path)
    else:
        if os.path.exists(folder_path) and is_main_rank():
            shutil.rmtree(folder_path)
            logger.info("Folder %s is removed.", folder_path)


def create_file(file_path, info=None):
    """create file."""
    if Validator.is_obs_url(file_path):
        if not check_in_modelarts():
            raise ValueError(f"When create {file_path}, \
            it is detected that it is not in the ModelArts platform.")
        with mox.file.File(file_path, 'w') as f:
            if info:
                if isinstance(info, list):
                    for sub_info in info:
                        f.write(str(sub_info) + "\n")
                else:
                    f.write(info)
            else:
                f.write("Hugging ModelArts.")
    else:
        flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        with os.fdopen(os.open(file_path, flags_, FILE_PERMISSION), 'w') as f:
            if info:
                if isinstance(info, list):
                    for sub_info in info:
                        f.write(str(sub_info) + "\n")
                else:
                    f.write(info)


def delete_file(file_path):
    """delete file"""
    if Validator.is_obs_url(file_path):
        if not check_in_modelarts():
            raise ValueError(f"When create {file_path}, \
            it is detected that it is not in the ModelArts platform.")
        if mox.file.exists(file_path):
            mox.file.remove(file_path, recursive=False)
    else:
        if os.path.exists(file_path):
            os.remove(file_path)


def clear_auto_trans_output(load_checkpoint=None, src_strategy_path_or_dir=None, load_ckpt_format='ckpt'):
    """
    Clear transformed_checkpoint and strategy folders based on the specified checkpoint format.

    Args:
        load_checkpoint (str, optional): Path to the checkpoint file/directory to load.
            If the target folder path matches this value, a ValueError is raised. Defaults to None.
        src_strategy_path_or_dir (str, optional): Path to the source strategy file/directory.
            If the target folder path matches this value, a ValueError is raised. Defaults to None.
        load_ckpt_format (str, optional): Format of the checkpoint file. Supports 'ckpt' and 'safetensors'.
            Determines which transformed checkpoint folder to clear. Defaults to 'ckpt'.

    Raises:
        ValueError: If `load_ckpt_format` is not one of the supported values ('ckpt' or 'safetensors').
        ValueError: If the resolved folder path (strategy or transformed/unified checkpoint) is the same as
            `load_checkpoint` or `src_strategy_path_or_dir`. This prevents overwriting critical input data.
    """
    if load_ckpt_format not in ('ckpt', 'safetensors'):
        raise ValueError(f"Invalid checkpoint format '{load_ckpt_format}'. "
                         "Only 'ckpt' and 'safetensors' formats are supported.")

    folder_list = ["strategy", "transformed_checkpoint"] if load_ckpt_format == 'ckpt' \
        else ["strategy", "unified_checkpoint"]
    for folder in folder_list:
        if check_in_modelarts():
            folder_path = os.path.join(get_remote_save_url(), folder)
        else:
            folder_path = os.path.join(get_output_root_path(), folder)
        if os.path.realpath(folder_path) in (load_checkpoint, src_strategy_path_or_dir):
            raise ValueError(
                "./transformed_checkpoint or ./strategy with given config.output_dir is same as "
                "load_checkpoint or src_strategy_path_or_dir which is not allowed when auto_trans is True."
                "Please move it to a different location or specify a different output folder.")
        remake_folder(folder_path, permissions=0o750)
