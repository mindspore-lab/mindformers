# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Utils For Tools."""
import json
import os
import re
import shutil
import stat
import tempfile
from multiprocessing import Process
from typing import Dict, List, Tuple, Union
from importlib import import_module
from pathlib import Path
import numpy as np
import psutil

try:
    import fcntl
except ImportError:
    fcntl = None

import mindspore as ms
from mindspore import Tensor, context
from mindspore._checkparam import args_type_check
from mindspore.communication import get_group_size, get_rank, comm_func, get_local_rank

PARALLEL_MODE = {'DATA_PARALLEL': context.ParallelMode.DATA_PARALLEL,
                 'SEMI_AUTO_PARALLEL': context.ParallelMode.SEMI_AUTO_PARALLEL,
                 'AUTO_PARALLEL': context.ParallelMode.AUTO_PARALLEL,
                 'HYBRID_PARALLEL': context.ParallelMode.HYBRID_PARALLEL,
                 'STAND_ALONE': context.ParallelMode.STAND_ALONE,
                 'MANUAL_PARALLEL': context.ParallelMode.STAND_ALONE,
                 0: context.ParallelMode.DATA_PARALLEL,
                 1: context.ParallelMode.SEMI_AUTO_PARALLEL,
                 2: context.ParallelMode.AUTO_PARALLEL,
                 3: context.ParallelMode.HYBRID_PARALLEL}

MODE = {'PYNATIVE_MODE': context.PYNATIVE_MODE,
        'GRAPH_MODE': context.GRAPH_MODE,
        0: context.GRAPH_MODE,
        1: context.PYNATIVE_MODE}

DEBUG_INFO_PATH = '/cache/debug'
PROFILE_INFO_PATH = '/cache/profile'
PLOG_PATH = '/root/ascend/log'
LOCAL_DEFAULT_PATH = os.getenv("LOCAL_DEFAULT_PATH", './output')
LAST_TRANSFORM_LOCK_PATH = "/tmp/last_transform_done.lock"

_PROTOCOL = 'obs'
_PROTOCOL_S3 = 's3'

FILE_PERMISSION = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP
DIRECTORY_PERMISSION = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP


def check_in_modelarts():
    """Check if the training is on modelarts.

    Returns:
        (bool): If it is True, it means ModelArts environment.
    """
    return 'MA_LOG_DIR' in os.environ or \
        'MA_JOB_DIR' in os.environ or \
        'MA_LOCAL_LOG_PATH' in os.environ or \
        'BATCH_GROUP_NAME' in os.environ or \
        'MA_LOCAL_LOG_PATH' in os.environ


class Validator:
    """validator for checking input parameters"""

    @staticmethod
    def check_type(arg_value, arg_type):
        """Check int."""
        if not isinstance(arg_value, arg_type):
            raise TypeError('{} should be {} type, but get {}'.format(arg_value, arg_type, type(arg_value)))

    @staticmethod
    def is_obs_url(url):
        """Check obs url."""
        return url.startswith(_PROTOCOL + '://') or url.startswith(_PROTOCOL_S3 + '://')


def check_obs_url(url):
    """Check obs url."""
    if not isinstance(url, str):
        raise TypeError('remote_save_url type should be a str, but get {}, '
                        'please check your remote_save_url config'.format(type(url)))
    if not (url.startswith(_PROTOCOL + '://') or url.startswith(_PROTOCOL_S3 + '://')):
        raise TypeError('remote_save_url should be start with obs:// or s3://, '
                        'but get {}, please check your remote_save_url config'.format(url))


def check_list(var_name: str, list_var: Union[Tuple, List], num: int):
    """Checks the legitimacy of elements within a node or device list.

    Args:
        var_name (str): The Name of variable need to check.
        list_var (tuple or list): Variables in list format.
        num (int): The number of nodes or devices.

    Returns:
        None
    """
    for value in list_var:
        if value >= num:
            raise ValueError('The index of the {} needs to be less than the number of nodes {}.'.format(var_name, num))


def check_file(file_path, file_type=None):
    """Check file."""
    if not os.path.exists(file_path):
        raise ValueError(f"The file_path:{file_path} is not exist.")
    if not os.path.isfile(file_path):
        raise ValueError(f"The file_path:{file_path} is not a {file_type} file.")


def format_path(path):
    """Check path."""
    if not path:
        return path
    return os.path.realpath(path)


def sync_trans(f):
    """Asynchronous transport decorator."""
    try:
        def wrapper(*args, **kwargs):
            pro = Process(target=f, args=args, kwargs=kwargs)
            pro.start()
            return pro

        return wrapper
    except Exception as e:
        raise e


def get_output_root_path():
    """get default output path in local/AICC."""
    path = os.getenv("LOCAL_DEFAULT_PATH", './output')
    expanduser_path = os.path.expanduser(path)
    return os.path.realpath(expanduser_path)


@args_type_check(path=str)
def set_output_path(path):
    """set output path"""
    from .logger import logger
    if path is None:
        path = './output'
    expanduser_path = os.path.expanduser(path)
    os.environ['LOCAL_DEFAULT_PATH'] = os.path.realpath(expanduser_path)
    logger.info(f"set output path to '{os.path.realpath(expanduser_path)}'")


def set_strategy_save_path(config):
    """set strategy path"""
    from .logger import logger
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


def get_log_path():
    path = os.getenv("LOG_MF_PATH", os.path.join(get_output_root_path(), "log"))
    return os.path.expanduser(path)


def get_output_subpath(sub_class, rank_id=0, append_rank=True):
    """get output store path for sub output class."""
    Validator.check_type(sub_class, str)
    root_path = get_output_root_path()
    directory = os.path.join(root_path, sub_class)
    if append_rank:
        directory = os.path.join(directory, 'rank_{}'.format(rank_id))
    return format_path(directory)


def set_remote_save_url(remote_save_url):
    check_obs_url(remote_save_url)
    os.environ.setdefault('REMOTE_SAVE_URL', remote_save_url)


def get_remote_save_url():
    return os.environ.get('REMOTE_SAVE_URL', None)


def get_net_outputs(params):
    """Get network outputs."""
    if isinstance(params, (tuple, list)):
        if isinstance(params[0], Tensor) and isinstance(params[0].asnumpy(), np.ndarray):
            params = params[0]
    elif isinstance(params, Tensor) and isinstance(params.asnumpy(), np.ndarray):
        params = np.mean(params.asnumpy())
    return params


def get_rank_info() -> Tuple[int, int]:
    """Get rank_info from environment variables.

    Returns:
        rank_id (int): Rank id.
        rand_size (int): The number of rank.
    """
    rank_id = get_real_rank()
    rank_size = get_real_group_size()

    return rank_id, rank_size


def get_num_nodes_devices(rank_size: int) -> Tuple[int, int]:
    """Derive the number of nodes and devices based on rank_size.

    Args:
        rank_size (int): rank size.

    Returns:
       num_nodes (int): number of nodes.
       num_devices (int): number of devices.
    """
    if rank_size in (2, 4, 8):
        num_nodes = 1
        num_devices = rank_size
    else:
        num_nodes = rank_size // 8
        num_devices = 8

    return num_nodes, num_devices


class Const:
    """Const."""

    def __setattr__(self, key, value):
        if key in self.__dict__:
            raise PermissionError('Can not change const {0}.'.format(key))
        if not key.isupper():
            raise ValueError('Const name {0} is not all uppercase.'.format(key))
        self.__dict__[key] = value


def generate_rank_list(stdout_nodes: Union[List, Tuple], stdout_devices: Union[List, Tuple]):
    """Generate a list of the ranks to output the log.

    Args:
        stdout_nodes (list or tuple): The compute nodes that will
            output the log to stdout.
        stdout_devices (list or tuple): The compute devices that will
            output the log to stdout.

    Returns:
        rank_list (list): A list of the ranks to output the log to stdout.
    """
    rank_list = []
    for node in stdout_nodes:
        for device in stdout_devices:
            rank_list.append(8 * node + device)

    return rank_list


def convert_nodes_devices_input(var: Union[List, Tuple, Dict[str, int], None], num: int) -> Union[List, Tuple]:
    """Convert node and device inputs to list format.

    Args:
        var (list[int] or tuple[int] or dict[str, int] or optional):
            The variables that need to be converted to the format.
        num (str): The number of nodes or devices.

    Returns:
        var (list[int] or tuple[int]): A list of nodes or devices.
    """
    if var is None:
        var = tuple(range(num))
    elif isinstance(var, dict):
        var = tuple(range(var['start'], var['end']))

    return var


def str2bool(b):
    """String convert to Bool."""
    if b.lower() in ["false"]:
        output = False
    elif b.lower() in ["true"]:
        output = True
    else:
        raise Exception("Invalid Bool Value")
    return output


def str2bool_or_str(b):
    """String convert to Bool or String."""
    mapping = {"false": False, "true": True}
    b_lower = b.lower()
    return mapping.get(b_lower, b)


def count_params(net):
    """Count number of parameters in the network
    Args:
        net (mindspore.nn.Cell): Mindspore network instance
    Returns:
        total_params (int): Total number of trainable params
    """
    total_params = [np.prod(param.shape) for param in net.trainable_params()]
    return sum(total_params) // 1000000


def try_sync_file(file_name):
    """If the file is still downloading, we need to wait before the file finished downloading"""
    if fcntl:
        with open(file_name, 'r') as fp:
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX)


def is_version_le(current_version, base_version):
    """
        return current_version <= base_version.
        Check whether the current version is lower than or equal to the base version.
        For example: for current_version: 1.8.1, base_version: 1.11.0, it return True.
    """
    version_split_char = '.'
    if version_split_char not in base_version or version_split_char not in current_version:
        raise ValueError("The version string will contain the `.`."
                         "For example, current_version 1.8.1, base_version: 1.11.0.")
    for x, y in zip(current_version.split(version_split_char), base_version.split(version_split_char)):
        if not x.isdigit() or not y.isdigit():
            continue
        if int(x) != int(y):
            return int(x) <= int(y)
    return True


def is_version_ge(current_version, base_version):
    """
        return current_version >= base_version.
        Check whether the current version is higher than or equal to the base version.
        for current_version: 1.8.1, base_version: 1.11.0, it return False.
    """
    version_split_char = '.'
    if version_split_char not in base_version or version_split_char not in current_version:
        raise ValueError("The version string will contain the `.`."
                         "For example, current_version 1.8.1， base_version: 1.11.0.")
    for x, y in zip(current_version.split(version_split_char), base_version.split(version_split_char)):
        if not x.isdigit() or not y.isdigit():
            continue
        if int(x) != int(y):
            return int(x) >= int(y)
    return True


def parse_value(value):
    """
        parse value from command line.
        handles with int, float, bool, string, list and dict.
    """

    def isint(x):
        try:
            a = float(x)
            b = int(a)
        except (TypeError, ValueError):
            return False
        else:
            return a == b

    def isfloat(x):
        try:
            float(x)
        except (TypeError, ValueError):
            return False
        else:
            return True

    def isbool(x):
        return x in ["True", "False"]

    def isjson(x):
        try:
            json.loads(x)
        except json.decoder.JSONDecodeError:
            return False
        else:
            return True

    if isint(value):
        return int(value)
    if isfloat(value):
        return float(value)
    if isbool(value):
        return value == "True"
    if isjson(value):
        return json.loads(value)
    return value


def replace_tk_to_mindpet(ckpt_dict):
    """replace 'tk_delta' in para name to 'mindpet_delta'"""
    ckpt_new = {}
    for k, v in ckpt_dict.items():
        ckpt_new[k.replace('tk_delta', 'mindpet_delta')] = v
    return ckpt_new


def check_shared_disk(disk_path):
    """
    Check whether the disk_path is a shared path.

    Args:
    disk_path (str): The path to check.

    Returns:
    bool: True if the path is on a shared disk, False otherwise.
    """
    disk_path = os.path.abspath(disk_path)
    partitions = psutil.disk_partitions(all=True)
    for partition in partitions:
        if partition.mountpoint != '/' and disk_path.startswith(partition.mountpoint):
            # Check if the partition is a network file system (NFS) or other network storage
            fstype = partition.fstype.lower()
            opts = partition.opts.lower()
            if fstype in ['nfs', 'dpc', 'nfs4'] or 'fuse.sshfs' in opts:
                return True
    return False


def check_in_dynamic_cluster():
    """check if in dynamic cluster."""
    return "MS_ROLE" in os.environ


def get_real_rank():
    try:
        return get_rank()
    except RuntimeError:
        return int(os.getenv("RANK_ID", "0"))


def get_real_local_rank():
    """Get local rank id from current collective communication group."""
    try:
        return get_local_rank()
    except RuntimeError:
        return 0


def get_dp_from_dataset_strategy():
    data_strategy = ms.get_auto_parallel_context("dataset_strategy")
    if isinstance(data_strategy, (tuple, list)):
        first_input_stra = data_strategy[0]
        dp = int(first_input_stra[0])
    else:
        raise TypeError(f"Dataset_strategy in mindspore auto parallel context is invalid, only support (tuple, list)")
    return dp


def get_real_group_size():
    try:
        return get_group_size()
    except RuntimeError:
        return int(os.getenv("RANK_SIZE", "1"))


def get_device_num_per_node():
    return int(os.getenv("DEVICE_NUM_PER_NODE", "8"))


def get_predict_run_mode():
    run_mode = os.environ.get("RUN_MODE")
    return run_mode == "predict"


def is_main_rank(ignore_check_modelarts=False):
    return not get_real_rank() or \
        ((ignore_check_modelarts or check_in_modelarts()) and get_real_rank() % get_device_num_per_node() == 0)


def is_last_pipeline_stage():
    """get if current rank is in the last stage of pipeline parallelism"""
    device_num = get_real_group_size()
    stage_num = ms.get_auto_parallel_context("pipeline_stages")
    device_num_per_stage = device_num // stage_num
    rank = get_real_rank()
    return (rank // device_num_per_stage + 1) == stage_num


def is_publicly_accessible_path(path):
    """Check a path is accessible by all rank."""
    from .logger import logger
    if get_real_group_size() <= get_device_num_per_node():
        return True

    if check_in_modelarts():
        return True

    if check_shared_disk(path):
        return True

    # For example, SHARED_PATHS="/mnt/shared1,/mnt/shared2"
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


def create_file(file_path, info=None):
    """create file."""
    if Validator.is_obs_url(file_path):
        if not check_in_modelarts():
            raise ValueError(f"When create {file_path}, \
            it is detected that it is not in the ModelArts platform.")
        import moxing as mox
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
        import moxing as mox
        if mox.file.exists(file_path):
            mox.file.remove(file_path, recursive=False)
    else:
        if os.path.exists(file_path):
            os.remove(file_path)


def remake_folder(folder_path, permissions=None):
    """make folder"""
    from .logger import logger
    rank_id = get_real_rank()
    logger.info("Remake %s...", folder_path)
    if Validator.is_obs_url(folder_path):
        if not check_in_modelarts():
            raise ValueError(f"When remaking {folder_path}, \
            it is detected that it is not in the ModelArts platform.")
        import moxing as mox
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
    from .logger import logger
    rank_id = rank_id or get_real_rank()
    logger.info("Remove %s...", folder_path)
    if Validator.is_obs_url(folder_path):
        if not check_in_modelarts():
            raise ValueError(f"When removing {folder_path}, \
            it is detected that it is not in the ModelArts platform.")
        import moxing as mox
        if mox.file.exists(folder_path) and not rank_id:
            mox.file.remove(folder_path, recursive=True)
            logger.info("OBS: Folder %s is removed.", folder_path)
    else:
        if os.path.exists(folder_path) and is_main_rank():
            shutil.rmtree(folder_path)
            logger.info("Folder %s is removed.", folder_path)


def set_safe_mode_for_file_or_dir(path):
    if isinstance(path, str):
        path = [path]
    for item in path:
        item = Path(item)
        if item.is_dir():
            item.chmod(DIRECTORY_PERMISSION)
        if item.is_file():
            item.chmod(FILE_PERMISSION)


def get_epoch_and_step_from_ckpt_name(ckpt_file, ckpt_fmt='ckpt'):
    """Get epoch and step from ckpt name."""
    ckpt_name = os.path.basename(ckpt_file)
    pattern = r'-(\d+)_(\d+)\.' + ckpt_fmt
    match = re.search(pattern, ckpt_name)
    if match:
        epoch = int(match.group(1))
        step = int(match.group(2))
        return epoch, step
    raise ValueError(f"Can't match epoch and step from checkpoint file: {ckpt_file}. Please ensure the format "
                     f"of the checkpoint file name is {{prefix}}-{{epoch}}_{{step}}.{ckpt_fmt}, for example, "
                     f"llama_7b_rank_0-3_2.{ckpt_fmt}.")


def get_times_epoch_and_step_from_ckpt_name(ckpt_file, ckpt_fmt='ckpt'):
    """Get times epoch and step from ckpt name."""
    ckpt_name = os.path.basename(ckpt_file)
    pattern = r'_(\d+)-(\d+)_(\d+)\.' + ckpt_fmt
    pattern_1 = r'-(\d+)_(\d+)\.' + ckpt_fmt
    match = re.search(pattern, ckpt_name)
    if match:
        times = int(match.group(1))
        epoch = int(match.group(2))
        step = int(match.group(3))
        return times, epoch, step
    match_1 = re.search(pattern_1, ckpt_name)
    if match_1:
        times = 0
        epoch = int(match.group(1))
        step = int(match.group(2))
        return times, epoch, step
    raise ValueError(f"Can't match epoch and step from checkpoint file: {ckpt_file}. Please ensure the format "
                     f"of the checkpoint file name is {{prefix}}-{{epoch}}_{{step}}.{ckpt_fmt}, for example, "
                     f"llama_7b_rank_0-3_2.{ckpt_fmt}.")


def get_rank_id_from_ckpt_name(ckpt_file):
    """Get rank id from ckpt name."""
    ckpt_name = os.path.basename(ckpt_file)
    match = re.search(r'_rank_(\d+)', ckpt_name)
    if match:
        rank_id = int(match.group(1))
        return rank_id
    raise ValueError(f"Can't match rank id from checkpoint file: {ckpt_file}. Please ensure the name of "
                     f"the checkpoint file is xxx_rank_x-{{epoch}}_{{step}}, for example, llama_7b_rank_0-3_2.")


def replace_rank_id_in_ckpt_name(ckpt_file, dst_rank_id):
    """Replace rank id to dst_rank_id in ckpt name"""
    ckpt_name = os.path.basename(ckpt_file)
    ori_rank_id = get_rank_id_from_ckpt_name(ckpt_name)
    ckpt_name = ckpt_name.replace(f"_rank_{ori_rank_id}", f"_rank_{dst_rank_id}")
    return ckpt_name


def clear_auto_trans_output(load_checkpoint=None, src_strategy_path_or_dir=None):
    """clear transformed_checkpoint and strategy"""
    folder_list = ["strategy", "transformed_checkpoint"]
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


def check_ckpt_file_name(ckpt_file, ckpt_fmt='ckpt'):
    """Check ckpt name in the format of {prefix}-{epoch}_{step}.ckpt"""
    ckpt_name = os.path.split(ckpt_file)[1]
    pattern = r'^[^/]+-\d+_\d+\.' + ckpt_fmt + r"$"
    match = re.match(pattern, ckpt_name)
    if match:
        return True
    return False


def create_and_write_info_to_txt(txt_path, info=None):
    """create and write info to txt"""
    if os.path.exists(txt_path):
        raise ValueError(f"{txt_path} already exists.")
    dir_path = os.path.dirname(txt_path)
    with tempfile.NamedTemporaryFile('w', delete=False, dir=dir_path) as temp_file:
        if info:
            if isinstance(info, str):
                temp_file.write(info)
            elif isinstance(info, list):
                for sub_info in info:
                    temp_file.write(sub_info + "\n")
            else:
                raise ValueError(f"The info to write should be str or list, but get {info}")
        temp_file_path = temp_file.name
    os.replace(temp_file_path, txt_path)


def is_pynative():
    """get whether the mode is pynative"""
    enforce_eager = os.getenv('ENFORCE_EAGER', "False")
    return enforce_eager.lower() == "true"


def barrier_world(action: str = None):
    """barrier all rank until action is done"""
    if get_real_group_size() > 1:
        from .logger import logger
        if action is not None:
            logger.info("Wait " + str(action))
        else:
            logger.info("Now barriered...")

        comm_func.barrier()


def get_pipeline_rank_ids():
    """Calculate rank id of each stage and return a list of first rank id in each stage.

    Returns:
        pipeline_rank_ids: a list of pipeline rank ids or
                           an invalid value(-1) if the configuration of pp is invalid.
    """
    device_num = get_real_group_size()
    current_stage_num = ms.get_auto_parallel_context('pipeline_stages')

    if device_num % current_stage_num != 0:
        return [-1]

    devices_per_stage = device_num // current_stage_num
    pipeline_rank_ids = [i * devices_per_stage for i in range(current_stage_num)]

    return pipeline_rank_ids


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    if numerator % denominator != 0:
        raise ValueError("{} is not divisible by {}".format(numerator, denominator))


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def calculate_pipeline_stage(layers_per_stage, model_layers):
    r"""Calculate pipeline stage for model

    Args:
        layers_per_stage (list): The number of layers per stage.
        model_layers (list): The number of layers of each part of the model.

    """
    pipeline_stages = []
    cur_stage = 0  # Initialize the start stage counter

    # Iterate directly over the model layers
    for model_layer in model_layers:
        model_layer_remaining = model_layer
        start_stage = cur_stage
        end_stage = cur_stage
        offset = []

        while cur_stage < len(layers_per_stage) and model_layer_remaining > 0:
            if model_layer_remaining < layers_per_stage[cur_stage]:
                layers_per_stage[cur_stage] -= model_layer_remaining
                offset.append(model_layer_remaining)
                model_layer_remaining = 0
            else:
                model_layer_remaining -= layers_per_stage[cur_stage]
                offset.append(layers_per_stage[cur_stage])
                layers_per_stage[cur_stage] = 0
                cur_stage += 1
            end_stage += 1

        stage_num = end_stage - start_stage
        avg_layer_per_stage = model_layer // stage_num if stage_num > 0 else 0

        for j in range(stage_num):
            offset[j] -= avg_layer_per_stage

        pipeline_stage = {
            "offset": offset,
            "start_stage": start_stage,
            "stage_num": stage_num
        }

        pipeline_stages.append(pipeline_stage)

    return pipeline_stages


def get_context(attr_key, default_value=None):
    context_module = import_module("mindformers.core.context.build_context")
    if context_module.Context.is_exists():
        attr_value = context_module.get_context(attr_key)
        if attr_value is not None:
            return attr_value
    return default_value


def get_ascend_log_path():
    """Get Ascend log path: $ASCEND_PROCESS_LOG_PATH > $ASCEND_WORK_PATH/log > default($HOME/ascend/log)"""
    ascend_log_path = os.getenv('ASCEND_PROCESS_LOG_PATH')
    if ascend_log_path:
        return ascend_log_path
    ascend_log_path = os.getenv('ASCEND_WORK_PATH')
    if ascend_log_path:
        return os.path.join(ascend_log_path, 'log')
    return os.path.join(os.path.expanduser("~"), 'ascend', 'log')
