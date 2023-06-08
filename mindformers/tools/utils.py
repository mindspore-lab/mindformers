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
import os
from typing import Dict, List, Tuple, Union
from multiprocessing import Process

try:
    import fcntl
except ImportError:
    fcntl = None

import numpy as np

from mindspore import Tensor
from mindspore import context

PARALLEL_MODE = {'DATA_PARALLEL': context.ParallelMode.DATA_PARALLEL,
                 'SEMI_AUTO_PARALLEL': context.ParallelMode.SEMI_AUTO_PARALLEL,
                 'AUTO_PARALLEL': context.ParallelMode.AUTO_PARALLEL,
                 'HYBRID_PARALLEL': context.ParallelMode.HYBRID_PARALLEL,
                 'STAND_ALONE': context.ParallelMode.STAND_ALONE,
                 0: context.ParallelMode.DATA_PARALLEL,
                 1: context.ParallelMode.SEMI_AUTO_PARALLEL,
                 2: context.ParallelMode.AUTO_PARALLEL,
                 3: context.ParallelMode.HYBRID_PARALLEL}

MODE = {'PYNATIVE_MODE': context.PYNATIVE_MODE,
        'GRAPH_MODE': context.GRAPH_MODE,
        0: context.GRAPH_MODE,
        1: context.PYNATIVE_MODE}

SAVE_WORK_PATH = '/cache/ma-user-work/rank_{}'

MA_OUTPUT_ROOT = '/cache/ma-user-work'
DEBUG_INFO_PATH = '/cache/debug'
PROFILE_INFO_PATH = '/cache/profile'
PLOG_PATH = '/root/ascend/log'
LOCAL_DEFAULT_PATH = os.getenv("LOCAL_DEFAULT_PATH", './output')

_PROTOCOL = 'obs'
_PROTOCOL_S3 = 's3'


def check_in_modelarts():
    """Check if the training is on modelarts.

    Returns:
        (bool): If it is True, it means ModelArts environment.
    """
    # 'KUBERNETES_PORT' in os.environ or \
    return 'MA_LOG_DIR' in os.environ or \
           'MA_JOB_DIR' in os.environ or \
           'MA_LOCAL_LOG_PATH' in os.environ or \
           'S3_ACCESS_KEY_ID' in os.environ or \
           'S3_SECRET_ACCESS_KEY' in os.environ or \
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
    if not (url.startswith(_PROTOCOL + '://') or url.startswith(_PROTOCOL_S3 + '://')):
        raise TypeError('obs url should be start with obs:// or s3://, but get {}'.format(url))
    return True


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


def format_path(path):
    """Check path."""
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
    if check_in_modelarts():
        return MA_OUTPUT_ROOT
    return LOCAL_DEFAULT_PATH


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
    rank_id = int(os.getenv('RANK_ID', '0'))
    rank_size = int(os.getenv('RANK_SIZE', '1'))

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
        For example: for current_version: 1.8.1, base_version: 2.0.0, it return True.
    """
    version_split_char = '.'
    if version_split_char not in base_version or version_split_char not in current_version:
        raise ValueError("The version string will contain the `.`."
                         "For example, current_version 1.8.1, base_version: 2.0.0.")
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
        for current_version: 1.8.1, base_version: 2.0.0, it return False.
    """
    version_split_char = '.'
    if version_split_char not in base_version or version_split_char not in current_version:
        raise ValueError("The version string will contain the `.`."
                         "For example, current_version 1.8.1， base_version: 2.0.0.")
    for x, y in zip(current_version.split(version_split_char), base_version.split(version_split_char)):
        if not x.isdigit() or not y.isdigit():
            continue
        if int(x) != int(y):
            return int(x) >= int(y)
    return True
