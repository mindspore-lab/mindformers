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
import os
from typing import Dict, List, Tuple, Union

from fm.src.aicc_tools.utils import Const

LOG_CONTENT_BLACK_LIST = ('\r', '\n', '\t', '\v', '\f', '\u000A', '\u000B', '\u000C', '\u000D', '\u0009')
BLANKS = '    '


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


def create_dirs(path, mode=0o750):
    """Recursively create folders."""
    p_path = os.path.join(path, "..")
    abspath = os.path.abspath(p_path)
    if not os.path.exists(abspath):
        os.makedirs(abspath, mode, exist_ok=True)
    if not os.path.exists(path):
        os.mkdir(path, mode)


def log_args_black_list_characters_replace(args):
    res = list()
    if args is None or len(args) == 0:
        return args
    if isinstance(args, (list, tuple)):
        for arg in args:
            replace = character_replace(content=arg)
            res.append(replace)
        args = tuple(res)
    else:
        args = character_replace(args)
    return args


def character_replace(content):
    if not isinstance(content, str):
        return content
    for forbidden_str in LOG_CONTENT_BLACK_LIST:
        if forbidden_str in content:
            content = content.replace(forbidden_str, '')
    while BLANKS in content:
        content = content.replace(BLANKS, ' ')
    return content


const = Const()
const.level = ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
const.modelarts_log_file_dir = '/cache/ma-user-work/service'
const.local_default_log_file_dir = '~/.cache/Huawei/mxFoundationModel/log/'
const.rank_dir_formatter = 'rank_{}/'
const.default_filehandler_format = '[%(levelname)s] %(asctime)s - [%(source)s] - [%(filename)s:%(lineno)d] - %(message)s'
const.default_stdout_format = '[%(levelname)s] %(asctime)s - [%(source)s] - %(message)s'
