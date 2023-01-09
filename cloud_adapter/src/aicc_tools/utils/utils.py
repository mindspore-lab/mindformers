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
from typing import Tuple

SERVICE_LOG_PATH = '/cache/ma-user-work/service'


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

    def __setattr__(self, key, value):
        if key in self.__dict__:
            raise PermissionError('Can not change const {0}.'.format(key))
        if not key.islower():
            raise ValueError('Const name {0} is not all lowercase.'.format(key))
        self.__dict__[key] = value
