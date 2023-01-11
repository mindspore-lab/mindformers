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
import platform

import fm.kmc.kmc_constants as kmc_constants
from fm.aicc_tools.ailog.log import service_logger_without_std


def read_os_file():
    os_list = []
    os_dict = {}
    try:
        with open('/etc/os-release') as f:
            for line in f:
                os_list.append(line)

    except IOError as err:
        service_logger_without_std.error('open os-release file failed, catch err: %s', err)

    for line in os_list:
        if '=' in line:
            key, value = line.rstrip().split('=')
            os_dict[key] = value.strip('"')
    return os_dict


def get_os_name():
    os_key = 'ID'
    os_dict = read_os_file()

    try:
        os_name = os_dict[os_key]
    except KeyError as err:
        raise KeyError('get os name from env failed.') from err

    if 'euler' in os_name.lower():
        os_name = 'euler'
    return os_name


def get_os_type():
    try:
        os_type = platform.uname()[kmc_constants.UNAME_MATCHINE_INDEX]
    except ValueError as err:
        service_logger_without_std.error('get os type from env failed, catch err: %s', err)
        raise ValueError('get os type from env failed.') from err
    if 'x86_64' in os_type:
        os_type = 'x86'
    elif 'aarch64' in os_type or 'arm64' in os_type:
        os_type = 'arm'

    return os_type


def get_kmc_lib_path():
    try:
        os_name = get_os_name()
    except KeyError as err:
        raise KeyError('get os name (ubuntu/centos/euler) failed.') from err

    try:
        os_type = get_os_type()
    except ValueError as err:
        raise ValueError('get os type (x86_64/arm) failed.') from err

    kmc_lib_path = 'kmc_{name}_{type}_lib'.format(
        name=os_name, type=os_type)
    return kmc_lib_path
