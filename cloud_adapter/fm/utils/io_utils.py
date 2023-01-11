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
import hashlib
import os
import platform
import stat
import yaml

from fm.aicc_tools.ailog.log import service_logger, service_logger_without_std

# flags: 允许读写, 文件不存在时新建
DEFAULT_FLAGS = os.O_RDWR | os.O_CREAT
# modes: 所有者读写
DEFAULT_MODES = stat.S_IWUSR | stat.S_IRUSR


def write_file_with_link_check(full_file_path, content, flags, modes):
    """
    写入本地文件(含软链接检查)
    """
    # 软链接校验
    if os.path.exists(full_file_path) and is_link(full_file_path):
        service_logger.error('detect link, illegal file path {}'.format(full_file_path))
        raise RuntimeError

    with os.fdopen(os.open(full_file_path, flags, modes), 'w') as f:
        yaml.dump(content, stream=f)
    service_logger_without_std.info('writing file resource')


def read_file_with_link_check(full_file_path, flags, modes):
    """
    读取本地文件(含软链接检查)
    """
    # 软链接校验
    if os.path.exists(full_file_path) and is_link(full_file_path):
        service_logger.error('detect link, illegal file path {}'.format(full_file_path))
        raise RuntimeError

    with os.fdopen(os.open(full_file_path, flags, modes), 'rb') as f:
        app_config_output = yaml.safe_load(f)

    service_logger_without_std.info('reading file resource')
    return app_config_output


def wrap_local_working_directory(file_name, specific_path_config=None):
    """
    根据当前用户身份, 组装对应本地缓存文件路径
    默认文件存放位置$HOME/.cache/Huawei/mxFoundationModel
    需要存放特定文件夹需指定 specific_path_config = {'path': 路径, 'rule': 权限范围}
    """
    if file_name is None or file_name == '':
        service_logger.error('illegal parameter file_name, check the parameter.')
        raise RuntimeError

    # 针对Linux环境, 在当前用户$HOME目录下指定文件夹存放文件, 同时约束文件夹访问权限
    if platform.system().lower() == 'linux':
        home_path = os.getenv('HOME')

        if home_path is None:
            service_logger.error('can not find environment param: HOME in linux, check the env setting.')
            raise RuntimeError

        full_path = os.path.join(home_path, '.cache', 'Huawei', 'mxFoundationModel')

        if not os.path.exists(full_path):
            os.makedirs(full_path, exist_ok=True)
            # mxFoundationModel文件夹权限 rwxr-x---
            os.chmod(full_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP)
    else:
        service_logger.error('unknown system type, only support Linux.')
        raise RuntimeError

    if specific_path_config is not None:
        specific_path, specific_path_rule = specific_path_config_legality_check(specific_path_config)

        full_path = os.path.join(full_path, specific_path)

        if not os.path.exists(full_path):
            os.makedirs(full_path, exist_ok=True)
            os.chmod(full_path, specific_path_rule)

    wrap_path = os.path.join(full_path, file_name).replace('\\', '/')

    return wrap_path


def specific_path_config_legality_check(specific_path_config):
    specific_path = specific_path_config.get('path')

    if specific_path is None:
        service_logger.error('illegal specific_path_config param, specific_path is required.')
        raise RuntimeError

    specific_path_rule = specific_path_config.get('rule')

    if specific_path_rule is None:
        service_logger.error('illegal specific_path_config param, specific_path_rule is required.')
        raise RuntimeError

    return [specific_path, specific_path_rule]


def is_link(file_path):
    """
    check whether file_path is a link
    """
    return os.path.islink(file_path)


def get_config_dir_setting():
    return {'path': 'config', 'rule': stat.S_IRWXU}


def get_ca_dir_setting():
    return {'path': 'cer', 'rule': stat.S_IRWXU}


def calculate_file_md5(file_path):
    dig = hashlib.md5()
    with open(file_path, 'rb') as f:
        for data in iter(lambda: f.read(1024), b''):
            dig.update(data)
    return dig.hexdigest()
