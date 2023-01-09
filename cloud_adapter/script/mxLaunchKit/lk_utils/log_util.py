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
"""
功能: common log utils of launcher
"""

import os
import platform
import stat
import logging

DEFAULT_LOG_MSG_PATTERN = "%(asctime)s [%(levelname)s] [cli] [%(module)s:%(lineno)d] %(message)s"
DEFAULT_LOG_TIME_PATTERN = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOCAL_LOG_FILE_DIR = os.path.expanduser('~/.cache/Huawei/mxLaunchKit/log/')
DEFAULT_LOCAL_LOG_FILE_NAME = 'operation.log'


def wrap_local_working_directory(file_name, specific_path_config=None):
    """
    根据当前用户身份, 组装对应本地日志文件路径
    默认文件存放位置$HOME/.cache/Huawei/mxLaunchKit
    需要存放特定文件夹需指定 specific_path_config = {'path': 路径, 'rule': 权限范围}
    """
    if file_name is None or file_name == '':
        raise RuntimeError

    # 针对Linux环境, 在当前用户$HOME目录下指定文件夹存放文件, 同时约束文件夹访问权限
    if platform.system().lower() == 'linux':
        home_path = os.getenv('HOME')

        if home_path is None:
            raise RuntimeError

        full_path = os.path.join(home_path, '.cache', 'Huawei', 'mxLaunchKit')

        if not os.path.exists(full_path):
            os.makedirs(full_path, exist_ok=True)
            # mxLaunchKit文件夹权限 rwxr-x---
            os.chmod(full_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP)
    else:
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
        raise RuntimeError

    specific_path_rule = specific_path_config.get('rule')

    if specific_path_rule is None:
        raise RuntimeError

    return [specific_path, specific_path_rule]


class MAFileHandler(logging.FileHandler):
    def __init__(self, *args, **kwargs):
        """Rewrite __init__ so that the new file has permissions of 640."""
        super(MAFileHandler, self).__init__(*args, **kwargs)
        if not os.path.islink(self.baseFilename) and os.path.exists(self.baseFilename):
            os.chmod(self.baseFilename, 0o640)


def get_stream_handler(msg_pattern=DEFAULT_LOG_MSG_PATTERN,
                       time_pattern=DEFAULT_LOG_TIME_PATTERN):
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(msg_pattern, time_pattern)
    stream_handler.setFormatter(formatter)
    return stream_handler


def get_file_handler(msg_pattern=DEFAULT_LOG_MSG_PATTERN,
                     time_pattern=DEFAULT_LOG_TIME_PATTERN,
                     file_path=DEFAULT_LOCAL_LOG_FILE_DIR,
                     file_name=DEFAULT_LOCAL_LOG_FILE_NAME):
    if not os.path.exists(file_path):
        directory = wrap_local_working_directory('log')
        os.mkdir(directory, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP)
    file_handler = MAFileHandler(filename=os.path.join(file_path, file_name))
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(msg_pattern, time_pattern)
    file_handler.setFormatter(formatter)
    return file_handler


def get_logger(name="default",
               msg_pattern=DEFAULT_LOG_MSG_PATTERN,
               time_pattern=DEFAULT_LOG_TIME_PATTERN,
               file_path=DEFAULT_LOCAL_LOG_FILE_DIR,
               file_name=DEFAULT_LOCAL_LOG_FILE_NAME,
               with_file_handler=True):
    logger = logging.getLogger(name)
    logger.propagate = False
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    stream_handler = get_stream_handler(msg_pattern, time_pattern)
    logger.addHandler(stream_handler)

    if with_file_handler:
        file_handler = get_file_handler(msg_pattern, time_pattern, file_path, file_name)
        logger.addHandler(file_handler)

    return logger
