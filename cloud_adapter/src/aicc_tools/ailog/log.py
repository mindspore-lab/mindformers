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
import logging
import logging.config
import logging.handlers
import os
import sys
from typing import Dict, List, Tuple, Union

from fm.src.aicc_tools.utils import check_in_modelarts, get_num_nodes_devices, get_rank_info
from fm.src.aicc_tools.ailog.ailog_utils import (check_list, const, convert_nodes_devices_input, create_dirs,
                                                 generate_rank_list, log_args_black_list_characters_replace)

logger_list = {}
stream_handler_list = {}
file_handler_list = {}
LOG_RECORD_MAX_LEN = 2048
MAX_FILE_SIZE = 10
MAX_FILE_NUMS = 10
LOGGER_SOURCE_LIST = ('sdk', 'cli')


def judge_stdout(rank_id: int,
                 rank_size: int,
                 is_output: bool,
                 nodes: Union[List, Tuple, Dict[str, int], None] = None,
                 devices: Union[List, Tuple, Dict[str, int], None] = None) -> bool:
    """Determines if logs will be output.

    Args:
        rank_id (int): Rank id.
        rank_size (int): Rank size.
        is_output (int): If set to true, logs or others will be output.
        nodes (list or tuple or dict or None): Node list. The nodes in the list
            will output the log to stdout.
        devices (list or tuple or dict or None): Device list. The devices
            in the list or output the log to stdout.

    Returns:
        is_output (bool): If set to true, logs or others will be output
            or redirect.
    """
    if is_output and rank_size > 1 and (nodes is not None or devices is not None):
        num_nodes, num_devices = get_num_nodes_devices(rank_size)
        stdout_nodes = convert_nodes_devices_input(nodes, num_nodes)
        stdout_devices = convert_nodes_devices_input(devices, num_devices)
        check_list('nodes', stdout_nodes, num_nodes)
        check_list('devices', stdout_devices, num_devices)
        rank_list = generate_rank_list(stdout_nodes, stdout_devices)
        if rank_id not in rank_list:
            is_output = False

    return is_output


def validate_nodes_devices_input(var_name: str, var):
    """Check the list of nodes or devices.

    Args:
        var_name (str): Variable name.
        var: The name of the variable to be checked.

    Returns:
        None
    """
    if not (var is None or isinstance(var, (list, tuple, dict))):
        raise TypeError(f'The value of {var_name} can be None or a value of type tuple, list, or dict.')
    if isinstance(var, (list, tuple)):
        for item in var:
            if not isinstance(item, int):
                raise TypeError('The elements of a variable of type list or ' 'tuple must be of type int.')


def validate_level(var_name: str, var):
    """Verify that the log level is correct.

    Args:
        var_name (str): Variable name.
        var: The name of variable to be checked.

    Returns:
        None
    """
    if not isinstance(var, str):
        raise TypeError(f'The format of {var_name} must be of type str.')
    if var not in const.level:
        raise ValueError(f'{var_name}={var} needs to be in {const.level}')


def validate_std_input_format(to_std: bool, stdout_nodes: Union[List, Tuple, None],
                              stdout_devices: Union[List, Tuple, None], stdout_level: str):
    """Validate the input about stdout of the get_logger function."""

    if not isinstance(to_std, bool):
        raise TypeError('The format of the to_std must be of type bool.')

    validate_nodes_devices_input('stdout_nodes', stdout_nodes)
    validate_nodes_devices_input('stdout_devices', stdout_devices)
    validate_level('stdout_level', stdout_level)


def validate_file_input_format(file_level: Union[List, Tuple], file_save_dir: str, append_rank_dir: bool,
                               file_name: Union[List, Tuple]):
    """Validate the input about file of the get_logger function."""

    if not (isinstance(file_level, tuple) or isinstance(file_level, list)):
        raise TypeError('The value of file_level should be list or a tuple.')
    for level in file_level:
        validate_level('level in file_level', level)

    if not len(file_level) == len(file_name):
        raise ValueError('The length of file_level and file_name should be equal.')

    if not isinstance(file_save_dir, str):
        raise TypeError('The value of file_save_dir should be a value of type str.')

    if not isinstance(append_rank_dir, bool):
        raise TypeError('The value of append_rank_dir should be a value of type bool.')

    if not (isinstance(file_name, tuple) or isinstance(file_name, list)):
        raise TypeError('The value of file_name should be list or a tuple.')
    for name in file_name:
        if not isinstance(name, str):
            raise TypeError('The value of name in file_name should be a value of type str.')
        if name.startswith('/'):
            raise ValueError('The file name cannot start with "/".')


def _convert_level(level: str) -> int:
    """Convert the format of the log to logging level.

    Args:
        level (str): User log level.

    Returns:
        level (str): Logging level.
    """
    level_convert = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    logging_level = level_convert.get(level, logging.INFO)

    return logging_level


def get_stream_handler(stdout_format: str, stdout_level: str):
    """Set stream handler of logger."""
    if not stdout_format:
        stdout_format = const.default_stdout_format
    handler_name = f'{stdout_format}.{stdout_level}'
    if handler_name in stream_handler_list:
        return stream_handler_list.get(handler_name)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(_convert_level(stdout_level))
    stream_formatter = logging.Formatter(stdout_format)
    stream_handler.setFormatter(stream_formatter)

    stream_handler_list[handler_name] = stream_handler

    return stream_handler


def get_file_path_list(base_save_dir: str,
                       append_rank_dir: bool,
                       rank_id: int,
                       file_name: Union[Tuple, List],
                       mode: int = 0o750) -> List:
    """Gets the list of files where the logs are saved."""
    if not base_save_dir:
        base_save_dir = os.path.expanduser(const.local_default_log_file_dir)
    if check_in_modelarts():
        base_save_dir = const.modelarts_log_file_dir
    create_dirs(base_save_dir, mode=mode)

    file_save_dir = base_save_dir
    if append_rank_dir:
        rank_str = const.rank_dir_formatter.format(rank_id)
        file_save_dir = os.path.join(base_save_dir, rank_str)

    file_path = []
    for name in file_name:
        path = os.path.join(file_save_dir, name)
        path = os.path.realpath(path)
        base_dir = os.path.dirname(path)
        create_dirs(base_dir)
        file_path.append(path)

    for root, dirs, _ in os.walk(base_save_dir):
        for dir_name in dirs:
            path = os.path.join(root, dir_name)
            if not os.path.islink(path):
                os.chmod(path, mode)

    return file_path


class RotatingFileHandlerWithFilePermissionControl(logging.handlers.RotatingFileHandler):

    def __init__(self, *args, **kwargs):
        """Rewrite __init__ so that the new file has permissions of 640."""
        super(RotatingFileHandlerWithFilePermissionControl, self).__init__(*args, **kwargs)
        """restrict the length of log content"""
        if not os.path.islink(self.baseFilename) and os.path.exists(self.baseFilename):
            os.chmod(self.baseFilename, 0o640)

    def doRollover(self) -> None:
        logging.handlers.RotatingFileHandler.doRollover(self)
        if not os.path.islink(self.baseFilename) and os.path.exists(self.baseFilename):
            os.chmod(self.baseFilename, 0o640)

        if self.backupCount > 0:
            for i in range(self.backupCount - 1, 0, -1):
                fn = self.rotation_filename('%s.%d' % (self.baseFilename, i))
                if os.path.exists(fn) and not os.path.islink(fn):
                    os.chmod(fn, 0o440)

    def emit(self, record) -> None:
        tmp_out = record.getMessage()
        if tmp_out is not None and len(tmp_out) > LOG_RECORD_MAX_LEN:
            record.msg = tmp_out[:LOG_RECORD_MAX_LEN]
            record.args = ()
        super().emit(record)


def get_file_handler_list(file_level: Union[List, Tuple], file_path: Union[List, Tuple], max_file_size: int,
                          max_num_of_files: int) -> List:
    """get file handler of logger."""
    logging_level = []
    for level in file_level:
        logging_level.append(_convert_level(level))

    max_file_size = max_file_size * 1024 * 1024

    file_formatter = logging.Formatter(const.default_filehandler_format)

    file_handlers = []
    for path, level in zip(file_path, logging_level):
        handler_name = f'{path}.{max_file_size}.{max_num_of_files}.{level}'

        if handler_name in file_handler_list:
            file_handlers.append(file_handler_list.get(handler_name))
        else:
            file_handler = RotatingFileHandlerWithFilePermissionControl(filename=path,
                                                                        maxBytes=max_file_size,
                                                                        backupCount=max_num_of_files)
            file_handler.setLevel(level)
            file_handler.setFormatter(file_formatter)
            file_handlers.append(file_handler)

            file_handler_list[handler_name] = file_handler
    return file_handlers


class MxLogger(logging.Logger):
    def __init__(self, name):
        super(MxLogger, self).__init__(name)
        self.source = None
        self.method = None
        self.propagate = False

    def makeRecord(
            self,
            name,
            level,
            fn,
            lno,
            msg,
            args,
            exc_info,
            func=None,
            extra=None,
            sinfo=None,
    ):
        if extra is None:
            extra = dict()
        extra['source'] = self.source
        args = log_args_black_list_characters_replace(args)
        return super(MxLogger, self).makeRecord(name, level=level, fn=fn, lno=lno, msg=msg, args=args,
                                                exc_info=exc_info,
                                                func=func, extra=extra, sinfo=sinfo)

    def error_skip_check(self, msg):
        super(MxLogger, self).error(msg=msg, extra={'skip': True})


def set_logger_source(source):
    for logger_name in logger_list:
        if source in LOGGER_SOURCE_LIST:
            logger = logger_list.get(logger_name)
            logger.source = source


def get_logger(logger_name: str = 'aicc', **kwargs) -> logging.Logger:
    """Get the logger. Both computing centers and bare metal servers are
    available.

    Args:
        logger_name (str): Logger name.
        kwargs (dict): Other input.
            to_std (bool): If set to True, output the log to stdout.
            stdout_nodes (list[int] or tuple[int] or optional):
                The computation nodes that will output the log to stdout.
                default: None, indicates that all nodes will output logs to stdout.
                eg: [0, 1, 2, 3] or (0, 1, 2, 3): indicates that nodes 0, 1, 2, and
                    3 all output logs to stdout.
            stdout_devices (list[int] or tuple[int] or optional):
                The computation devices that will output the log to stdout.
                default: None, indicates that all devices will output logs to stdout.
                eg: [0, 1, 2, 3] or (0, 1, 2, 3): indicates that devices 0, 1, 2,
                    and 3 all output logs to stdout.
            stdout_level (str): The level of the log output to stdout.
                If the type is str, the options are DEBUG, INFO, WARNING, ERROR, CRITICAL.
            stdout_format (str): Log format.
            file_level (list[str] or tuple[str]): The level of the log output to file.
                eg: ['INFO', 'ERROR'] Indicates that the logger will output logs above
                    the level INFO and ERROR in the list to the corresponding file.
                The length of the list needs to be the same as the length of file_name.
            file_save_dir (str): The folder where the log files are stored.
            append_rank_dir (bool): Whether to add a folder with the format rank{}.
            file_name (list[str] or list[tuple]): Store a list of output file names.
            max_file_size (int): The maximum size of a single log file. Unit: MB.
            max_num_of_files (int): The maximum number of files to save.

    Returns:
        logger (logging.Logger): Logger.
    """
    if logger_name in logger_list:
        return logger_list.get(logger_name)

    logger = MxLogger(name=logger_name)
    to_std = kwargs.get('to_std', True)
    stdout_nodes = kwargs.get('stdout_nodes', None)
    stdout_devices = kwargs.get('stdout_devices', (0,))
    stdout_level = kwargs.get('stdout_level', 'INFO')
    stdout_format = kwargs.get('stdout_format', '')
    file_level = kwargs.get('file_level', ('INFO', 'ERROR'))
    file_save_dir = kwargs.get('file_save_dir', '')
    append_rank_dir = kwargs.get('append_rank_dir', True)
    file_name = kwargs.get('file_name', ('aicc.log', 'error.log'))
    max_file_size = kwargs.get('max_file_size', MAX_FILE_SIZE)
    max_num_of_files = kwargs.get('max_num_of_files', MAX_FILE_NUMS)

    validate_std_input_format(to_std, stdout_nodes, stdout_devices, stdout_level)
    validate_file_input_format(file_level, file_save_dir, append_rank_dir, file_name)

    rank_id, rank_size = get_rank_info()

    to_std = judge_stdout(rank_id, rank_size, to_std, stdout_nodes, stdout_devices)
    if to_std:
        stream_handler = get_stream_handler(stdout_format, stdout_level)
        logger.addHandler(stream_handler)
    file_path = get_file_path_list(file_save_dir, append_rank_dir, rank_id, file_name)
    file_handlers = get_file_handler_list(file_level, file_path, max_file_size, max_num_of_files)
    for file_handler in file_handlers:
        logger.addHandler(file_handler)

    logger.propagate = False
    logger.setLevel(_convert_level('DEBUG'))

    logger_list[logger_name] = logger

    return logger


aicc_logger = get_logger('aicc', file_name=['aicc.INFO.log', 'aicc.ERROR.log'], file_level=['INFO', 'ERROR'])
service_logger = get_logger('service', to_std=True, file_name=['service.log'], file_level=['INFO'],
                            append_rank_dir=False)
service_logger_without_std = get_logger('service_logger_without_std', to_std=False, file_name=['service.log'],
                                        file_level=['INFO'], append_rank_dir=False)
operation_logger = get_logger('operation', to_std=True, file_name=['operation.log'], file_level=['INFO'],
                              append_rank_dir=False)
operation_logger_without_std = get_logger('operation_logger_without_std', to_std=False, file_name=['operation.log'],
                                          file_level=['INFO'], append_rank_dir=False)
