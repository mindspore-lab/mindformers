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
import re

from fm.src.utils import is_legal_node_num, read_file_with_link_check
from fm.src.aicc_tools.ailog.log import service_logger, service_logger_without_std
from fm.src.utils.args_check import is_legal_args, clean_space_and_quotes, path_check, \
    LegalArgsCheckParam
from fm.src.utils.callback_utils import prepare_scenario, app_config_arg_check, app_config_info_check
from fm.src.utils.local_cache_utils import path_exist_check
from fm.src.utils.io_utils import DEFAULT_FLAGS, DEFAULT_MODES

FRAMEWORKS = ['mindspore']
DEVICE_TYPES = ['npu']
SUPPORT_RESUME_TYPE = [True, False]
MODEL_VERSION_PATTERN = r"^(\d|[1-9]\d).(\d|[1-9]\d).(\d|[1-9]\d)$"
AVAILABLE_DEVICE_NUMS = [1, 2, 4, 8]


def cached_app_callback(ctx, param, value):
    return app_callback(ctx, param, value, virtual_cache=False)


def app_callback(ctx, param, value, virtual_cache=True):
    if value is None:
        return value

    app_config_path = clean_space_and_quotes(value)

    # app_config参数内容入参校验
    app_config_arg_check(param, app_config_path)
    service_logger_without_std.info('param: %s legality check success.', param.name)

    # 获取默认scenario参数, 优先读取--scenario指定值, 未传值时读取本地default_scenario.yaml缓存内容
    default_scenario = prepare_scenario(ctx)

    # 检验app_config路径存在情况
    app_config_path = path_exist_check(default_scenario, param.name, app_config_path, cache_config=True,
                                       virtual_cache=virtual_cache, with_file=True)

    try:
        app_config_info = read_file_with_link_check(app_config_path, DEFAULT_FLAGS, DEFAULT_MODES)
    except Exception as ex:
        service_logger.error('content in app_config yaml is not legal yaml-like format, '
                             'check the file and rerun config command to initialize.')
        raise ex
    finally:
        if virtual_cache and os.path.exists(app_config_path):
            os.remove(app_config_path)

    status = app_config_info_check(app_config_info, default_scenario)

    if not status:
        service_logger.error('incorrect settings in app_config yaml file, check the setting.')
        raise ValueError

    return app_config_info.get('scenario').get(default_scenario)


def name_callback(ctx, param, value):
    if value is None:
        return value

    value = clean_space_and_quotes(value)

    is_legal_args(LegalArgsCheckParam(appendix=['_', '-'], arg_key=param.name, arg_val=value, mode='default',
                                      min_len_limit=0, max_len_limit=64, entry='param'))

    service_logger_without_std.info('param: %s legality check success.', param.name)

    return value


def model_version_callback(ctx, param, value):
    if value is None:
        return value
    if not re.match(MODEL_VERSION_PATTERN, str(value)):
        service_logger.error('illegal param: %s, check the setting.', param.name)
        raise RuntimeError
    service_logger_without_std.info('param: %s legality check success.', param.name)
    return value


def cert_path_callback(ctx, param, value):
    if value is None:
        return value
    if os.path.islink(value) or not path_check(value, s3_path=False):
        service_logger.error('Illegel path param')
        raise ValueError
    if not os.path.exists(value):
        service_logger.error('CA cert does not exist')
        raise RuntimeError
    if not os.path.isfile(value):
        service_logger.error('path is not a file')
        raise ValueError
    return value


def data_path_callback(ctx, param, value):
    return obs_path_callback(ctx, param, value, s3_path=False, with_file=False, param_nec=False)


def obs_path_callback(ctx, param, value, s3_path=True, with_file=False, param_nec=False):
    if value is None:
        if param_nec:
            service_logger.error('param: %s is necessary.', param.name)
            raise ValueError()
        return value

    value = clean_space_and_quotes(value)

    is_legal_args(LegalArgsCheckParam(appendix=[':', '/', '-', '_', ' '], arg_key=param.name,
                                      arg_val=value, mode='default', entry='param'))

    if not path_check(value, s3_path=s3_path):
        service_logger.error('illegal param: %s, check the setting.', param.name)
        raise RuntimeError

    # 针对OBS来源进行远端路径真实性校验
    if value.startswith('obs://') or value.startswith('s3://'):
        default_scenario = prepare_scenario(ctx)
        path_exist_check(default_scenario, param.name, value, cache_config=False, with_file=with_file)

    service_logger_without_std.info('param: %s legality check success.', param.name)

    return value


def node_num_callback(ctx, param, value):
    if value is None:
        return value

    if not is_legal_node_num(value):
        service_logger.error('illegal param: %s, check the setting.', param.name)
        raise RuntimeError

    service_logger_without_std.info('param: %s legality check success.', param.name)

    return value


def device_num_callback(ctx, param, value):
    if value is None:
        return value

    if value <= 0:
        service_logger.error('illegal param: %s, check the setting.', param.name)
        raise RuntimeError

    if value not in AVAILABLE_DEVICE_NUMS:
        service_logger.error('illegal param: %s, only support 1/2/4/8.', param.name)
        raise RuntimeError

    service_logger_without_std.info('param: %s legality check success.', param.name)

    return value


def backend_callback(ctx, param, value):
    if value is None:
        return value

    value = clean_space_and_quotes(value)

    if value not in FRAMEWORKS:
        service_logger.error('illegal param: %s, check the setting.', param.name)
        raise RuntimeError

    service_logger_without_std.info('param: %s legality check success.', param.name)

    return value


def boolean_option_callback(ctx, param, value):
    if value is None:
        return value

    if value not in SUPPORT_RESUME_TYPE:
        service_logger.error('illegal param: %s, check the setting.', param.name)
        raise RuntimeError

    service_logger_without_std.info('param: %s legality check success.', param.name)

    return value


def obs_path_with_file_callback(ctx, param, value, value_required=False):
    if value is None:
        if value_required:
            service_logger.error('param: %s is necessary.', param.name)
            raise ValueError()
        return value

    value = clean_space_and_quotes(value)

    is_legal_args(LegalArgsCheckParam(appendix=[':', '/', '-', '_', '.', ' '], arg_key=param.name,
                                      arg_val=value, mode='default', entry='param'))

    if not path_check(value, s3_path=True):
        service_logger.error('illegal param: %s, check the setting.', param.name)
        raise RuntimeError

    default_scenario = prepare_scenario(ctx)
    path_exist_check(default_scenario, param.name, value, cache_config=False, with_file=True)

    service_logger_without_std.info('param: %s legality check success.', param.name)

    return value


def id_callback(ctx, param, value):
    if value is None:
        return value

    value = clean_space_and_quotes(value)
    is_legal_args(LegalArgsCheckParam(appendix=['_', '-'], arg_key=param.name, arg_val=value, mode='default',
                                      min_len_limit=0, max_len_limit=64, entry='param'))

    service_logger_without_std.info('param: %s legality check success.', param.name)

    return value


def instance_type_callback(ctx, param, value):
    if value is None:
        return value

    if value in ["job", "model", "service"]:
        service_logger_without_std.info('param: %s legality check success.', param.name)
    else:
        service_logger.error('illegal param: %s, check the setting.', param.name)
        raise RuntimeError

    return value


def instance_num_callback(ctx, param, value):
    if value is None:
        return value

    if value <= 0:
        service_logger.error('illegal param: %s, check the setting.', param.name)
        raise RuntimeError

    service_logger_without_std.info('param: %s legality check success.', param.name)

    return value


def ckpt_path_callback(ctx, param, value):
    return obs_path_callback(ctx, param, value, s3_path=True, with_file=False, param_nec=False)
