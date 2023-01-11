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

from fm.src.aicc_tools.utils import check_in_modelarts

from fm.src.adapter import strategy_register

import fm.src.aicc_tools as ac
from fm.src.aicc_tools.ailog.log import service_logger
from fm.src.utils.daemon import upload
from fm.src.utils.io_utils import DEFAULT_FLAGS, DEFAULT_MODES
from fm.src.utils import constants, wrap_local_working_directory, read_file_with_link_check, get_config_dir_setting, \
    get_cached_cert, set_obs_path


def prepare_configs(kwargs, flags, modes):
    """
    针对scenario, cert, app_config三部分参数, 优先取用户明文指定的配置, 如用户未配置, 取本地缓存
    """
    wrap_kwargs_with_local_config(kwargs, flags, modes)
    kwargs_empty_check(kwargs)

    return kwargs


def load_scenario_from_local(flags, modes):
    """
    加载本地缓存的default_scenario
    """
    scenario_cache_file_local_path = wrap_local_working_directory(file_name=constants.DEFAULT_SCENARIO_LOCAL_PATH,
                                                                  specific_path_config=get_config_dir_setting())

    if not os.path.exists(scenario_cache_file_local_path):
        service_logger.warning('scenario is not initialized, using specify inputs from param instead.')
        raise RuntimeError

    scenario_output = read_file_with_link_check(scenario_cache_file_local_path, flags, modes).get('default_scenario')

    return scenario_output


def load_config_from_local(flags, modes):
    """
    加载本地缓存的app_config
    """
    app_config_cache_local_path = wrap_local_working_directory(file_name=constants.APP_CONFIG_LOCAL_PATH,
                                                               specific_path_config=get_config_dir_setting())

    if not os.path.exists(app_config_cache_local_path):
        service_logger.warning('config is not initialized, using specify inputs from param instead.')
        raise RuntimeError

    try:
        app_config_output = read_file_with_link_check(app_config_cache_local_path, flags, modes)
    except Exception as ex:
        service_logger.error('content in app_config yaml is not legal yaml-like format, '
                             'check the file and rerun config command to initialize.')
        raise ex

    return app_config_output


def load_cert_from_local():
    """
    加载本地缓存的cert
    """
    return get_cached_cert(flatten_to_list=True)


def kwargs_empty_check(kwargs):
    """
    必要参数存在性检查
    """
    if not check_in_modelarts():
        if kwargs.get('cert') is None:
            service_logger.error('registry is not initialized, use registry command to initialize.')
            raise ValueError

    if kwargs.get('scenario') is None:
        service_logger.error(
            'scenario is not initialized, use config command or param to specify.')
        raise ValueError

    if kwargs.get('app_config') is None:
        service_logger.error(
            'app_config is not initialized, use config command or param to specify.')
        raise ValueError


def wrap_kwargs_with_local_config(kwargs, flags, modes):
    """
    针对用户未主动设置的参数, 使用本地缓存文件内容补齐
    """
    wrap_scenario(kwargs, flags, modes)
    wrap_app_config(kwargs, flags, modes)

    if check_in_modelarts():
        kwargs['cert'] = []
    else:
        wrap_cert(kwargs)


def wrap_cert(kwargs):
    try:
        cert = load_cert_from_local()
    except Exception:
        cert = None

    if cert is not None:
        kwargs['cert'] = cert
        del cert


def wrap_scenario(kwargs, flags, modes):
    if kwargs.get('scenario') is None:
        try:
            scenario = load_scenario_from_local(flags, modes)
        except Exception:
            scenario = None

        if scenario is not None:
            kwargs['scenario'] = scenario


def wrap_app_config(kwargs, flags, modes):
    if kwargs.get('app_config') is None:
        try:
            app_config = load_config_from_local(flags, modes)
        except Exception:
            app_config = None

        if app_config is not None:
            kwargs['app_config'] = app_config.get('scenario').get(kwargs.get('scenario'))


@ac.aicc_monitor
def run_strategy(*args, **kwargs):
    kwargs = prepare_configs(kwargs, DEFAULT_FLAGS, DEFAULT_MODES)

    # register aicc_tools
    register(kwargs)

    # upload log files automatically
    upload()

    strategy_runner = strategy_register.get(kwargs.get('scenario'))()
    kwargs.pop('scenario')

    output = getattr(strategy_runner, args[0])(**kwargs)

    return output


def register(kwargs):
    set_obs_path(kwargs.get("app_config"))


def commands_generator(header, kwargs):
    output = []
    for key, value in kwargs.items():
        output.append('--{}'.format(key))
        output.append('{}'.format(value))
    return [header] + output


# job, model, service
def which_instance_type(kwargs, is_show=False):
    count, instance_id, instance_type = 0, None, None

    if kwargs.get('job_id'):
        count, instance_id, instance_type = count + 1, kwargs.get('job_id'), "job"
    if kwargs.get('model_id'):
        count, instance_id, instance_type = count + 1, kwargs.get('model_id'), "model"
    if kwargs.get('service_id'):
        count, instance_id, instance_type = count + 1, kwargs.get('service_id'), "service"

    kwargs.pop('job_id')
    kwargs.pop('model_id')
    kwargs.pop('service_id')

    if count > 1:
        service_logger.error("Only accept id of one type.")
        raise ValueError("Only accept id of one type.")

    if not is_show and count == 0:
        service_logger.error("job_id/model_id/service_id is required.")
        raise ValueError("job_id/model_id/service_id is required.")

    if not is_show and not instance_id:
        service_logger.error("Require a valid id.")
        raise ValueError("Require a valid id.")

    return instance_id, instance_type, kwargs
