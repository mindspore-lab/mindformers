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

from fm.src.aicc_tools.ailog.log import service_logger_without_std, service_logger
from fm.src.utils import constants
from fm.src.utils.args_check import is_legal_args, LegalArgsCheckParam, path_check, \
    LegalLocalPathCheckParam, clean_space_and_quotes, \
    url_content_black_list_characters_check, app_config_content_length_check
from fm.src.utils.default_scenario_utils import get_default_scenario_from_local
from fm.src.utils.local_cache_utils import path_exist_check
from fm.src.utils.io_utils import get_config_dir_setting, wrap_local_working_directory
from fm.src.utils.yaml_property_check import is_legal_device_num, is_legal_node_num, is_legal_ip_format


def prepare_scenario(ctx):
    if 'scenario' not in ctx.params:
        service_logger_without_std.warning(
            'can not find scenario in param, try to find it in local default_scenario cache file.')

        default_scenario_cache_file_path = wrap_local_working_directory(file_name=constants.DEFAULT_SCENARIO_LOCAL_PATH,
                                                                        specific_path_config=get_config_dir_setting())

        if not os.path.exists(default_scenario_cache_file_path):
            service_logger.error("parameter scenario is necessary, please specify it.")
            raise RuntimeError
        else:
            default_scenario = get_default_scenario_from_local()
    else:
        default_scenario = ctx.params.get('scenario')

    return default_scenario


def app_config_arg_check(param, value):
    is_legal_args(LegalArgsCheckParam(appendix=[':', '/', '-', '_', '.', ' '], arg_key=param.name, arg_val=value,
                                      mode='default', entry='param'))

    if not path_check(value, s3_path=True):
        service_logger.error('illegal param: %s, check the setting.', param.name)
        raise RuntimeError


def app_config_info_check(app_config, scenario):
    if 'scenario' not in app_config.keys():
        service_logger.error('scenario should in app config file')
        return False

    if scenario not in app_config.get('scenario'):
        service_logger.error('%s should in app_config.scenario', scenario)
        return False

    if scenario == 'modelarts':
        if not modelarts_scenario_check(app_config, scenario):
            return False

    service_logger_without_std.info('app_config yaml content legality check success.')

    return True


def modelarts_scenario_check(app_config, scenario):
    if not necessary_keys_check_item(app_config, scenario):
        return False

    if not computing_center_configs_check_item(app_config, scenario):
        return False

    if not operation_configs_check_item(app_config, scenario):
        return False

    if not computing_resource_configs_check_item(app_config, scenario):
        return False

    if not nas_configs_check_item(app_config, scenario):
        return False

    return True


def nas_configs_check_item(app_config, scenario):
    if not nas_share_addr_check_item(app_config, scenario):
        return False

    if not local_path_check_item(
            LegalLocalPathCheckParam(app_config=app_config, scenario=scenario, search_key='nas_mount_path',
                                     min_len_limit=1, max_len_limit=256)):
        return False

    return True


def computing_resource_configs_check_item(app_config, scenario):
    config = app_config.get('scenario').get(scenario)
    if not pool_id_check_item(config):
        return False
    if config.get("deployment") and not pool_id_check_item(config.get("deployment")):
        return False
    if not node_num_check_item(config):
        return False
    if config.get("deployment") and not node_num_check_item(config.get("deployment")):
        return False
    if not device_num_check_item(config):
        return False
    if config.get("deployment") and not device_num_check_item(config.get("deployment")):
        return False
    return True


def operation_configs_check_item(app_config, scenario):
    if not obs_path_check_item(app_config, scenario, 'data_path', s3_path=False):
        return False
    if not obs_path_check_item(app_config, scenario, 'output_path'):
        return False
    if not obs_path_check_item(app_config, scenario, 'code_url'):
        return False
    if not obs_path_check_item(app_config, scenario, 'model_config_path', with_file=True):
        return False
    if not obs_path_check_item(app_config, scenario, 'pretrained_model_path'):
        return False
    if not obs_path_check_item(app_config, scenario, 'ckpt_path'):
        return False

    if app_config.get('scenario').get(scenario).get('boot_file_path'):
        boot_file_path = str(app_config.get('scenario').get(scenario).get('boot_file_path'))
    else:
        service_logger.error('boot_file_path is required in scenario %s, check the setting.', scenario)
        return False

    if not boot_file_path.startswith(app_config.get('scenario').get(scenario).get('code_url')):
        service_logger.error('boot_file_path must be under code_url in scenario %s, check the setting.', scenario)
        return False
    else:
        if not obs_path_check_item(app_config, scenario, 'boot_file_path', with_file=True):
            return False

    if not obs_path_check_item(app_config, scenario, 'log_path'):
        return False
    if not user_image_url_check_item(app_config, scenario):
        return False
    return True


def computing_center_configs_check_item(app_config, scenario):
    if not endpoint_check_item(app_config, scenario, 'iam_endpoint'):
        return False
    if not endpoint_check_item(app_config, scenario, 'obs_endpoint'):
        return False
    if not endpoint_check_item(app_config, scenario, 'modelarts_endpoint'):
        return False
    if not endpoint_check_item(app_config, scenario, 'swr_endpoint'):
        return False
    if not region_name_check_item(app_config, scenario):
        return False
    if not project_id_check_item(app_config, scenario):
        return False
    return True


def necessary_keys_check_item(app_config, scenario):
    necessary_keys = [
        'iam_endpoint',
        'obs_endpoint',
        'modelarts_endpoint',
        'region_name',
        'project_id',
        'data_path',
        'output_path',
        'code_url',
        'boot_file_path',
        'log_path',
        'user_image_url'
    ]

    for necessary_key in necessary_keys:
        if necessary_key not in app_config.get('scenario').get(scenario):
            service_logger.error('%s should in app_config.scenario.%s', necessary_key, scenario)
            return False

    return True


def endpoint_check_item(app_config, scenario, search_key):
    if search_key in app_config.get('scenario').get(scenario):
        value = app_config.get('scenario').get(scenario).get(search_key)

        if not has_valid_parameter_setting(search_key, value):
            return False

        value = clean_space_and_quotes(value)

        try:
            is_legal_args(LegalArgsCheckParam(appendix=[':', '/', '-', '_', '.'], arg_key=search_key,
                                              arg_val=value, mode='default', entry='config'))
        except Exception:
            return False

        if not url_content_black_list_characters_check(value):
            service_logger.error('param %s in app config file has illegal value, check the setting.', search_key)
            return False

    return True


def region_name_check_item(app_config, scenario):
    if 'region_name' in app_config.get('scenario').get(scenario):
        value = app_config.get('scenario').get(scenario).get('region_name')

        if not has_valid_parameter_setting('region_name', value):
            return False

        value = clean_space_and_quotes(value)

        try:
            is_legal_args(LegalArgsCheckParam(appendix=['-', '_'], arg_key='region_name', arg_val=value, mode='default',
                                              entry='config'))
        except Exception:
            return False

    return True


def project_id_check_item(app_config, scenario):
    if 'project_id' in app_config.get('scenario').get(scenario):
        value = app_config.get('scenario').get(scenario).get('project_id')

        if not has_valid_parameter_setting('project_id', value):
            return False

        value = clean_space_and_quotes(value)

        try:
            is_legal_args(LegalArgsCheckParam(appendix=['-', '_'], arg_key='project_id', arg_val=value, mode='default',
                                              entry='config'))
        except Exception:
            return False

    return True


def obs_path_check_item(app_config, scenario, search_key, s3_path=True, with_file=False):
    if search_key in app_config.get('scenario').get(scenario):
        value = app_config.get('scenario').get(scenario).get(search_key)

        if not has_valid_parameter_setting(search_key, value):
            return False

        value = clean_space_and_quotes(value)

        appendix = [':', '/', '-', '_', ' ']
        appendix = appendix + ['.'] if with_file else appendix
        try:
            is_legal_args(LegalArgsCheckParam(appendix=appendix, arg_key=search_key, arg_val=value, mode='default',
                                              entry='config'))
        except Exception:
            return False

        if not path_check(value, s3_path=s3_path):
            service_logger.error('illegal param: %s from app config file, check the setting.', search_key)
            return False

        # 针对OBS来源进行远端路径真实性校验
        if value.startswith('obs://') or value.startswith('s3://'):
            try:
                path_exist_check(scenario, search_key, value, cache_config=False, with_file=with_file)
            except Exception:
                return False

    return True


def local_path_check_item(check_param):
    if check_param.search_key in check_param.app_config.get('scenario').get(check_param.scenario):
        value = check_param.app_config.get('scenario').get(check_param.scenario).get(check_param.search_key)

        if not has_valid_parameter_setting(check_param.search_key, value):
            return False

        value = clean_space_and_quotes(value)

        try:
            if check_param.contains_file:
                is_legal_args(LegalArgsCheckParam(appendix=['/', '-', '_', '.', ' '], arg_key=check_param.search_key,
                                                  arg_val=value, mode='default', entry='config'))
            else:
                is_legal_args(LegalArgsCheckParam(appendix=['/', '-', '_', ' '], arg_key=check_param.search_key,
                                                  arg_val=value, mode='default', entry='config'))
        except Exception:
            return False

        if not path_check(value, s3_path=False):
            service_logger.error(
                'illegal param: %s from app config file, check the setting.', check_param.search_key)
            return False

        if not app_config_content_length_check(check_param, value):
            return False

    return True


def user_image_url_check_item(app_config, scenario):
    if 'user_image_url' in app_config.get('scenario').get(scenario):
        value = app_config.get('scenario').get(scenario).get('user_image_url')

        if not has_valid_parameter_setting('user_image_url', value):
            return False

        value = clean_space_and_quotes(value)

        try:
            is_legal_args(
                LegalArgsCheckParam(appendix=[':', '/', '-', '_', '.'], arg_key='user_image_url', arg_val=value,
                                    mode='default', entry='config'))
        except Exception:
            return False

        if not url_content_black_list_characters_check(value):
            service_logger.error('illegal character in config: user_image_url, check the setting.')
            return False

    return True


def pool_id_check_item(config):
    if 'pool_id' in config:
        value = config.get('pool_id')

        if not has_valid_parameter_setting('pool_id', value):
            return False

        value = clean_space_and_quotes(value)

        try:
            is_legal_args(LegalArgsCheckParam(appendix=['_', '-'], arg_key='pool_id', arg_val=value, mode='default',
                                              min_len_limit=1, max_len_limit=256, entry='config'))
        except Exception:
            return False

    return True


def device_num_check_item(config):
    if 'device_num' in config:
        legal_device_num = is_legal_device_num(config.get('device_num'))
        if not legal_device_num:
            service_logger.error('illegal device_num, check the setting from app config file.')
            return False
    return True


def node_num_check_item(config):
    # node_num must be a positive integer
    if 'node_num' in config:
        legal_node_num = is_legal_node_num(config.get('node_num'))

        if not legal_node_num:
            service_logger.error('illegal node_num, check the setting from app config file.')
            return False

    return True


def nas_share_addr_check_item(app_config, scenario):
    # nas_share_addr must follow ip format, like a.b.c.d:/
    if 'nas_share_addr' in app_config.get('scenario').get(scenario):
        legal_ip = is_legal_ip_format('nas_share_addr', app_config.get('scenario').get(scenario).get('nas_share_addr'))

        if not legal_ip:
            service_logger.error('illegal nas_share_addr, check the setting from app config file.')
            return False

    return True


def has_valid_parameter_setting(param_name, param_value):
    if param_value is None:
        service_logger.error('parameter %s setting is empty, check the setting.', param_name)
        return False

    return True
