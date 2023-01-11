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
import uuid
from importlib import import_module

from fm.aicc_tools.ailog.log import service_logger, service_logger_without_std
from fm.aicc_tools.utils import check_in_modelarts
from fm.utils import constants, obs_connection_check, extract_ak_sk_endpoint_token_from_cert
from fm.utils.cert_utils import decrypt_cert
from fm.utils.io_utils import DEFAULT_FLAGS, DEFAULT_MODES, \
    wrap_local_working_directory, read_file_with_link_check, get_config_dir_setting
from fm.utils.obs_tool import download_from_obs, check_obs_path


def path_exist_check(scenario, param_name, obs_path, cache_config=False, virtual_cache=False, with_file=False):
    if scenario == 'local':
        path_exist_check_for_local(param_name, obs_path)
    elif scenario == 'modelarts':
        obs_path = path_exist_check_for_modelarts(cache_config, param_name, obs_path, virtual_cache, with_file)
    else:
        service_logger.error('illegal scenario: %s, check the setting.' % scenario)
        raise RuntimeError

    return obs_path


def path_exist_check_for_modelarts(cache_config, param_name, obs_path, virtual_cache, with_file):
    """
    cache_config: 本地缓存路径
    param_name: 参数名称，model_config_path/app_config/..
    value： 参数值
    virtual_cache: 非config命令之外不需要更新缓存
    with_file：文件-True或者文件夹-False
    """
    # 针对路径配置值要求以文件结尾, 需要保障整个配置路径不是以/结尾, /结尾表示文件夹
    if with_file and obs_path.endswith('/'):
        service_logger.error('param %s should be a file path, please check the setting.' % param_name)
        raise FileNotFoundError

    # 针对文件夹路径判断时，末尾添加 /
    if not with_file and not obs_path.endswith('/'):
        obs_path += '/'

    if check_in_modelarts():
        obs_path = path_exist_check_with_moxing(cache_config, param_name, obs_path, virtual_cache)
    else:
        obs_path = path_exist_check_with_obs_client(cache_config, param_name, obs_path, virtual_cache)

    return obs_path


def path_exist_check_with_moxing(cache_config, param_name, obs_path, virtual_cache):

    try:
        mox = import_module('moxing')
        mox.file.set_auth(retry=2)
        is_exists = mox.file.exists(obs_path)
    except mox.file.MoxFileNotExistsException as mox_ex:
        service_logger.error('the obs bucket is wrong, check the file path settings.')
        raise mox_ex

    if is_exists and cache_config:
        if not virtual_cache:
            obs_path = download_config_file_from_obs(None, param_name, obs_path)
        else:
            obs_path = download_tmp_config_file_from_obs(None, param_name, obs_path)
    return obs_path


def path_exist_check_with_obs_client(cache_config, param_name, value, virtual_cache):
    cached_cert_package = get_cached_cert(flatten_to_list=True)  # 从本地拿取认证信息
    cert_package = extract_ak_sk_endpoint_token_from_cert(cached_cert_package)
    del cached_cert_package

    # 检验用户提供的obs文件路径是否存在
    check_obs_path(cert_package, param_name, value)

    if cache_config:
        # download and refresh local cache
        if not virtual_cache:
            value = download_config_file_from_obs(cert_package, param_name, value)
        else:
            # download temp local file
            value = download_tmp_config_file_from_obs(cert_package, param_name, value)
        service_logger_without_std.info('scenario mode: modelarts, download app config file from obs success.')
    del cert_package

    return value


def path_exist_check_for_local(param_name, value):
    if not os.path.exists(value):
        service_logger.error('%s: %s is not found.', param_name, value)
        raise FileNotFoundError


def get_cached_cert(flatten_to_list=False):
    """
    根据本地缓存获取AK、SK
    """
    local_cert_cache_file_path = wrap_local_working_directory(file_name=constants.CERT_FILE_LOCAL_PATH,
                                                              specific_path_config=get_config_dir_setting())

    if not os.path.exists(local_cert_cache_file_path):
        service_logger.error(
            'no local registry cache file found, use registry command to init local registry cache.')
        raise RuntimeError

    cert_info_cipher = read_file_with_link_check(local_cert_cache_file_path, DEFAULT_FLAGS, DEFAULT_MODES)

    cert_info_plain = decrypt_cert(cert_info_cipher, flatten_to_list=flatten_to_list)

    if 'scenario' not in cert_info_plain:
        service_logger.error('key scenario not found in local registry cache file.')
        del cert_info_plain
        raise RuntimeError

    cert_info = cert_info_plain.get('scenario')

    # 校验认证信息是否可以成功连接至obs
    obs_connection_check(cert_info)

    del cert_info_plain

    return cert_info


def download_config_file_from_obs(cert_package, param_name, obs_path):
    local_cache_file_name = wrap_local_working_directory(file_name=constants.APP_CONFIG_LOCAL_PATH,
                                                         specific_path_config=get_config_dir_setting())

    download_from_obs(local_cache_file_name, cert_package, param_name, obs_path)

    return local_cache_file_name


def download_tmp_config_file_from_obs(cert_package, param_name, obs_path):
    local_cache_file_name = wrap_local_working_directory(file_name=generate_tmp_config_file_name_with_uuid(),
                                                         specific_path_config=get_config_dir_setting())

    download_from_obs(local_cache_file_name, cert_package, param_name, obs_path)

    return local_cache_file_name


def generate_tmp_config_file_name_with_uuid():
    return constants.APP_CONFIG_LOCAL_PATH.strip('.yaml') + '_' + str(uuid.uuid4()).replace('-', '_') + '.yaml'
