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
from importlib import import_module

from obs import ObsClient

from fm.aicc_tools.ailog.log import service_logger, service_logger_without_std
from fm.aicc_tools.utils.validator import check_in_modelarts
from fm.utils import constants
from fm.utils.args_check import is_legal_args, LegalArgsCheckParam
from fm.utils.io_utils import DEFAULT_FLAGS, DEFAULT_MODES, is_link, calculate_file_md5
from fm.utils.io_utils import wrap_local_working_directory, read_file_with_link_check, get_ca_dir_setting

OBS_SEP = '/'
OBS_PREFIX = ['obs://', 's3://']
RESP_STATUS_CODE = 300
MD5_KEYWORD_LIST = ('md5chksum', 'contentmd5', 'content_md5', 'content-md5')


def is_startswith_obs_prefix(path):
    if not isinstance(path, str):
        return False
    if not path.startswith(OBS_PREFIX[0]) and not path.startswith(OBS_PREFIX[1]):
        return False
    return True


def check_obs_path(cert_package, param_name, obs_path):
    """
        功能: 对指定registry信息新建obs连接并检查用户提供文件路径（包括远端存在与否校验）
        参数:
            cert_package:
            obs_path: 用户给定obs路径
        返回值:
            True or False
    """
    # 根据用户给定obs_path路径提取出有效信息
    try:
        bucket_name, object_key = extract_bucket_name_and_object_key(obs_path)
    except Exception as ex:
        service_logger.error('error occurred during accessing obs, check the obs path settings.')
        raise ex

    # 根据用户给定路径初始化obs客户端
    try:
        obs_client = new_obs_client(cert_package)
    except Exception as ex:
        service_logger.error('error occurred during accessing obs, check the registry settings.')
        raise ex
    finally:
        del cert_package

    # 根据用户给定桶名查找桶是否存在
    try:
        if obs_client.headBucket(bucket_name).status >= RESP_STATUS_CODE:
            raise RuntimeError
    except Exception as ex:
        if obs_client:
            obs_client.close()
        service_logger.error('error occurred during accessing obs bucket, please check obs path/registry '
                             'settings')
        raise ex

    # 检查指定obs文件是否存在
    if object_key is not None:
        try:
            if obs_client.getObjectMetadata(bucket_name, object_key).status >= RESP_STATUS_CODE:
                raise FileNotFoundError
        except Exception as ex:
            if param_name != 'pretrained_model_path':
                service_logger.error('%s: %s is not found.', param_name, obs_path)
            raise ex
        finally:
            if obs_client:
                obs_client.close()


def extract_bucket_name_and_object_key(obs_path):
    """
        功能: 提取用户传入的obs路径信息
        参数:
            obs_path: obs路径信息
        返回值:
            bucket_name: 桶名
            object_key: obs路径头与桶名/之后的字符串内容
    """
    if not obs_path:
        raise RuntimeError

    obs_path = str(obs_path)
    if not is_startswith_obs_prefix(obs_path):
        raise RuntimeError

    try:
        is_legal_args(LegalArgsCheckParam(
            appendix=[':', '/', '-', '_', '.'],
            arg_key='obs_path',
            arg_val=obs_path,
            mode='default',
            entry='param'))
    except Exception as ex:
        raise ex

    path_split_result = str(obs_path).split(OBS_SEP, 3)  # 拆分路径为['obs:', '', bucket_name, object_key]
    bucket_name = path_split_result[2]
    if len(path_split_result) < 3:
        raise FileNotFoundError
    elif len(path_split_result) == 3:
        object_key = None
    else:
        object_key = path_split_result[3]

    return bucket_name, object_key


def get_verify_info():
    """
    load enable CA cert and CA file path from yaml
    """
    verify_config_path = wrap_local_working_directory(file_name=constants.VERIFY_CONFIG_LOCAL_PATH,
                                                      specific_path_config=get_ca_dir_setting())
    if os.path.exists(verify_config_path):
        verify_info = read_file_with_link_check(verify_config_path, DEFAULT_FLAGS, DEFAULT_MODES)
    else:
        verify_info = {'enable': False}
    return verify_info


def set_ssl_verify(params):
    """
    if user set ca cert file , the connection between obs should use CA
    :param params
    :return: None
    """
    verify_info = get_verify_info()
    if verify_info.get('enable'):
        service_logger_without_std.info('ssl_verify is enable')
        ca_file_path = wrap_local_working_directory(file_name=constants.CA_FILE_LOCAL_PATH,
                                                    specific_path_config=get_ca_dir_setting())
        if os.path.exists(ca_file_path) and os.path.isfile(ca_file_path):
            params['ssl_verify'] = ca_file_path
            service_logger_without_std.info('set obs ssl_verify with ca file')
        else:
            service_logger.error('ca file path does not exist')
            raise RuntimeError
    service_logger_without_std.info('register to obs without ssl_verify')


def new_obs_client(cert):
    """
        功能: 获取 obs_client
        参数:
            cert: 鉴权信息元组(ak, sk, obs_endpoint, security_token)
        返回值:
            obs_client
    """
    params = dict()
    params['access_key_id'] = cert[0]
    params['secret_access_key'] = cert[1]
    params['server'] = cert[2]
    params['security_token'] = cert[3]
    set_ssl_verify(params)
    obs_client = ObsClient(**params)

    return obs_client


def get_file_md5_from_obs(obs_object):
    md5_origin = None
    header_list = obs_object.header
    for item in header_list:
        if str.lower(item[0]) in MD5_KEYWORD_LIST:
            md5_origin = item[1]
    return md5_origin


def validate_file_integrity(obs_object, download_path):
    """validate downloaded by obs file's integrity"""
    md5_origin = get_file_md5_from_obs(obs_object)
    if md5_origin is None:
        service_logger.info('file integrity is not set, download without validation')
        return
    md5_download = calculate_file_md5(download_path)
    if md5_download != md5_origin:
        service_logger.error('file integrity validate fail')
        raise RuntimeError()
    service_logger.info('file integrity validate success')


def validate_mox_file_integrity(mox, obs_path, download_path):
    """validate downloaded by moxing file's integrity"""
    obs_client = mox.file.file_io._create_or_get_obs_client()
    bucket_name, object_key = extract_bucket_name_and_object_key(obs_path)
    metadata = obs_client.getObjectMetadata(bucket_name, object_key)
    md5_origin = get_file_md5_from_obs(metadata)
    if md5_origin is None:
        service_logger.info('file integrity is not set, download without validation')
        return
    file_md5 = calculate_file_md5(download_path)
    if md5_origin != file_md5:
        service_logger.error('file downloaded by moxing validate integrity fail')
        raise RuntimeError
    service_logger.info('file downloaded by moxing validate integrity success')


def download_from_obs(local_cache_file_name, cert_package, param_name, obs_path):
    if os.path.exists(local_cache_file_name):
        os.remove(local_cache_file_name)

    if check_in_modelarts():
        download_with_moxing(local_cache_file_name, param_name, obs_path)
    else:
        download_with_obs_client(local_cache_file_name, cert_package, param_name, obs_path)


def download_with_obs_client(local_cache_file_name, cert_package, param_name, obs_path):
    obs_client = new_obs_client(cert_package)

    value_split = obs_path.split('/')
    bucket_name = value_split[2]
    object_key = '/'.join(value_split[3:])

    try:
        resp = obs_client.getObject(bucket_name, object_key)
    except Exception as ex:
        obs_client.close()
        service_logger.error('download or obs is wrong, please check the setting.')
        raise ex

    if resp.get('status') != 200:
        service_logger.error('%s: %s is not found.', param_name, obs_path)
        raise FileNotFoundError

    # OBS只传桶异常场景校验
    if resp.get('body').get('contentLength') is None:
        service_logger.error('detect obs path without folder/file info, please check the setting.')
        raise IOError

    # OBS空对象校验
    if resp.get('body').get('contentLength') == 0:
        service_logger.error('detect empty object, please check the setting.')
        raise IOError

    # OBS过长内容对象校验
    if resp.get('body').get('contentLength') > constants.APP_CONFIG_INFO_MAX_LENGTH_FROM_OBS:
        service_logger.error('detect too big object, please check the setting.')
        raise IOError

    if os.path.exists(local_cache_file_name) and is_link(local_cache_file_name):
        service_logger.error('detect link, illegal file path: {}'.format(local_cache_file_name))
        raise RuntimeError

    obs_object = obs_client.getObject(bucket_name, object_key, downloadPath='{}'.format(local_cache_file_name))
    os.chmod(local_cache_file_name, DEFAULT_MODES)

    validate_file_integrity(obs_object=obs_object, download_path=local_cache_file_name)


def download_with_moxing(local_cache_file_name, param_name, obs_path):
    try:
        mox = import_module('moxing')
        mox.file.stat(obs_path)
    except mox.file.MoxFileNotExistsException as mox_ex:
        service_logger.error('error occurred during get file with moxing, %s: %s is not found.', param_name, obs_path)
        raise mox_ex

    size = mox.file.get_size(obs_path)
    if size < 0:
        service_logger.error('detect obs path without folder/file info, please check the setting.')
    if size == 0:
        service_logger.error('detect empty object, please check the setting.')
        raise IOError
    if size > constants.APP_CONFIG_INFO_MAX_LENGTH_FROM_OBS:
        service_logger.error('detect too big object, please check the setting.')
        raise IOError

    mox.file.copy(obs_path, local_cache_file_name)
    os.chmod(local_cache_file_name, DEFAULT_MODES)

    validate_mox_file_integrity(mox, obs_path, local_cache_file_name)


def obs_connection_check(cert_info):
    """
        功能: 对给定的明文cert_info检验是否能成功连接至obs
        参数:
            cert_info = [type, ak, sk, endpoint, encryption] 列表形式的cert认证信息，元素为字符串
        返回值:
            None
    """
    cert_package = extract_ak_sk_endpoint_token_from_cert(cert_info)
    try:
        obs_client = new_obs_client(cert_package)
    except Exception as ex:
        service_logger.error('error occurred during accessing obs, check the registry settings.')
        raise ex

    try:
        get_result = obs_client.listBuckets(False)  # 根据给定ak，sk，endpoint获取obs桶列表
        if get_result.status < RESP_STATUS_CODE:
            service_logger_without_std.info('connect to OBS successfully (HTTP status: %s).', get_result.status)
        else:
            service_logger.error('error occurred during accessing obs (HTTP status: %s), check the registry settings',
                                 get_result.status)
            raise RuntimeError
    except Exception as ex:
        raise ex
    finally:
        del cert_package
        if obs_client:
            obs_client.close()


def extract_ak_sk_endpoint_token_from_cert(cert):
    if cert[0] == '1':
        ak, sk = cert[1], cert[2]
        security_token = None
        obs_endpoint = cert[3]
    else:
        service_logger.error('unknown cert type {} on modelarts scenario'.format(cert[0]))
        raise PermissionError

    cert_package = ak, sk, obs_endpoint, security_token

    return cert_package
