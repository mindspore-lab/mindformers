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
import getpass

from fm.src.aicc_tools.ailog.log import service_logger, service_logger_without_std, operation_logger_without_std
from fm.src.utils.args_check import clean_space_and_quotes, is_legal_args, LegalArgsCheckParam
from fm.src.utils import wrap_local_working_directory, constants, write_file_with_link_check, \
    read_file_with_link_check, encrypt_cert, decrypt_cert, get_config_dir_setting, obs_connection_check
from fm.src.utils.io_utils import DEFAULT_FLAGS, DEFAULT_MODES

CERT_TYPE_LEN_LIMIT = 1
CERT_AK_LEN_LIMIT = 40
CERT_SK_LEN_LIMIT = 80
CERT_ENDPOINT_LEN_LIMIT = 100


def manually_input_cert():
    """
    通过交互方式获取用户手动输入认证凭据信息
    """
    cert_type = input('registry type: ').strip()
    cert_item_legality_check(cert_item=cert_type, item_type='registry type')

    if cert_type == '1':
        return get_and_check_cert_item_with_category_one(cert_type=cert_type)
    else:
        service_logger.error('registry type is illegal, support type [%s], check the setting.',
                             ' '.join(constants.REGISTRY_SUPPORT_TYPES))
        raise RuntimeError


def get_and_check_cert_item_with_category_one(cert_type):
    cert_ak = getpass.getpass('registry ak: ').strip()
    cert_item_legality_check(cert_item=cert_ak, item_type='registry ak')

    cert_sk = getpass.getpass('registry sk: ').strip()
    cert_item_legality_check(cert_item=cert_sk, item_type='registry sk')

    cert_endpoint = input('registry endpoint: ').strip()
    cert_item_legality_check(cert_item=cert_endpoint, item_type='registry endpoint')

    return ' '.join([cert_type, cert_ak, cert_sk, cert_endpoint])


def cert_item_legality_check(cert_item, item_type):
    if cert_item is None or cert_item == '' or not isinstance(cert_item, str):
        service_logger.error('legal %s is necessary.', item_type)
        raise RuntimeError


def cert_param_existence_check(cert):
    """
    判断认证信息是否存在
    """
    if cert is None or cert == '':
        service_logger.error('registry is necessary, please specify cert in parameter.')
        return False

    return True


def cache_cert(cert):
    """
    将认证凭据缓存到本地
    """
    operation_logger_without_std.info('registry starts')
    if not isinstance(cert, str):
        try:
            cert = cert.decode('utf-8', 'ignore')
        except Exception:
            service_logger.error('can not cast registry info into str type, check the registry info.')
            return False

    cert_info = cert.split(' ')
    del cert

    # cert 格式检查
    cert_format_check(cert_info)

    # cert 内容合法性校验
    cert_arg_check(cert_info)

    # cert 内容是否能够成功访问到给定节点（obs）
    obs_connection_check(cert_info)

    # 刷新本地.cert文件缓存
    refresh_local_cert_cache(cert_info)

    service_logger.info('set registry info successfully.')

    operation_logger_without_std.info('registry ends, the result is True')

    return True


def cert_format_check(cert_info):
    """
    认证凭据格式检查
    """
    # registry首位表示凭证类型, 需要在指定的范围内
    cert_format_category_check(cert_info)

    # 类型1认证信息校验
    cert_format_check_with_category_one(cert_info)


def cert_format_category_check(cert_info):
    """
    认证信息类型合法性校验
    """
    if cert_info[0] not in ['1']:
        service_logger.error('registry type is illegal, support type [%s], check the setting.',
                             ' '.join(constants.REGISTRY_SUPPORT_TYPES))
        raise ValueError


def cert_format_check_with_category_one(cert_info):
    """
    类型1认证信息校验
    """
    # 针对认证类型1的凭据, 需要按照1 ak sk endpoint格式提供
    if cert_info[0] == '1' and len(cert_info) != 4:
        service_logger.error('illegal registry format, type 1 should follow format \'1 ak sk obs_endpoint\'.')
        raise ValueError

    # 针对认证类型1的凭据, 需要保障endpoint以https/http开头
    if cert_info[0] == '1' and not (cert_info[3].startswith('http://') or cert_info[3].startswith('https://')):
        service_logger.error('illegal endpoint format in registry, check the setting.')
        raise ValueError

    # 针对认证类型1的凭据, 如果endpoint只为'https://', 'http://'即为错误，需要有网址内容
    if cert_info[0] == '1' and (cert_info[3] == 'http://' or cert_info[3] == 'https://'):
        service_logger.error('illegal endpoint format in registry, check the setting.')
        raise ValueError


def cert_arg_check(cert_info):
    """
    认证信息内容合法性检查
    """
    # 认证类型1的内容合法性校验
    if cert_info[0] == '1':
        # 检查cert整体长度是否合法
        cert_overall_length_check_item(cert_info)

        # 检查cert每一部分长度是否合法, 以及对应字符类型是否合法
        cert_item_length_character_check_item(cert_info)


def cert_overall_length_check_item(cert_info):
    """
    认证信息整体长度校验项
    """
    # 认证信息整体字符串长度限制
    overall_length_limit = CERT_TYPE_LEN_LIMIT + CERT_AK_LEN_LIMIT + CERT_SK_LEN_LIMIT + CERT_ENDPOINT_LEN_LIMIT

    if sum([len(c) for c in cert_info]) > overall_length_limit:
        service_logger.error('registry info length is too long, check the input.')
        raise RuntimeError


def cert_item_length_character_check_item(cert_info):
    """
    认证信息子部分长度及字符合法性校验项
    """
    # 认证凭据各子部分允许的字符范围及长度范围
    item_character_limit = [
        LegalArgsCheckParam(appendix=[], arg_val=clean_space_and_quotes(cert_info[0]),
                            entry='param', mode='only_number', arg_key='registry type',
                            min_len_limit=1, max_len_limit=CERT_TYPE_LEN_LIMIT),
        LegalArgsCheckParam(appendix=[], arg_val=clean_space_and_quotes(cert_info[1]),
                            entry='param', mode='default', arg_key='registry ak',
                            min_len_limit=1, max_len_limit=CERT_AK_LEN_LIMIT),
        LegalArgsCheckParam(appendix=[], arg_val=clean_space_and_quotes(cert_info[2]),
                            entry='param', mode='default', arg_key='registry sk',
                            min_len_limit=1, max_len_limit=CERT_SK_LEN_LIMIT),
        LegalArgsCheckParam(appendix=[':', '/', '.', '-'], arg_val=clean_space_and_quotes(cert_info[3]),
                            entry='param', mode='default', arg_key='registry endpoint',
                            min_len_limit=1, max_len_limit=CERT_ENDPOINT_LEN_LIMIT)
    ]

    for i in range(len(cert_info)):
        try:
            is_legal_args(item_character_limit[i])
        except Exception as ex:
            service_logger.error('registry info is illegal, check the setting.')
            raise ex

    del item_character_limit


def refresh_local_cert_cache(cert_input):
    """
    将认证信息刷新到本地缓存文件
    """
    # 组装认证凭据本地缓存文件位置
    cert_file_local_path = wrap_local_working_directory(file_name=constants.CERT_FILE_LOCAL_PATH,
                                                        specific_path_config=get_config_dir_setting())

    # 获取本地缓存认证信息/默认空认证信息
    if os.path.exists(cert_file_local_path):
        try:
            cert_output_cipher = read_file_with_link_check(cert_file_local_path, DEFAULT_FLAGS, DEFAULT_MODES)
        except Exception as ex:
            service_logger.error('error occurred when loading local registry cache, see service log for error message.')
            service_logger_without_std(ex)
            raise ex

        try:
            cert_output = decrypt_cert(cert_output_cipher)
        except Exception as ex:
            service_logger.error('error occurred when decrypting, see service log for error message.')
            service_logger_without_std(ex)
            raise ex
    else:
        cert_output = {'scenario': dict()}

    # 将新认证信息加密刷新到存量认证信息中
    try:
        encrypt_cert(cert_input, cert_output)
    except Exception as ex:
        service_logger.error('error occurred when encrypting local registry cache, see service log for error message.')
        service_logger_without_std(ex)
        raise ex

    if os.path.isfile(cert_file_local_path):
        os.remove(cert_file_local_path)

    try:
        write_file_with_link_check(cert_file_local_path, cert_output, DEFAULT_FLAGS, DEFAULT_MODES)
    except Exception as ex:
        service_logger.error('error occurred when saving local registry cache, see service log for error message.')
        service_logger_without_std(ex)
        raise ex

    del cert_output
