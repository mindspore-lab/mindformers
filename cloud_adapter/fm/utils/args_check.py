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
import copy

from fm.aicc_tools.ailog.log import service_logger

UPPER_CASE_LETTER_LIST = [chr(i) for i in range(65, 91)]
LOWER_CASE_LETTER_LIST = [chr(i) for i in range(97, 123)]
NUMBER_LIST = [chr(i) for i in range(48, 58)]
DEFAULT_WHITE_LIST = UPPER_CASE_LETTER_LIST + LOWER_CASE_LETTER_LIST + NUMBER_LIST

S3_PROTOCOLS = ['s3://', 'obs://']
PATH_CHARACTER_BLACK_LIST = ['..', '../', '%2e%2e', '%2e./', '.%2e/', '..%2f', '%2e%2e/', '%2e.%2f', '.%2e%2f',
                             '%2e%2e%2f']

URL_CHARACTER_BLACK_LIST = ['#', '@']

# param指命令行或sdk传入的参数, config指配置文件中的参数
ENTRIES = ['param', 'config']


class LegalArgsCheckParam:
    def __init__(self, appendix, arg_key, arg_val, mode='default', min_len_limit=None, max_len_limit=None,
                 entry='param'):
        self.appendix = appendix
        self.arg_key = arg_key
        self.arg_val = arg_val
        self.mode = mode
        self.min_len_limit = min_len_limit
        self.max_len_limit = max_len_limit
        self.entry = entry


class LegalLocalPathCheckParam:
    def __init__(self, app_config, scenario, search_key, min_len_limit=None, max_len_limit=None, contains_file=False):
        self.app_config = app_config
        self.scenario = scenario
        self.search_key = search_key
        self.min_len_limit = min_len_limit
        self.max_len_limit = max_len_limit
        self.contains_file = contains_file


def is_legal_args(arg_check_item):
    """
    判断是否为合法参数
    """
    # 判断参数入口合法性(命令行/sdk入参/配置文件)
    entry_check(arg_check_item)

    # 参数长度合法性校验
    arg_content_length_check(arg_check_item)

    # 根据不同的模式, 准备对应的白名单字符库
    white_list = prepare_character_white_list(arg_check_item)

    # 白名单校验
    for c in arg_check_item.arg_val:
        if c not in white_list:
            service_logger.error('illegal character appear in %s, check the setting.', arg_check_item.arg_key)
            raise RuntimeError

    return True


def prepare_character_white_list(arg_check_item):
    # 根据不同的模式, 准备对应的白名单字符库
    if arg_check_item.mode == 'default':
        white_list = copy.deepcopy(DEFAULT_WHITE_LIST)
    elif arg_check_item.mode == 'only_letter':
        white_list = copy.deepcopy(UPPER_CASE_LETTER_LIST + LOWER_CASE_LETTER_LIST)
    elif arg_check_item.mode == 'only_number':
        white_list = copy.deepcopy(NUMBER_LIST)
    elif arg_check_item.mode == 'only_lower_letter':
        white_list = copy.deepcopy(LOWER_CASE_LETTER_LIST)
    elif arg_check_item.mode == 'only_upper_letter':
        white_list = copy.deepcopy(UPPER_CASE_LETTER_LIST)
    else:
        service_logger.error(
            'illegal parameter mode, only support default/only_letter/only_lower_letter/only_upper_letter/only_number.')
        raise RuntimeError

    # 将特定场景自定义白名单字符补充到白名单字符库
    if arg_check_item.appendix:
        white_list += arg_check_item.appendix

    return white_list


def arg_content_length_check(arg_check_item):
    # 入参合法性校验
    parameter_legality_check(arg_check_item)

    # 最大值边界判断
    if arg_check_item.max_len_limit is not None:
        content_max_len_check(arg_check_item)

    # 最小值边界判断
    if arg_check_item.min_len_limit is not None:
        content_min_len_check(arg_check_item)


def app_config_content_length_check(check_param, value):
    if check_param.min_len_limit is not None and check_param.max_len_limit is not None \
            and check_param.min_len_limit > check_param.max_len_limit:
        service_logger.error(
            'wrong parameter min_len_limit/max_len_limit, min_len_limit must be smaller than max_len_limit.')
        return False

    if check_param.min_len_limit is not None:
        if check_param.min_len_limit < 0:
            service_logger.error('wrong parameter min_len_limit, only support non-negative integer.')
            return False
        else:
            if 0 <= len(value) < check_param.min_len_limit:
                service_logger.error('param: %s from app config file is too short', check_param.search_key)
                return False

    if check_param.max_len_limit is not None:
        if check_param.max_len_limit < 0:
            service_logger.error('wrong parameter max_len_limit, only support non-negative integer.')
            return False
        else:
            if len(value) >= 0 and len(value) > check_param.max_len_limit:
                service_logger.error('param: %s from app config file is too long', check_param.search_key)
                return False

    return True


def clean_space_and_quotes(content):
    return str(content).strip().replace('\'', '').replace('"', '')


def path_check(content, s3_path=False):
    if s3_path:
        if not content:
            return False
        match_s3_protocol = False
        for p in S3_PROTOCOLS:
            if content.startswith(p):
                match_s3_protocol = True
        if not match_s3_protocol:
            return False

    for c in PATH_CHARACTER_BLACK_LIST:
        if c in content:
            return False

    return True


def url_content_black_list_characters_check(content):
    for c in URL_CHARACTER_BLACK_LIST:
        if c in content:
            return False

    return True


def entry_check(arg_check_item):
    if arg_check_item.entry not in ENTRIES:
        service_logger.error('illegal parameter entry, only support param/config.')
        raise RuntimeError


def parameter_legality_check(arg_check_item):
    if arg_check_item.min_len_limit is not None and arg_check_item.max_len_limit is not None \
            and arg_check_item.min_len_limit > arg_check_item.max_len_limit:
        service_logger.error(
            'illegal parameter min_len_limit and max_len_limit, min_len_limit must be lower than max_len_limit.')
        raise RuntimeError


def content_min_len_check(arg_check_item):
    if arg_check_item.min_len_limit < 0:
        service_logger.error('illegal parameter min_len_limit, only support non-negative integer')
        raise RuntimeError
    else:
        if 0 <= len(arg_check_item.arg_val) < arg_check_item.min_len_limit:
            service_logger.error('%s: %s is too short.', arg_check_item.entry, arg_check_item.arg_key)
            raise RuntimeError


def content_max_len_check(arg_check_item):
    if arg_check_item.max_len_limit < 0:
        service_logger.error('illegal parameter max_len_limit, only support non-negative integer')
        raise RuntimeError
    else:
        if len(arg_check_item.arg_val) >= 0 and len(arg_check_item.arg_val) > arg_check_item.max_len_limit:
            service_logger.error('%s: %s is too long.', arg_check_item.entry, arg_check_item.arg_key)
            raise RuntimeError
