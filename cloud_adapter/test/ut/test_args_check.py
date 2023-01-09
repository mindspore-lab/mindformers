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
from unittest import TestCase

from fm.src.utils import args_check
from fm.src.utils.args_check import UPPER_CASE_LETTER_LIST, LOWER_CASE_LETTER_LIST, NUMBER_LIST, DEFAULT_WHITE_LIST
from fm.src.aicc_tools.ailog.log import ContentFilter


class TestArgsCheck(TestCase):
    # is_legal_args函数中白名单校验部分
    def test_is_legal_args_with_white_list_check(self):
        args_check_item = args_check.LegalArgsCheckParam(appendix=['@'], arg_key='key', arg_val='v%', mode='default',
                                                         min_len_limit=2, max_len_limit=2, entry='param')
        with self.assertRaises(RuntimeError):
            args_check.is_legal_args(args_check_item)

    def test_is_legal_args_with_white_list_check_true(self):
        args_check_item = args_check.LegalArgsCheckParam(appendix=['@'], arg_key='key', arg_val='va', mode='default',
                                                         min_len_limit=2, max_len_limit=2, entry='param')
        self.assertTrue(args_check.is_legal_args(args_check_item))

    # entry_check取值为['param', 'config']
    def test_entry_check_with_legal_input(self):
        args_check_item1 = args_check.LegalArgsCheckParam(appendix=['@'], arg_key='key', arg_val='va', mode='default',
                                                          min_len_limit=2, max_len_limit=2, entry='param')
        args_check_item2 = args_check.LegalArgsCheckParam(appendix=['@'], arg_key='key', arg_val='va', mode='default',
                                                          min_len_limit=2, max_len_limit=2, entry='config')
        self.assertIsNone(args_check.entry_check(args_check_item1))
        self.assertIsNone(args_check.entry_check(args_check_item2))

    def test_entry_check_with_illegal_input(self):
        args_check_item = args_check.LegalArgsCheckParam(appendix=['@'], arg_key='key', arg_val='va', mode='default',
                                                         min_len_limit=2, max_len_limit=2, entry='doahdiobciwbc')
        with self.assertRaises(RuntimeError):
            args_check.entry_check(args_check_item)

    # arg_content_length_check
    # parameter_legality_check
    # min_len_limit > max_len_limit
    def test_parameter_legality_check_with_illegal_min_greater_than_max_len_limit(self):
        args_check_item = args_check.LegalArgsCheckParam(appendix=[], arg_key='key', arg_val='val', mode='default',
                                                         min_len_limit=3, max_len_limit=2, entry='param')
        with self.assertRaises(RuntimeError):
            args_check.parameter_legality_check(args_check_item)

    # content_max_len_check
    # max_len_limit 小于等于 0
    def test_content_max_len_check_with_illegal_max_len_limit_lower_than_zero(self):
        args_check_item = args_check.LegalArgsCheckParam(appendix=[], arg_key='key', arg_val='', mode='default',
                                                         min_len_limit=-1, max_len_limit=-1, entry='param')
        with self.assertRaises(RuntimeError):
            args_check.content_max_len_check(args_check_item)

    # length of arg_val 大于 max_len_limit
    def test_arg_content_length_check_with_illegal_arg_val_len_greater_than_max_len_limit(self):
        args_check_item = args_check.LegalArgsCheckParam(appendix=[], arg_key='key', arg_val='vl', mode='default',
                                                         min_len_limit=-1, max_len_limit=1, entry='param')
        with self.assertRaises(RuntimeError):
            args_check.arg_content_length_check(args_check_item)

    # content_min_len_check
    # min_len_limit小于0
    def test_content_min_len_check_with_illegal_min_len_limit_lower_than_zero(self):
        args_check_item = args_check.LegalArgsCheckParam(appendix=[], arg_key='key', arg_val='v', mode='default',
                                                         min_len_limit=-1, max_len_limit=1, entry='param')
        with self.assertRaises(RuntimeError):
            args_check.content_min_len_check(args_check_item)

    # length of arg_val 小于 min_len_limit
    def test_content_min_len_check_illegal_arg_val_len_lower_than_min_len_limit(self):
        args_check_item = args_check.LegalArgsCheckParam(appendix=[], arg_key='key', arg_val='', mode='default',
                                                         min_len_limit=1, max_len_limit=2, entry='param')
        with self.assertRaises(RuntimeError):
            args_check.content_min_len_check(args_check_item)

    # prepare_character_white_list
    # wrong mode
    def test_prepare_character_white_list_with_illegal_mode(self):
        args_check_item = args_check.LegalArgsCheckParam(appendix=[], arg_key='key', arg_val='va', mode='others',
                                                         min_len_limit=2, max_len_limit=2, entry='param')
        with self.assertRaises(RuntimeError):
            args_check.prepare_character_white_list(args_check_item)

    # mode branch
    def test_prepare_character_white_list_with_mode_default(self):
        args_check_item = args_check.LegalArgsCheckParam(appendix=[], arg_key='key', arg_val='va', mode='default',
                                                         min_len_limit=2, max_len_limit=2, entry='param')
        ret = args_check.prepare_character_white_list(args_check_item)
        self.assertEqual(ret, DEFAULT_WHITE_LIST)

    def test_prepare_character_white_list_with_mode_only_letter(self):
        args_check_item = args_check.LegalArgsCheckParam(appendix=[], arg_key='key', arg_val='va', mode='only_letter',
                                                         min_len_limit=2, max_len_limit=2, entry='param')
        ret = args_check.prepare_character_white_list(args_check_item)
        self.assertEqual(ret, UPPER_CASE_LETTER_LIST + LOWER_CASE_LETTER_LIST)

    def test_prepare_character_white_list_with_mode_only_number(self):
        args_check_item = args_check.LegalArgsCheckParam(appendix=[], arg_key='key', arg_val='va', mode='only_number',
                                                         min_len_limit=2, max_len_limit=2, entry='param')
        ret = args_check.prepare_character_white_list(args_check_item)
        self.assertEqual(ret, NUMBER_LIST)

    def test_prepare_character_white_list_with_mode_only_lower_letter(self):
        args_check_item = args_check.LegalArgsCheckParam(appendix=[], arg_key='key', arg_val='va',
                                                         mode='only_lower_letter',
                                                         min_len_limit=2, max_len_limit=2, entry='param')
        ret = args_check.prepare_character_white_list(args_check_item)
        self.assertEqual(ret, LOWER_CASE_LETTER_LIST)

    def test_prepare_character_white_list_with_mode_only_upper_letter(self):
        args_check_item = args_check.LegalArgsCheckParam(appendix=[], arg_key='key', arg_val='va',
                                                         mode='only_upper_letter',
                                                         min_len_limit=2, max_len_limit=2, entry='param')
        ret = args_check.prepare_character_white_list(args_check_item)
        self.assertEqual(ret, UPPER_CASE_LETTER_LIST)

    # white list supplement
    def test_is_legal_args_with_white_list_plus_appendix(self):
        args_check_item = args_check.LegalArgsCheckParam(appendix=['@'], arg_key='key', arg_val='va',
                                                         mode='only_upper_letter',
                                                         min_len_limit=2, max_len_limit=2, entry='param')
        ret = args_check.prepare_character_white_list(args_check_item)
        self.assertEqual(ret, UPPER_CASE_LETTER_LIST + ['@'])

    # app_config_content_length_check
    # min_len_limit 大于 max_len_limit
    def test_app_config_content_length_check_with_illegal_min_greater_than_max_len_limit(self):
        check_param = args_check.LegalLocalPathCheckParam(app_config='a', scenario='a', search_key='a', min_len_limit=3,
                                                          max_len_limit=2, contains_file=False)
        self.assertFalse(args_check.app_config_content_length_check(check_param, value='a'))

    # min_len_limit 小于等于 0
    def test_app_config_content_length_check_illegal_min_len_limit_smaller_than_zero(self):
        check_param = args_check.LegalLocalPathCheckParam(app_config='a', scenario='a', search_key='a', min_len_limit=0,
                                                          max_len_limit=2, contains_file=False)
        self.assertTrue(args_check.app_config_content_length_check(check_param, value='a'))

    # len(value)=0 小于 min_len_limit
    def test_app_config_content_length_check_illegal_min_len_limit_smaller_than(self):
        check_param = args_check.LegalLocalPathCheckParam(app_config='a', scenario='a', search_key='a', min_len_limit=1,
                                                          max_len_limit=1, contains_file=False)
        self.assertFalse(args_check.app_config_content_length_check(check_param, value=''))

    # max_len_limit 小于等于 0
    def test_app_config_content_length_check_with_illegal_min_len_limit_smaller_than_zero(self):
        check_param = args_check.LegalLocalPathCheckParam(app_config='a', scenario='a', search_key='a',
                                                          min_len_limit=-5, max_len_limit=-5, contains_file=False)
        self.assertFalse(args_check.app_config_content_length_check(check_param, value='1'))

    # clean_space_and_quotes
    def test_clean_space_and_quotes(self):
        input_sample = '    test\'test\'test"test"     '
        result = args_check.clean_space_and_quotes(input_sample)
        self.assertEqual(result, 'testtesttesttest')

    #  path_check
    def test_path_check_with_no_content(self):
        content = None
        self.assertFalse(args_check.path_check(content, s3_path=True))

    # url_content_black_list_characters_check
    def test_url_content_black_list_characters_check(self):
        #  check if characters ['#', '@'] can be detected
        self.assertTrue(args_check.url_content_black_list_characters_check('8340y9fsahf904r29th29308rh '))
        self.assertFalse(args_check.url_content_black_list_characters_check('8340y9#sfsahf904r29th29308rh '))
        self.assertFalse(args_check.url_content_black_list_characters_check('8340y9@sfsahf904r29th29308rh '))

    # log_content_black_list_characters_check
    def test_log_content_black_list_characters_check_with_legal_input(self):  # ['    ', '\r', '\n', '\'', '"']
        content_filter = ContentFilter()
        # single space
        self.assertTrue(content_filter.content_check('8340y9fsahf904r29th29308rh '))
        # single \
        self.assertTrue(content_filter.content_check('8340y9fsahf904r29th29308rh\ '))

    def test_log_content_black_list_characters_check_with_illegal_content(self):
        content_filter = ContentFilter()

        with self.assertRaises(RuntimeError):
            content_filter.content_check('8340y9#sfsahf904r29th29308rh    ')
        with self.assertRaises(RuntimeError):
            content_filter.content_check('8340y9#sfsahf904r29th29308\rh    ')
        with self.assertRaises(RuntimeError):
            content_filter.content_check('8340y9\n#sfsahf904r29th29308rh    ')
        with self.assertRaises(RuntimeError):
            content_filter.content_check('8340y9#sfsahf904r29th29308rh    \'')
        with self.assertRaises(RuntimeError):
            content_filter.content_check("8340y9#sfsahf904r29th29308rh0'")
        with self.assertRaises(RuntimeError):
            content_filter.content_check('8340y9#sfsahf904r29th29308rh "   ')
