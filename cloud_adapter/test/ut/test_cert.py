#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
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

from fm.src.engine.mgr import cert
from fm.src.engine.mgr.cert import CERT_TYPE_LEN_LIMIT, CERT_AK_LEN_LIMIT, CERT_SK_LEN_LIMIT, CERT_ENDPOINT_LEN_LIMIT


class TestCert(TestCase):
    # cert_param_existence_check
    def test_cert_param_existence_check_with_none_input(self):
        result1 = cert.cert_param_existence_check(cert=None)
        result2 = cert.cert_param_existence_check(cert='')
        self.assertFalse(result1)
        self.assertFalse(result2)

    def test_cert_param_legality_check_with_valid_input(self):
        input_cert = '1 ak sk endpoint'
        result = cert.cert_param_existence_check(input_cert)
        self.assertTrue(result)

    # cert_format_check
    # cert_format_category_check
    def test_cert_format_category_check_with_illegal_cert_info(self):
        cert_info = ['2', 'ak', 'sk', 'endpoint']
        with self.assertRaises(ValueError):
            cert.cert_format_check(cert_info)

    # cert_format_check_with_category_one
    # type is '1' and length of cert_info not equal to 4
    def test_cert_format_check_with_category_one_with_illegal_cert_info_len(self):
        cert_info = ['1', 'ak', 'sk']
        with self.assertRaises(ValueError):
            cert.cert_format_check(cert_info)

    # cert_info[0] == '1' and endpoint does not start with 'https'
    def test_cert_format_check_with_category_one_with_illegal_endpoint_start(self):
        cert_info = ['1', 'ak', 'sk', 'http:/']
        with self.assertRaises(ValueError):
            cert.cert_format_check(cert_info)

    def test_cert_format_check_with_category_one_with_legal_endpoint_start(self):
        cert_info1 = ['1', 'ak', 'sk', 'http://']
        cert_info2 = ['1', 'ak', 'sk', 'http://']
        self.assertIsNone(cert.cert_format_check(cert_info1))
        self.assertIsNone(cert.cert_format_check(cert_info2))

    # cert_arg_check
    # cert_overall_length_check_item
    def test_cert_overall_length_check_item_with_illegal_cert_length(self):
        cert_info = ['1' * CERT_TYPE_LEN_LIMIT, 'a' * CERT_AK_LEN_LIMIT, 's' * CERT_SK_LEN_LIMIT,
                     'e' * CERT_ENDPOINT_LEN_LIMIT + 'e']
        with self.assertRaises(RuntimeError):
            cert.cert_overall_length_check_item(cert_info)

    def test_cert_overall_length_check_item_with_legal_cert_length(self):
        cert_info = ['1' * CERT_TYPE_LEN_LIMIT, 'a' * CERT_AK_LEN_LIMIT, 's' * CERT_SK_LEN_LIMIT,
                     'e' * CERT_ENDPOINT_LEN_LIMIT]
        self.assertIsNone(cert.cert_overall_length_check_item(cert_info))

    # cert_item_length_character_check_item
    # This function only check whether the input cert_info according with the given white list rules
    def test_cert_item_length_character_check_item_with_illegal_character_in_type(self):
        # only_number, min_len_limit is 1
        cert_info = ['@', 'ak', 'sk', 'endpoint']
        with self.assertRaises(RuntimeError):
            cert.cert_item_length_character_check_item(cert_info)

    def test_cert_item_length_character_check_item_with_illegal_cert_type_length_in_type(self):
        # only_number, min_len_limit is 1, max_len_limit is 1
        cert_info1 = ['', 'ak', 'sk', 'endpoint']
        cert_info2 = ['11', 'ak', 'sk', 'endpoint']
        with self.assertRaises(RuntimeError):
            cert.cert_item_length_character_check_item(cert_info1)
        with self.assertRaises(RuntimeError):
            cert.cert_item_length_character_check_item(cert_info2)

    def test_cert_item_length_character_check_item_with_illegal_character_in_ak(self):
        cert_info = ['1 ', 'a@k', 'sk', 'endpoint']
        with self.assertRaises(RuntimeError):
            cert.cert_item_length_character_check_item(cert_info)

    def test_cert_item_length_character_check_item_with_illegal_character_in_sk(self):
        cert_info = ['1', 'ak', 's@k', 'endpoint']
        with self.assertRaises(RuntimeError):
            cert.cert_item_length_character_check_item(cert_info)

    def test_cert_item_length_character_check_item_with_illegal_character_in_endpoint(self):
        cert_info = ['1', 'ak', 'sk', 'end@point']
        with self.assertRaises(RuntimeError):
            cert.cert_item_length_character_check_item(cert_info)
