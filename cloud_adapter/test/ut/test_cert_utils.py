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
import unittest.mock as mock
from unittest import TestCase

from fm.src.utils import cert_utils


class TestCertUtils(TestCase):
    # pack_cert_plain
    # 此处给unittest编号指定局部test执行顺序
    def test01_pack_cert_plain(self):
        cert_input = ['1', 'ak', 'sk', 'endpoint']
        cert_utils.encrypt_with_kmc = mock.Mock(return_value='ak_sk_cipher')
        result = cert_utils.pack_cert_plain(cert_input)
        target = {
            'type': '1',
            'ak': 'ak_sk_cipher',
            'sk': 'ak_sk_cipher',
            'endpoint': 'endpoint'
        }
        self.assertEqual(result, target)

    # encrypt_cert
    def test02_encrypt_cert(self):
        cert_input = '1 ak sk endpoint'
        cert_output = {'scenario': dict()}  # cert_output is a dic, the outside key is scenario
        cert_dict = {
            'type': '1',
            'ak': 'cert_ak_cipher',
            'sk': ' cert_sk_cipher',
            'endpoint': 'cert_endpoint'
        }
        target_ret = {
            'scenario': {
                'type': '1',
                'ak': 'cert_ak_cipher',
                'sk': ' cert_sk_cipher',
                'endpoint': 'cert_endpoint'
            }
        }
        cert_utils.pack_cert_plain = mock.Mock(return_value=cert_dict)
        cert_utils.encrypt_cert(cert_input, cert_output)
        self.assertEqual(cert_output, target_ret)

    # decrypt_cert
    # legal input, flatten_to_list is True
    def test_decrypt_cert_with_flatten_to_list_on(self):
        cert_cipher = {'scenario': {'type': '1', 'ak': 'ak', 'sk': 'sk', 'endpoint': 'endpoint'}}
        cert_utils.decrypt_with_kmc = mock.Mock(return_value='aksk1')
        ret = cert_utils.decrypt_cert(cert_cipher, flatten_to_list=True)
        self.assertEqual(ret, {'scenario': ['1', 'aksk1', 'aksk1', 'endpoint']})

    # legal input, flatten_to_list is False
    def test_decrypt_cert_with_flatten_to_list_off(self):
        cert_cipher = {'scenario': {'type': '1', 'ak': 'ak', 'sk': 'sk', 'endpoint': 'endpoint'}}
        cert_utils.decrypt_with_kmc = mock.Mock(return_value='aksk2')
        ret = cert_utils.decrypt_cert(cert_cipher, flatten_to_list=False)
        self.assertEqual(ret, {'scenario': {'type': '1', 'ak': 'aksk2', 'sk': 'aksk2', 'endpoint': 'endpoint'}})

    # unpack_cert_cipher
    # no scenario
    def test_unpack_cert_cipher_with_no_scenario(self):
        cert_cipher = {'illegal': {'type': '1', 'ak': 'ak', 'sk': 'sk', 'endpoint': 'endpoint'}}
        with self.assertRaises(RuntimeError):
            cert_utils.unpack_cert_cipher(cert_cipher)

    # scenario is None
    def test_unpack_cert_cipher_with_scenario_is_none(self):
        cert_cipher = {'scenario': None}
        with self.assertRaises(RuntimeError):
            cert_utils.unpack_cert_cipher(cert_cipher)

    # scenario has no keys
    def test_unpack_cert_cipher_with_scenario_has_no_keys(self):
        cert_cipher = {'scenario': dict()}
        with self.assertRaises(RuntimeError):
            cert_utils.unpack_cert_cipher(cert_cipher)

    # cert_info_value is None
    def test_unpack_cert_cipher_with_cert_info_value_is_none(self):
        cert_cipher = {'scenario': {'type': '1', 'ak': 'ak', 'sk': None, 'endpoint': 'endpoint'}}
        with self.assertRaises(RuntimeError):
            cert_utils.unpack_cert_cipher(cert_cipher)

    # true input
    def test_unpack_cert_cipher_with_true_input(self):
        cert_cipher = {'scenario': {'type': '1', 'ak': 'ak', 'sk': 'dw', 'endpoint': 'endpoint'}}
        ret = cert_utils.unpack_cert_cipher(cert_cipher)
        self.assertEqual(ret, {'type': '1', 'ak': 'ak', 'sk': 'dw', 'endpoint': 'endpoint'})
