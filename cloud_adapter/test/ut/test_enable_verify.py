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
import os
from unittest import TestCase

from fm.fm_sdk import enable_verify
from fm.src.utils import constants, wrap_local_working_directory, \
    get_ca_dir_setting, read_file_with_link_check, set_ssl_verify
from fm.src.engine.mgr.enable_verify import cert_verify
from fm.src.utils.io_utils import DEFAULT_FLAGS, DEFAULT_MODES


CA_FILE_PATH = wrap_local_working_directory(file_name=constants.CA_FILE_LOCAL_PATH,
                                            specific_path_config=get_ca_dir_setting())
VERIFY_CONFIG_PATH = wrap_local_working_directory(file_name=constants.VERIFY_CONFIG_LOCAL_PATH,
                                                  specific_path_config=get_ca_dir_setting())


class EnableVerifyTest(TestCase):
    def tearDown(self) -> None:
        if os.path.exists(CA_FILE_PATH):
            os.remove(CA_FILE_PATH)

    def test_cert_path_check_raise_error(self):
        with self.assertRaises(ValueError):
            cert_verify(enable=True, path=None)

    def test_cert_path_check(self):
        enable_verify(enable=False)
        verify_info = read_file_with_link_check(VERIFY_CONFIG_PATH, DEFAULT_FLAGS, DEFAULT_MODES)
        self.assertFalse(verify_info.get('enable'))

    def test_ca_file_exist(self):
        enable_verify(enable=True, path='/home/normal157/ch_test/fm.cer')
        self.assertTrue(os.path.exists(CA_FILE_PATH))

    def test_ca_file_not_exist(self):
        enable_verify(enable=True, path='/home/normal157/ch_test/fm.cer')
        os.remove(CA_FILE_PATH)
        with self.assertRaises(RuntimeError):
            set_ssl_verify(dict())

    def test_ca_file_still_exist(self):
        enable_verify(enable=True, path='/home/normal157/ch_test/fm.cer')
        enable_verify(enable=False)
        self.assertTrue(os.path.exists(CA_FILE_PATH))

    def test_ca_file_prama_error(self):
        with self.assertRaises(ValueError):
            enable_verify(enable=True, path='/home/fm.cer')
