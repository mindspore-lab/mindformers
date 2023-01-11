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
import hashlib
import os
import shutil
from unittest import TestCase

from fm.src.utils.io_utils import wrap_local_working_directory
from fm.src.utils.obs_tool import calculate_file_md5
from fm.src.aicc_tools.utils.validator import check_in_modelarts

TXT_FILE_PATH = wrap_local_working_directory('test.txt')


class TestObsTool(TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        os.remove(TXT_FILE_PATH)

    def testMd5ShouldEqual(self):
        data_content = '0' * 32 * 1024 * 1024
        hexdigest = hashlib.md5(data_content.encode('utf-8')).hexdigest()
        with open(TXT_FILE_PATH, 'w') as file:
            file.write(data_content)
        file_md = calculate_file_md5(TXT_FILE_PATH)
        self.assertEqual(file_md, hexdigest)

    def testMd5ShouldNotEqual(self):
        data_content = '0' * 32 * 1024 * 1024
        hexdigest = hashlib.md5(data_content.encode('utf-8')).hexdigest()
        with open(TXT_FILE_PATH, 'w') as file:
            file.write(data_content * 2)
        file_md = calculate_file_md5(TXT_FILE_PATH)
        self.assertNotEqual(file_md, hexdigest)

    def testFileMd5ShouldEqual(self):
        data_content = '0' * 32 * 1024 * 1024
        with open(TXT_FILE_PATH, 'w') as file:
            file.write(data_content)
        new_path = wrap_local_working_directory('testtest.txt')
        shutil.copyfile(TXT_FILE_PATH, new_path)
        self.assertEqual(calculate_file_md5(TXT_FILE_PATH), calculate_file_md5(new_path))
        os.remove(new_path)

    def testLargeFile(self):
        large_content = '0' * 6 * 1024 * 1024 * 1024
        hexdigest = hashlib.md5(large_content.encode('utf-8')).hexdigest()
        if check_in_modelarts():
            import moxing as mox
            remote_content = mox.file.read('obs://chenhao-test/test/a.txt')
            remote_md5 = hashlib.md5(remote_content)
            self.assertEqual(remote_md5, hexdigest)
