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
import stat

from fm.src.utils import io_utils, get_config_dir_setting


class TestIOUtils(TestCase):
    # wrap_local_working_directory
    # file_name is None or ''
    def test_wrap_local_working_directory_with_illegal_file_name(self):
        file_name1 = None
        file_name2 = ''
        with self.assertRaises(RuntimeError):
            io_utils.wrap_local_working_directory(file_name1)
        with self.assertRaises(RuntimeError):
            io_utils.wrap_local_working_directory(file_name2)

    # specific_path_config is None
    def test_wrap_local_working_directory_with_specific_path_config_is_none(self):
        file_name = 'test.txt'
        result = io_utils.wrap_local_working_directory(file_name=file_name,
                                                       specific_path_config=None)
        target = os.path.join(os.getenv('HOME'), '.cache', 'Huawei/mxFoundationModel/', file_name)
        self.assertEqual(result, target)

    # specific_path_config is not None
    def test_wrap_local_working_directory_with_true_input(self):
        file_name = 'test.txt'
        result = io_utils.wrap_local_working_directory(file_name=file_name,
                                                       specific_path_config=get_config_dir_setting())
        target = os.path.join(os.getenv('HOME'), '.cache', 'Huawei/mxFoundationModel/config', file_name)
        self.assertEqual(result, target)

    # specific_path_config_legality_check
    # specific_path is None
    def test_specific_path_config_legality_check_with_specific_path_is_none(self):
        specific_path_config = {'path': None, 'rule': stat.S_IRWXU}
        with self.assertRaises(RuntimeError):
            io_utils.specific_path_config_legality_check(specific_path_config)

    # specific_path_rule is None
    def test_specific_path_config_legality_check_with_specific_rule_is_none(self):
        specific_path_config = {'path': 'config', 'rule': None}
        with self.assertRaises(RuntimeError):
            io_utils.specific_path_config_legality_check(specific_path_config)

    # True input and output
    def test_specific_path_config_legality_check_with_true_input(self):
        specific_path_config = {'path': 'config', 'rule': stat.S_IRWXU}
        tar = ['config', stat.S_IRWXU]
        res = io_utils.specific_path_config_legality_check(specific_path_config)
        self.assertEqual(res, tar)
