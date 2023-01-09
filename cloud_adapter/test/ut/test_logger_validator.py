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
import logging
import copy
from unittest import TestCase

import fm.src.aicc_tools.ailog.log as ac

LOG_STD_INPUT_PARAM = {
    'to_std': True,
    'stdout_nodes': None,
    'stdout_devices': None,
    'stdout_level': 'WARNING'
}
LOG_FILE_INPUT_PARAM = {
    'file_level': ['INFO'],
    'file_save_dir': "~/.cache/Huawei/mxFoundationModel/log/",
    'append_rank_dir': False,
    'file_name': ['aicc.INFO.log']
}


class TestValidateInputFormat(TestCase):
    LOG_STD_INPUT_PARAM = None
    LOG_FILE_INPUT_PARAM = None

    def setUp(self) -> None:
        self.get_logger_std_input = copy.deepcopy(LOG_STD_INPUT_PARAM)
        self.get_logger_file_input = copy.deepcopy(LOG_FILE_INPUT_PARAM)

    def test_to_std(self):
        ac.validate_std_input_format(**self.get_logger_std_input)

        self.get_logger_std_input['to_std'] = 0
        with self.assertRaises(TypeError):
            ac.validate_std_input_format(**self.get_logger_std_input)

    def stdout_nodes_or_device_test(self, key):
        ac.validate_std_input_format(**self.get_logger_std_input)

        self.get_logger_std_input[key] = [0, 1]
        ac.validate_std_input_format(**self.get_logger_std_input)

        self.get_logger_std_input[key] = (0, 1)
        ac.validate_std_input_format(**self.get_logger_std_input)

        self.get_logger_std_input[key] = {'start': 0, 'end': 1}
        ac.validate_std_input_format(**self.get_logger_std_input)

        self.get_logger_std_input[key] = '0'
        with self.assertRaises(TypeError):
            ac.validate_std_input_format(**self.get_logger_std_input)

        self.get_logger_std_input[key] = ('0', '1')
        with self.assertRaises(TypeError):
            ac.validate_std_input_format(**self.get_logger_std_input)

        self.get_logger_std_input[key] = {'start': 0}
        ac.validate_std_input_format(**self.get_logger_std_input)

        self.get_logger_std_input[key] = {'start': '0', 'end': '1'}
        ac.validate_std_input_format(**self.get_logger_std_input)

        self.get_logger_std_input[key] = {'start': 1, 'end': 0}
        ac.validate_std_input_format(**self.get_logger_std_input)

    def test_stdout_nodes(self):
        self.stdout_nodes_or_device_test('stdout_nodes')

    def test_stdout_devices(self):
        self.stdout_nodes_or_device_test('stdout_devices')

    def level_test(self, key):
        ac.validate_std_input_format(**self.get_logger_std_input)

        self.get_logger_std_input[key] = 'INFO'
        ac.validate_std_input_format(**self.get_logger_std_input)

        self.get_logger_std_input[key] = 4
        with self.assertRaises(TypeError):
            ac.validate_std_input_format(**self.get_logger_std_input)

        self.get_logger_std_input[key] = '0'
        with self.assertRaises(ValueError):
            ac.validate_std_input_format(**self.get_logger_std_input)

        self.get_logger_std_input[key] = 5
        with self.assertRaises(TypeError):
            ac.validate_std_input_format(**self.get_logger_std_input)

        self.get_logger_std_input[key] = -1
        with self.assertRaises(TypeError):
            ac.validate_std_input_format(**self.get_logger_std_input)

        self.get_logger_std_input[key] = 3.14
        with self.assertRaises(TypeError):
            ac.validate_std_input_format(**self.get_logger_std_input)

    def file_level_test(self, key):
        ac.validate_file_input_format(**self.get_logger_file_input)

        self.get_logger_file_input[key] = 'INFO'
        with self.assertRaises(TypeError):
            ac.validate_file_input_format(**self.get_logger_file_input)

        self.get_logger_file_input[key] = 4
        with self.assertRaises(TypeError):
            ac.validate_file_input_format(**self.get_logger_file_input)

        self.get_logger_file_input[key] = '0'
        with self.assertRaises(TypeError):
            ac.validate_file_input_format(**self.get_logger_file_input)

        self.get_logger_file_input[key] = 5
        with self.assertRaises(TypeError):
            ac.validate_file_input_format(**self.get_logger_file_input)

        self.get_logger_file_input[key] = -1
        with self.assertRaises(TypeError):
            ac.validate_file_input_format(**self.get_logger_file_input)

        self.get_logger_file_input[key] = 3.14
        with self.assertRaises(TypeError):
            ac.validate_file_input_format(**self.get_logger_file_input)

    def test_stdout_level(self):
        self.level_test('stdout_level')

    def test_file_level(self):
        self.file_level_test('file_level')

    def test_file_save_dir(self):
        ac.validate_file_input_format(**self.get_logger_file_input)

        self.get_logger_file_input['file_save_dir'] = ''
        ac.validate_file_input_format(**self.get_logger_file_input)

        self.get_logger_file_input['file_save_dir'] = './'
        ac.validate_file_input_format(**self.get_logger_file_input)

        self.get_logger_file_input['file_save_dir'] = 1
        with self.assertRaises(TypeError):
            ac.validate_file_input_format(**self.get_logger_file_input)

    def test_max_file_size(self):
        ac.validate_file_input_format(**self.get_logger_file_input)

        self.get_logger_file_input['max_file_size'] = 3.14
        with self.assertRaises(TypeError):
            ac.validate_std_input_format(**self.get_logger_file_input)

    def test_max_num_of_files(self):
        ac.validate_file_input_format(**self.get_logger_file_input)

        self.get_logger_file_input['max_num_of_files'] = 3.14
        with self.assertRaises(TypeError):
            ac.validate_std_input_format(**self.get_logger_file_input)

    def test_to_std(self):
        ac.validate_std_input_format(**self.get_logger_std_input)

        self.get_logger_std_input['to_std'] = 0
        with self.assertRaises(TypeError):
            ac.validate_std_input_format(**self.get_logger_std_input)

    def test_get_rank_info(self):
        os.environ['RANK_ID'] = '2'
        os.environ['RANK_SIZE'] = '8'

        rank_id, rank_size = ac.get_rank_info()
        self.assertEqual(rank_id, 2)
        self.assertEqual(rank_size, 8)

        os.environ.pop('RANK_ID')
        os.environ.pop('RANK_SIZE')

    def test_convert_nodes_devices_input(self):
        var = None
        num = 4
        self.assertEqual(ac.convert_nodes_devices_input(var, num), (0, 1, 2, 3))

        var = {'start': 0, 'end': 4}
        self.assertEqual(ac.convert_nodes_devices_input(var, num), (0, 1, 2, 3))

        var = (0, 1)
        self.assertEqual(ac.convert_nodes_devices_input(var, num), (0, 1))

        var = [0, 1]
        self.assertEqual(ac.convert_nodes_devices_input(var, num), [0, 1])

    def test_get_num_nodes_devices(self):
        rank_size = 4
        num_nodes, num_devices = ac.get_num_nodes_devices(rank_size)
        self.assertEqual(num_nodes, 1)
        self.assertEqual(num_devices, 4)

        rank_size = 16
        num_nodes, num_devices = ac.get_num_nodes_devices(rank_size)
        self.assertEqual(num_nodes, 2)
        self.assertEqual(num_devices, 8)

    def test_check_list(self):
        var_name = 'stdout_nodes'
        list_var = [0, 1]
        num = 4
        ac.check_list(var_name, list_var, num)

        var_name = 'stdout_nodes'
        list_var = [0, 1, 2, 3]
        num = 2
        with self.assertRaises(ValueError):
            ac.check_list(var_name, list_var, num)

    def test_generate_rank_list(self):
        stdout_nodes = [0, 1]
        stdout_devices = [0, 1]
        self.assertEqual(ac.generate_rank_list(stdout_nodes, stdout_devices), [0, 1, 8, 9])


class TestValidate(TestCase):
    def test_input_validate_level_type(self):
        with self.assertRaises(TypeError):
            ac.validate_level('std_out_level', 1)

    def test_input_validate_level_value(self):
        with self.assertRaises(ValueError):
            ac.validate_level('std_out_level', "AA")

    def test_validate_file_input_len(self):
        with self.assertRaises(ValueError):
            ac.validate_file_input_format(file_level=['INFO', 'ERROR'], file_name=['logger'], file_save_dir='./',
                                          append_rank_dir=False)

    def test_validate_file_input_type(self):
        with self.assertRaises(TypeError):
            ac.validate_file_input_format(file_level='INFO', file_name=['logger'],
                                          file_save_dir='./', append_rank_dir=False)
        with self.assertRaises(TypeError):
            ac.validate_file_input_format(file_level='INFO', file_name='logger',
                                          file_save_dir='./', append_rank_dir=False)

    def test_judge_stdout_should_false(self):
        self.assertFalse(ac.judge_stdout(0, 1, False))

    def test_judge_stdout_should_true(self):
        self.assertTrue(ac.judge_stdout(0, 1, True, None, (0,)))

    def test_get_stream_handler(self):
        handler = ac.get_stream_handler(None, "INFO")
        self.assertIsInstance(handler, logging.Handler)
