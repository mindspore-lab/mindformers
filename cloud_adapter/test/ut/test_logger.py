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
import shutil
import pytest
from unittest import TestCase

from fm.src.aicc_tools.ailog.log import get_logger

CONFIG_TEST_BASE_DIR = os.path.expanduser('~/.cache/Huawei/mxFoundationModel/log')
CONFIG_TEST_LOG_DIR = os.path.realpath(os.path.join(CONFIG_TEST_BASE_DIR, 'rank_0'))

LOG_MESSAGE = {
    'debug': 'debug message',
    'info': 'info message',
    'warning': 'warning message',
    'error': 'error message',
    'critical': 'critical message'
}


class TestLogger(TestCase):

    def test_logger_level(self):
        logger = get_logger("logger", file_level=['INFO', 'ERROR'])
        with self.assertLogs(logger) as cm:
            logger.info("A test info message")
            logger.error("A test error message")
        self.assertEqual(cm.output, ['INFO:logger:A test info message', 'ERROR:logger:A test error message'])

    def test_logger_to_std(self):
        logger = get_logger("logger", file_level=['INFO', 'ERROR'])
        with self.assertLogs(logger) as cm:
            try:
                raise ValueError("value error")
            except Exception as e:
                logger.error(e)
        self.assertEqual(cm.output, ['ERROR:logger:value error'])

    def test_get_logger_twice_should_same(self):
        logger = get_logger("logger", file_level=['INFO', 'ERROR'])
        twice = get_logger("logger")
        self.assertEqual(twice, logger)

    def test_get_logger_diff(self):
        logger = get_logger("logger", file_level=['INFO', 'ERROR'])
        other = get_logger()
        self.assertNotEqual(other, logger)

    def test_same_file_hanlder(self):
        service_logger = get_logger('service', to_std=True, file_name=['service.log'], file_level=['INFO'],
                                    append_rank_dir=False)
        service_logger_without_std = get_logger('service_logger_without_std', to_std=False, file_name=['service.log'],
                                                file_level=['INFO'], append_rank_dir=False)
        self.assertEqual(len(service_logger.handlers), 2)
        self.assertEqual(len(service_logger_without_std.handlers), 1)
        self.assertIn(service_logger_without_std.handlers[0], service_logger.handlers)


class TestGetLogger(TestCase):

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.exists(CONFIG_TEST_BASE_DIR):
            shutil.rmtree(CONFIG_TEST_BASE_DIR, ignore_errors=True)

    @pytest.fixture(autouse=True)
    def _pass_fixture(self, capsys):
        self.capsys = capsys

    def test_get_logger(self):
        logger = get_logger('aicc', file_name=['aicc.INFO.log', 'aicc.ERROR.log'], file_level=['INFO', 'ERROR'])
        logger.debug(LOG_MESSAGE.get('debug'))
        logger.info(LOG_MESSAGE.get('info'))
        logger.warning(LOG_MESSAGE.get('warning'))
        logger.error(LOG_MESSAGE.get('error'))
        logger.critical(LOG_MESSAGE.get('critical'))

        captured = self.capsys.readouterr()
        out = captured.out

        self.assertNotIn(LOG_MESSAGE.get('debug'), out)
        self.assertIn(LOG_MESSAGE.get('info'), out)
        self.assertIn(LOG_MESSAGE.get('warning'), out)
        self.assertIn(LOG_MESSAGE.get('error'), out)
        self.assertIn(LOG_MESSAGE.get('critical'), out)
        log_dir_file_list = os.listdir(CONFIG_TEST_LOG_DIR)
        self.assertIn('aicc.ERROR.log', log_dir_file_list)
        self.assertIn('aicc.INFO.log', log_dir_file_list)

        with open(os.path.join(CONFIG_TEST_LOG_DIR, 'aicc.INFO.log'), 'r') as f:
            content = f.read()
            self.assertNotIn(LOG_MESSAGE.get('debug'), content)
            self.assertIn(LOG_MESSAGE.get('info'), content)
            self.assertIn(LOG_MESSAGE.get('warning'), content)
            self.assertIn(LOG_MESSAGE.get('error'), content)
            self.assertIn(LOG_MESSAGE.get('critical'), content)

        with open(os.path.join(CONFIG_TEST_LOG_DIR, 'aicc.ERROR.log'), 'r') as f:
            content = f.read()
            self.assertNotIn(LOG_MESSAGE.get('debug'), content)
            self.assertNotIn(LOG_MESSAGE.get('info'), content)
            self.assertNotIn(LOG_MESSAGE.get('warning'), content)
            self.assertIn(LOG_MESSAGE.get('error'), content)
            self.assertIn(LOG_MESSAGE.get('critical'), content)

    def test_get_logger_without_std(self):
        service_logger_without_std = get_logger('service_logger_without_std', to_std=False, file_name=['service.log'],
                                                file_level=['INFO'], append_rank_dir=False)
        service_logger_without_std.debug(LOG_MESSAGE.get('debug'))
        service_logger_without_std.info(LOG_MESSAGE.get('info'))
        service_logger_without_std.warning(LOG_MESSAGE.get('warning'))
        service_logger_without_std.error(LOG_MESSAGE.get('error'))
        service_logger_without_std.critical(LOG_MESSAGE.get('critical'))

        captured = self.capsys.readouterr()
        out = captured.out

        self.assertNotIn(LOG_MESSAGE.get('debug'), out)
        self.assertNotIn(LOG_MESSAGE.get('info'), out)
        self.assertNotIn(LOG_MESSAGE.get('warning'), out)
        self.assertNotIn(LOG_MESSAGE.get('error'), out)
        self.assertNotIn(LOG_MESSAGE.get('critical'), out)

        log_dir_file_list = os.listdir(CONFIG_TEST_BASE_DIR)
        self.assertIn('service.log', log_dir_file_list)

        with open(os.path.join(CONFIG_TEST_BASE_DIR, 'service.log'), 'r') as f:
            content = f.read()
            self.assertNotIn(LOG_MESSAGE.get('debug'), content)
            self.assertIn(LOG_MESSAGE.get('info'), content)
            self.assertIn(LOG_MESSAGE.get('warning'), content)
            self.assertIn(LOG_MESSAGE.get('error'), content)
            self.assertIn(LOG_MESSAGE.get('critical'), content)

    def test_get_service_logger(self):
        service_logger = get_logger('service', to_std=True, file_name=['service.log'], file_level=['INFO'],
                                    append_rank_dir=False)
        service_logger.debug(LOG_MESSAGE.get('debug'))
        service_logger.info(LOG_MESSAGE.get('info'))
        service_logger.warning(LOG_MESSAGE.get('warning'))
        service_logger.error(LOG_MESSAGE.get('error'))
        service_logger.critical(LOG_MESSAGE.get('critical'))

        captured = self.capsys.readouterr()
        out = captured.out

        self.assertNotIn(LOG_MESSAGE.get('debug'), out)
        self.assertIn(LOG_MESSAGE.get('info'), out)
        self.assertIn(LOG_MESSAGE.get('warning'), out)
        self.assertIn(LOG_MESSAGE.get('error'), out)
        self.assertIn(LOG_MESSAGE.get('critical'), out)

        log_dir_file_list = os.listdir(CONFIG_TEST_BASE_DIR)
        self.assertIn('service.log', log_dir_file_list)

        with open(os.path.join(CONFIG_TEST_BASE_DIR, 'service.log'), 'r') as f:
            content = f.read()
            self.assertNotIn(LOG_MESSAGE.get('debug'), content)
            self.assertIn(LOG_MESSAGE.get('info'), content)
            self.assertIn(LOG_MESSAGE.get('warning'), content)
            self.assertIn(LOG_MESSAGE.get('error'), content)
            self.assertIn(LOG_MESSAGE.get('critical'), content)
