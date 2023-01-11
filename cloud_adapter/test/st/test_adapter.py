# !/usr/bin/env python3
#
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
import logging
from unittest import TestCase

from fm.src.adapter.strategy import strategy_register


class AdapterTestCase(TestCase):
    def test_register_logical(self):
        scenario = 'modelarts'
        logging.info(strategy_register)
        strategy = strategy_register.get(scenario)
        logging.info(strategy)
