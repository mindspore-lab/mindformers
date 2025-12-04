# Copyright 2025 Huawei Technologies Co., Ltd
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
"""
Test module for testing import utils for mindformers.
"""
import os
import tempfile
import pytest

from mindformers.utils.import_utils import direct_mindformers_import


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_direct_mindformers_import_success():
    """
    Feature: Import utils
    Description: Test direct_mindformers_import.
    Expectation: Run successfully.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "__init__.py")
        with open(init_file, "w", encoding='utf-8') as f:
            f.write('''
def hello():
    return "Hello from mocked mindformers!"
            ''')
        module = direct_mindformers_import(tmpdir)
        assert module is not None
