# Copyright 2023 Huawei Technologies Co., Ltd
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
Test internal function
"""
import pytest

from mindformers.modules.transformer.transformer import _inspect_dropout


@pytest.fixture
def dropout():
    return lambda p: p

def test_all_keys_exist_new_dropout():
    """
    Feature: Test _inspect_dropout APIs
    Description: Test the new dropout apis using p as args
    Expectation: No exception
    """
    kwargs = {'keep_prob': 0.8}
    expected_result = {'p': 0.19999999999999996}
    result = _inspect_dropout(function=dropout, **kwargs)
    assert result == expected_result

def test_all_keys_exist_old_dropout():
    """
    Feature: Test _inspect_dropout APIs
    Description: Test the old dropout apis using p as args
    Expectation: No exception
    """
    kwargs = {'keep_prob': 0.8}
    expected_result = {'keep_prob': 0.8}
    result = _inspect_dropout(**kwargs)
    assert result == expected_result

def test_only_p_key_exist_new_dropout():
    """
    Feature: Test _inspect_dropout APIs
    Description: Test the new dropout apis using p as args
    Expectation: No exception
    """
    kwargs = {'p': 0.2}
    expected_result = {'p': 0.2}
    result = _inspect_dropout(function=dropout, **kwargs)
    assert result == expected_result

def test_only_p_key_exist_old_dropout():
    """
    Feature: Test _inspect_dropout APIs
    Description: Test the old dropout apis using p as args
    Expectation: No exception
    """
    kwargs = {'p': 0.2}
    expected_result = {'p': 0.2}
    result = _inspect_dropout(**kwargs)
    assert result == expected_result
