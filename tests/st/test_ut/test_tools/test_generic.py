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
test generic.py
"""
import tempfile
from unittest.mock import MagicMock
import os
import re
import pytest

from mindformers.tools.generic import (working_or_temp_dir, add_model_info_to_auto_map, experimental_mode_func_checker,
                                       is_experimental_mode)


class TestWorkingOrTempDir:
    """ A test class for testing working_or_temp_dir."""
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_working_dir_without_temp(self):
        """Test using working directory when use_temp_dir is False"""
        with tempfile.TemporaryDirectory() as temp_working_dir:
            with working_or_temp_dir(temp_working_dir, use_temp_dir=False) as result_dir:
                assert result_dir == temp_working_dir
                assert os.path.exists(result_dir)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_working_dir_with_temp(self):
        """Test using temporary directory when use_temp_dir is True"""
        with tempfile.TemporaryDirectory() as temp_working_dir:
            with working_or_temp_dir(temp_working_dir, use_temp_dir=True) as result_dir:
                assert result_dir != temp_working_dir


class TestAddModelInfoToAutoMap:
    """ A test class for testing add_model_info_to_auto_map."""
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_add_model_info_with_string_values(self):
        """Test with string values in auto_map"""
        auto_map = {
            "key1": "value1",
            "key2": "value2",
            "key3": None
        }
        repo_id = "my_repo"
        result = add_model_info_to_auto_map(auto_map, repo_id)
        expected = {
            "key1": "my_repo--value1",
            "key2": "my_repo--value2",
            "key3": None
        }
        assert result == expected

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_add_model_info_with_list_values(self):
        """Test with list values in auto_map"""
        auto_map = {
            "key1": ["value1", "value2"],
            "key2": [None, "value3"],
            "key3": "single_value"
        }
        repo_id = "my_repo"
        result = add_model_info_to_auto_map(auto_map, repo_id)
        expected = {
            "key1": ["my_repo--value1", "my_repo--value2"],
            "key2": [None, "my_repo--value3"],
            "key3": "my_repo--single_value"
        }
        assert result == expected

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_add_model_info_with_existing_dashes(self):
        """Test with values that already contain dashes"""
        auto_map = {
            "key1": "existing--value",
            "key2": ["normal_value", "existing--value"],
            "key3": None
        }
        repo_id = "my_repo"
        result = add_model_info_to_auto_map(auto_map, repo_id)
        expected = {
            "key1": "existing--value",
            "key2": ["my_repo--normal_value", "existing--value"],
            "key3": None
        }
        assert result == expected

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_add_model_info_empty_auto_map(self):
        """Test with empty auto_map"""
        auto_map = {}
        repo_id = "my_repo"
        result = add_model_info_to_auto_map(auto_map, repo_id)
        assert not result


class TestExperimentalModeFuncChecker:
    """ A test class for testing experimental_mode_func_checker."""
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_decorator_success_case(self):
        """Test decorator when function executes successfully"""
        mock_cls = MagicMock()
        mock_cls.__name__ = "TestClass"

        @experimental_mode_func_checker()
        def test_function(x, y):
            return x + y

        result = test_function(2, 3)
        assert result == 5

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_decorator_with_custom_error_message(self):
        """Test decorator with custom error message"""
        mock_cls = MagicMock()
        mock_cls.__name__ = "TestClass"
        custom_msg = "Custom error message"

        @experimental_mode_func_checker(custom_err_msg=custom_msg)
        def test_function(cls, x, y):
            raise ValueError("Test error")

        with pytest.raises(RuntimeError) as exc_info:
            test_function(mock_cls, 2, 3)

        error_str = str(exc_info.value)
        assert "Error occurred when executing function test_function" in error_str
        assert custom_msg in error_str
        assert "You are using TestClass in experimental mode" in error_str
        assert isinstance(exc_info.value.__cause__, ValueError)


class TestIsExperimentalMode:
    """ A test class for testing is_experimental_mode."""
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_path(self):
        """Test with non-string path parameter"""
        with pytest.raises(ValueError, match=re.escape(
                "param 'path' in AutoConfig.from_pretrained() must be str, but got <class 'int'>")):
            is_experimental_mode(123)
        result = is_experimental_mode("some/path/that/does/not/exist")
        assert result is True
        result = is_experimental_mode("mindspore/some/path")
        assert result is False
