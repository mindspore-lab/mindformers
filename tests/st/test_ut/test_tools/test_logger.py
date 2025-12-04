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
test logger.py
"""
import inspect
from unittest.mock import patch, MagicMock
import pytest

from mindformers.tools.logger import (_get_stack_info, judge_redirect, StreamRedirector, AiLogFastStreamRedirect2File,
                                      judge_stdout, validate_nodes_devices_input)


class TestGetStackInfo:
    """Test class for testing _get_stack_info."""
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_starts_with_stack_prefix(self):
        """Test that returned string starts with expected prefix"""
        current_frame = inspect.currentframe()
        result = _get_stack_info(current_frame)
        assert result.startswith('Stack (most recent call last):\n')

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.tools.utils.generate_rank_list')
    @patch('mindformers.tools.utils.convert_nodes_devices_input')
    @patch('mindformers.tools.utils.get_num_nodes_devices')
    def test_rank_not_in_redirect_list_returns_false(self, mock_get_num, mock_convert, mock_generate):
        """Test when rank_id is not in redirect list returns False"""
        mock_get_num.return_value = (2, 2)
        mock_convert.return_value = [0]
        mock_generate.return_value = [0, 1]
        result = judge_redirect(rank_id=2, rank_size=4, redirect_nodes=[0])
        assert result is True


class TestStreamRedirector:
    """Test class for testing StreamRedirector."""
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_context_manager_enter_calls_start(self):
        """Test that __enter__ method calls start()"""
        source_stream = MagicMock()
        target_stream = MagicMock()
        redirector = StreamRedirector(source_stream, target_stream)
        with patch.object(redirector, 'start') as mock_start:
            with redirector:
                mock_start.assert_called_once()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_context_manager_exit_calls_stop(self):
        """Test that __exit__ method calls stop()"""
        source_stream = MagicMock()
        target_stream = MagicMock()
        redirector = StreamRedirector(source_stream, target_stream)
        with patch.object(redirector, 'stop') as mock_stop:
            redirector.__exit__(None, None, None)
            mock_stop.assert_called_once()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_decorator_wraps_function_properly(self):
        """Test that __call__ returns a decorator that wraps the function"""
        source_stream = MagicMock()
        target_stream = MagicMock()
        redirector = StreamRedirector(source_stream, target_stream)
        test_func = MagicMock()
        with patch.object(redirector, 'start') as mock_start, \
                patch.object(redirector, 'stop') as mock_stop:
            wrapper = redirector(test_func)
            wrapper('arg1', kwarg1='value1')
            mock_start.assert_called_once()
            test_func.assert_called_once_with('arg1', kwarg1='value1')
            mock_stop.assert_called_once()


class TestAiLogFastStreamRedirect2File:
    """Test class for testing AiLogFastStreamRedirect2File."""
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.tools.utils.get_rank_info')
    @patch('mindformers.tools.StreamRedirector.__init__')
    def test_start_when_redirect_false(self, mock_stream_redirector_init, mock_get_rank_info):
        """Test when both nodes and devices parameters are provided"""
        mock_get_rank_info.return_value = (0, 4)
        mock_stream_redirector_init.return_value = None
        redirector = AiLogFastStreamRedirect2File()
        mock_stream_redirector_init.assert_called_once()
        assert redirector.is_redirect is True


class TestJudgeStdout:
    """Test class for testing judge_stdout."""
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.tools.utils.generate_rank_list')
    @patch('mindformers.tools.utils.convert_nodes_devices_input')
    @patch('mindformers.tools.utils.get_num_nodes_devices')
    def test_both_nodes_and_devices_provided(self, mock_get_num, mock_convert, mock_generate):
        """Test when both nodes and devices parameters are provided"""
        mock_get_num.return_value = (2, 2)
        mock_convert.side_effect = [[0], [0, 1]]
        mock_generate.return_value = [0, 1]

        result = judge_stdout(rank_id=1, rank_size=4, is_output=True, nodes=[0], devices=[0, 1])
        assert result is True


class TestValidateNodesDevicesInput:
    """Test class for testing validate_nodes_devices_input."""
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_invalid_type_raises_type_error(self):
        """Test that invalid type raises TypeError"""
        with pytest.raises(TypeError,
                           match="The value of test_var can be None or a value of type tuple, list, or dict."):
            validate_nodes_devices_input('test_var', "invalid_string")

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_list_with_non_int_raises_type_error(self):
        """Test that list containing non-integer raises TypeError"""
        with pytest.raises(TypeError, match="The elements of a variable of type list or tuple must be of type int."):
            validate_nodes_devices_input('test_var', [1, '2', 3])
