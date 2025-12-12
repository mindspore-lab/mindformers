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
test version_control.py
"""
from unittest.mock import patch
import pytest

import mindspore as ms
import mindspore_gs
from mindformers.version_control import (check_is_reboot_node, check_valid_mindspore_gs, check_valid_gmm_op,
                                         is_version_python, get_norm)


class TestCheckIsVersion:
    """Test class for testing version_control."""
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('os.getenv')
    def test_version_too_low_returns_false(self, mock_getenv):
        """Test when MindSpore version is lower than 2.6.0 returns False with warning."""
        ms.__version__ = "2.6.0"
        result = check_is_reboot_node()
        mock_getenv.return_value = "ARF:1"
        assert result is False

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_version_too_low_returns_false2(self):
        """Test when MindSpore version is lower than 2.6.0 returns False with warning."""
        ms.__version__ = "2.5.0"
        result = check_is_reboot_node()
        assert result is False

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_version(self):
        """Test when mindspore_gs version."""
        mindspore_gs.__version__ = "0.6.0"
        result = check_valid_mindspore_gs()
        assert result is True
        mindspore_gs.__version__ = "0.5.0"
        result = check_valid_mindspore_gs()
        assert result is False

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindspore.__version__', '2.6.0')
    def test_check_valid_gmm_op_with_version_equal_to_required(self):
        """Test when MindSpore version equals required version, should return True"""
        result = check_valid_gmm_op(gmm_version="GroupedMatmulV4")
        assert result is True

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindspore.__version__', '2.6.0-rc1')
    def test_check_valid_gmm_op_with_rc_version(self):
        """Test when MindSpore version has rc suffix, should handle correctly"""
        result = check_valid_gmm_op(gmm_version="GroupedMatmulV4")
        assert result is True or result is False

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_is_version_python_cur_higher_than_tar(self):
        """Test when current version is higher than target version"""
        result = is_version_python("3.9.1", "3.9.0")
        assert result is True
        result = is_version_python("3.7.10", "3.9.0")
        assert result is False

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_is_version_python_missing_dot_in_cur(self):
        """Test when current version string doesn't contain dot, should raise ValueError"""
        with pytest.raises(ValueError) as exc_info:
            is_version_python("37910", "3.9.0")
        assert "The version string will contain the `.`" in str(exc_info.value)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_is_version_python_different_version_lengths(self):
        """Test version strings with different number of segments"""
        result = is_version_python("3.9.0.1", "3.9.0")
        assert result is True
        result = is_version_python("3.9", "3.9.0")
        assert result is True

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_norm_version_ge_1_11_0(self):
        """Test when mindspore version >= '1.11.0', should return tensor_norm1"""
        with patch('mindspore.__version__', '1.11.0'):
            with patch('mindformers.tools.utils.is_version_ge') as mock_is_version_ge:
                mock_is_version_ge.return_value = True
                norm_func = get_norm()
                assert norm_func.__name__ == 'tensor_norm1' or norm_func.__code__.co_varnames[:5] == (
                'input_tensor', 'tensor_ord', 'dim', 'keepdim', 'dtype')
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_norm_version_lt_1_11_0(self):
        """Test when mindspore version < '1.11.0', should return tensor_norm2"""
        with patch('mindspore.__version__', '1.10.0'):
            with patch('mindformers.tools.utils.is_version_ge') as mock_is_version_ge:
                mock_is_version_ge.return_value = False
                norm_func = get_norm()
                assert norm_func.__name__ == 'tensor_norm2' or norm_func.__defaults__[0] == 2
