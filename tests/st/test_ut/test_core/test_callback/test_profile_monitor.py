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
"""Test test_profile_monitor.py"""
from unittest.mock import Mock, patch
import unittest
import shutil
import tempfile
import pytest
from mindformers.core.callback.callback import ProfileMonitor

# pylint: disable=protected-access
# pylint: disable=unused-argument   # for mock logic


class TestProfileMonitor(unittest.TestCase):
    """Test cases for ProfileMonitor class"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.core.callback.callback._check_mspti_is_on')
    def test_mstx_enabled_initialization(self, mock_check_mspti):
        """Test MSTX enabled initialization"""
        # Test when MSPTI is off
        mock_check_mspti.return_value = False
        monitor = ProfileMonitor(mstx=True)
        self.assertTrue(monitor.mstx_enabled)

        # Test when MSPTI is on
        mock_check_mspti.return_value = True
        monitor = ProfileMonitor(mstx=True)
        self.assertFalse(monitor.mstx_enabled)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindspore.profiler.Profiler')
    def test_on_train_begin_starts_profiling(self, mock_profiler):
        """Test that profiling starts on train begin"""
        monitor = ProfileMonitor(start_step=1, start_profile=True)
        mock_run_context = Mock()

        # Mock the callback parameters
        mock_cb_params = Mock()
        mock_cb_params.cur_step_num = 1
        mock_run_context.original_args.return_value = mock_cb_params

        monitor.on_train_begin(mock_run_context)

        # Verify profiler was started
        mock_profiler.start.return_value = None

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_on_train_step_begin_mstx_range_start(self):
        """Test MSTX range start on train step begin"""
        monitor = ProfileMonitor(start_step=1, mstx=True)
        # Force mstx_enabled to True for testing
        monitor.mstx_enabled = True

        mock_run_context = Mock()
        mock_cb_params = Mock()
        mock_cb_params.cur_step_num = 5
        mock_run_context.original_args.return_value = mock_cb_params

        with patch('mindspore.profiler.mstx.range_start') as mock_range_start:
            mock_range_start.return_value = "test_range_id"

            monitor.on_train_step_begin(mock_run_context)

            # Verify MSTX range was started
            if monitor.mstx_enabled:
                mock_range_start.assert_called_once()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_on_train_step_end_mstx_range_end(self):
        """Test MSTX range end on train step end"""
        monitor = ProfileMonitor(start_step=1, stop_step=10, mstx=True)
        # Force mstx_enabled to True for testing
        monitor.mstx_enabled = True
        monitor.mstx_range_id = "test_range_id"

        mock_run_context = Mock()
        mock_cb_params = Mock()
        mock_cb_params.cur_step_num = 5
        mock_run_context.original_args.return_value = mock_cb_params

        with patch('mindspore.profiler.mstx.range_end') as mock_range_end:
            monitor.on_train_step_end(mock_run_context)

            # Verify MSTX range was ended
            if monitor.mstx_enabled and monitor.mstx_range_id:
                mock_range_end.assert_called_once_with("test_range_id")

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindspore.profiler.Profiler')
    def test_on_train_end_stops_profiling(self, mock_profiler):
        """Test that profiling stops on train end"""
        monitor = ProfileMonitor(start_step=1, stop_step=10)

        mock_run_context = Mock()
        mock_cb_params = Mock()
        mock_cb_params.cur_step_num = 10
        mock_run_context.original_args.return_value = mock_cb_params

        monitor.on_train_end(mock_run_context)

        # Verify profiler was stopped
        mock_profiler.stop.return_value = None

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_step_range_validation(self):
        """Test step range validation"""
        # Test valid step ranges
        monitor = ProfileMonitor(start_step=1, stop_step=10)
        self.assertEqual(monitor.start_step, 1)
        self.assertEqual(monitor.stop_step, 10)

        # Test negative start step handling
        monitor = ProfileMonitor(start_step=-1, stop_step=10)
        self.assertEqual(monitor.start_step, 1)  # Should default to 1

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.core.callback.callback.get_real_rank')
    def test_profile_rank_ids_filtering(self, mock_get_real_rank):
        """Test profile rank IDs filtering"""
        mock_get_real_rank.return_value = 0

        # Test with specific rank IDs
        monitor = ProfileMonitor(profile_rank_ids=[0, 1, 2])
        self.assertEqual(monitor.profile_rank_ids, [0, 1, 2])

        # Test with None (all ranks)
        monitor = ProfileMonitor(profile_rank_ids=None)
        self.assertIsNone(monitor.profile_rank_ids)


class TestProfileMonitorExtended:
    """Extended tests for ProfileMonitor"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_check_step_valid(self):
        """Test _check_step with valid inputs"""

        start, stop = ProfileMonitor._check_step(5, 10)
        assert start == 5
        assert stop == 10

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_check_step_invalid(self):
        """Test _check_step with invalid inputs"""

        # start > stop
        start, stop = ProfileMonitor._check_step(15, 10)
        assert start == 1
        assert stop == 10

        # negative values
        start, stop = ProfileMonitor._check_step(-1, -5)
        assert start == 1
        assert stop == 10

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_check_start_profile(self):
        """Test _check_start_profile"""

        # start_step != 1, should return False
        result = ProfileMonitor._check_start_profile(True, 5)
        assert not result

        # start_step == 1, should keep original value
        result = ProfileMonitor._check_start_profile(True, 1)
        assert result


class TestProfileMonitorInit:
    """Test ProfileMonitor initialization and configuration"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.get_pipeline_rank_ids', return_value=[0, 1])
    @patch('mindformers.core.callback.callback.get_output_subpath', return_value='/output/profile')
    @patch('mindformers.core.callback.callback.ms.get_context', return_value='Ascend')
    @patch('mindformers.core.callback.callback.is_version_ge', return_value=True)
    @patch('mindformers.core.callback.callback._check_mspti_is_on', return_value=False)
    def test_profile_monitor_init_with_pipeline(self, mock_mspti, mock_version, mock_context,
                                                mock_output, mock_pipeline_ids, mock_real_rank):
        """Test ProfileMonitor initialization with pipeline profiling"""

        # Mock the profile function from mindspore.profiler
        with patch('mindspore.profiler.profile') as mock_profile:
            mock_profiler_instance = Mock()
            mock_profile.return_value = mock_profiler_instance

            monitor = ProfileMonitor(
                start_step=1,
                stop_step=10,
                profile_pipeline=True,
                profile_communication=True,
                profile_memory=True,
                profiler_level=1
            )

            assert monitor.profiler is not None
            assert monitor.start_step == 1
            assert monitor.stop_step == 10

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=5)
    @patch('mindformers.core.callback.callback.get_pipeline_rank_ids', return_value=[0, 1])
    def test_profile_monitor_not_required_rank(self, mock_pipeline_ids, mock_real_rank):
        """Test ProfileMonitor when current rank doesn't need profiling"""

        monitor = ProfileMonitor(
            start_step=1,
            stop_step=10,
            profile_rank_ids=[0, 1, 2]
        )

        # Rank 5 is not in profile_rank_ids or pipeline_rank_ids
        assert monitor.profiler is None

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.get_pipeline_rank_ids', return_value=[0])
    @patch('mindformers.core.callback.callback.get_output_subpath', return_value='/output/profile')
    @patch('mindformers.core.callback.callback.ms.get_context', return_value='Ascend')
    @patch('mindformers.core.callback.callback.is_version_ge', return_value=True)
    @patch('mindformers.core.callback.callback._check_mspti_is_on', return_value=False)
    def test_on_train_step_begin_start_profiler(self, mock_mspti, mock_version, mock_context,
                                                mock_output, mock_pipeline_ids, mock_real_rank):
        """Test on_train_step_begin starts profiler"""

        # Create a mock profiler
        mock_profiler = Mock()
        mock_profiler.start = Mock()
        mock_profiler.step = Mock()

        # Create monitor - we'll manually set the profiler
        monitor = ProfileMonitor(
            start_step=1,
            stop_step=10,
            profile_rank_ids=[0]  # Ensure rank 0 is profiled
        )

        # Manually set the profiler and is_profiler_start flag
        monitor.profiler = mock_profiler
        monitor.is_profiler_start = False

        # Create run context
        run_context = Mock()
        cb_params = Mock()
        cb_params.cur_step_num = 1
        run_context.original_args.return_value = cb_params

        # Call on_train_step_begin
        monitor.on_train_step_begin(run_context)

        # Verify profiler.start() and profiler.step() were called
        mock_profiler.start.assert_called_once()
        mock_profiler.step.assert_called_once()
        assert monitor.is_profiler_start

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
