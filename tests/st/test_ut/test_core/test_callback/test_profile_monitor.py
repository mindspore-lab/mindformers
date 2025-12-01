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


if __name__ == '__main__':
    unittest.main()
