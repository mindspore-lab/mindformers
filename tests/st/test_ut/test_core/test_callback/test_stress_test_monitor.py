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
"""Test callback.py using pytest framework."""
import os
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pytest

from mindformers.core.callback.callback import StressTestModelMonitor

# pylint: disable=unused-argument   # for mock logic


class TestStressTestModelMonitorBasic:
    """Test StressTestModelMonitor basic methods"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindspore.communication.get_local_rank_size', return_value=8)
    @patch('os.getenv')
    def test_get_value_from_line(self, mock_getenv, mock_rank_size):
        """Test get_value_from_line method"""

        model_dir = tempfile.mkdtemp()
        dataset_dir = tempfile.mkdtemp()

        # Mock MS_SCHED_PORT environment variable
        def getenv_side_effect(key, default=None):
            if key == "MS_SCHED_PORT":
                return "8118"  # Return a valid port number as string
            return default

        mock_getenv.side_effect = getenv_side_effect

        monitor = StressTestModelMonitor(
            interval_steps=10,
            stress_model_dir=model_dir,
            stress_dataset_dir=dataset_dir
        )

        line = "loss: 0.5234, global_norm: [1.234]"
        loss = monitor.get_value_from_line(line, r"loss: (\d+\.\d+)")
        assert loss == 0.5234

        global_norm = monitor.get_value_from_line(line, r"global_norm: \[(\d+\.\d+)\]")
        assert global_norm == 1.234

        # No match
        result = monitor.get_value_from_line(line, r"notfound: (\d+\.\d+)")
        assert result is None


class TestStressTestModelMonitorMethods:
    """Test StressTestModelMonitor methods"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_rank', return_value=0)
    @patch('mindformers.core.callback.callback.os.getenv')
    @patch('mindformers.core.callback.callback.os.path.exists', return_value=True)
    @patch('mindformers.core.callback.callback.ms.communication.get_local_rank_size',
           return_value=8)
    def test_on_train_step_end_skip(self, mock_local_rank, mock_exists, mock_getenv, mock_get_rank):
        """Test on_train_step_end when interval not reached"""

        def getenv_side_effect(key, default=None):
            return "8118" if key == "MS_SCHED_PORT" else default

        mock_getenv.side_effect = getenv_side_effect

        monitor = StressTestModelMonitor(
            interval_steps=100,
            stress_model_dir='/path/to/model',
            stress_dataset_dir='/path/to/dataset'
        )

        monitor.last_checked_step = 0
        monitor.check_stress_test_model = Mock()

        run_context = Mock()
        cb_params = Mock()
        cb_params.cur_step_num = 50  # Less than interval
        run_context.original_args.return_value = cb_params

        monitor.on_train_step_end(run_context)

        # Should not call check_stress_test_model
        monitor.check_stress_test_model.assert_not_called()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_rank', return_value=0)
    @patch('mindformers.core.callback.callback.os.getenv')
    @patch('mindformers.core.callback.callback.os.path.exists', return_value=True)
    @patch('mindformers.core.callback.callback.ms.communication.get_local_rank_size',
           return_value=8)
    def test_extract_interval_step_results_empty(
            self, mock_local_rank, mock_exists, mock_getenv, mock_get_rank):
        """Test extract_interval_step_results with no matching intervals"""

        def getenv_side_effect(key, default=None):
            return "8118" if key == "MS_SCHED_PORT" else default

        mock_getenv.side_effect = getenv_side_effect

        monitor = StressTestModelMonitor(
            interval_steps=100,
            stress_model_dir='/path/to/model',
            stress_dataset_dir='/path/to/dataset',
            compare_interval_steps=1000  # Very large interval
        )

        # Create a temporary log file with few steps
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            log_line1 = "{Epoch:[  1], step:[  10/ 100], loss: 2.5, global_norm: [1.2]}"
            f.write(f"2024-01-01 10:00:00 - INFO - {log_line1}\n")
            log_line2 = "{Epoch:[  1], step:[  20/ 100], loss: 2.3, global_norm: [1.1]}"
            f.write(f"2024-01-01 10:01:00 - INFO - {log_line2}\n")
            log_file = f.name

        try:
            results, global_step = monitor.extract_interval_step_results(log_file)
            # Should return None when interval is too large
            assert results is None
            assert global_step == 20
        finally:
            os.remove(log_file)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_rank', return_value=0)
    @patch('mindformers.core.callback.callback.os.getenv')
    @patch('mindformers.core.callback.callback.os.path.exists', return_value=True)
    @patch('mindformers.core.callback.callback.ms.communication.get_local_rank_size',
           return_value=8)
    def test_compare_gathered_results_consistent(
            self, mock_local_rank, mock_exists, mock_getenv, mock_get_rank):
        """Test compare_gathered_results with consistent results"""

        def getenv_side_effect(key, default=None):
            return "8118" if key == "MS_SCHED_PORT" else default

        mock_getenv.side_effect = getenv_side_effect

        monitor = StressTestModelMonitor(
            interval_steps=100,
            stress_model_dir='/path/to/model',
            stress_dataset_dir='/path/to/dataset'
        )

        # Create consistent results from multiple ranks
        gathered_results = np.array([
            [[1, 10, 2.5, 1.2]],
            [[1, 10, 2.5, 1.2]],
            [[1, 10, 2.5, 1.2]],
            [[1, 10, 2.5, 1.2]]
        ])

        result = monitor.compare_gathered_results(gathered_results)

        # Should return True for consistent results
        assert result

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_rank', return_value=0)
    @patch('mindformers.core.callback.callback.os.getenv')
    @patch('mindformers.core.callback.callback.os.path.exists', return_value=True)
    @patch('mindformers.core.callback.callback.ms.communication.get_local_rank_size',
           return_value=8)
    def test_compare_gathered_results_inconsistent(
            self, mock_local_rank, mock_exists, mock_getenv, mock_get_rank):
        """Test compare_gathered_results with inconsistent results"""

        def getenv_side_effect(key, default=None):
            return "8118" if key == "MS_SCHED_PORT" else default

        mock_getenv.side_effect = getenv_side_effect

        monitor = StressTestModelMonitor(
            interval_steps=100,
            stress_model_dir='/path/to/model',
            stress_dataset_dir='/path/to/dataset'
        )

        # Create inconsistent results from multiple ranks
        gathered_results = np.array([
            [[1, 10, 2.5, 1.2]],
            [[1, 10, 2.6, 1.3]],  # Different values
            [[1, 10, 2.5, 1.2]],
            [[1, 10, 2.5, 1.2]]
        ])

        result = monitor.compare_gathered_results(gathered_results)

        # Should return False for inconsistent results
        assert not result


class TestStressTestModelMonitorCheckStressTest:
    """Test StressTestModelMonitor.check_stress_test_model method"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_rank', return_value=0)
    @patch('mindformers.core.callback.callback.os.getenv')
    @patch('mindformers.core.callback.callback.os.path.exists', return_value=True)
    @patch('mindformers.core.callback.callback.ms.communication.get_local_rank_size',
           return_value=8)
    @patch('mindformers.core.callback.callback.logger')
    def test_check_stress_test_model_dataset_not_exists(self, mock_logger, mock_local_rank,
                                                        mock_exists, mock_getenv, mock_get_rank):
        """Test check_stress_test_model when dataset_dir doesn't exist"""

        def getenv_side_effect(key, default=None):
            return "8118" if key == "MS_SCHED_PORT" else default

        mock_getenv.side_effect = getenv_side_effect

        monitor = StressTestModelMonitor(
            interval_steps=100,
            stress_model_dir='/path/to/model',
            stress_dataset_dir='/path/to/dataset'
        )

        # Make dataset_dir check return False
        mock_exists.return_value = False
        monitor.dataset_dir = '/nonexistent/path'

        # Should return early without running stress test
        monitor.check_stress_test_model(current_step=100)

        # Should log error about dataset not found
        mock_logger.error.assert_called()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_rank', return_value=0)
    @patch('mindformers.core.callback.callback.os.getenv')
    @patch('mindformers.core.callback.callback.os.path.exists', return_value=True)
    @patch('mindformers.core.callback.callback.ms.communication.get_local_rank_size',
           return_value=8)
    @patch('mindformers.core.callback.callback.logger')
    def test_check_stress_test_model_dataset_dir_none(self, mock_logger, mock_local_rank,
                                                      mock_exists, mock_getenv, mock_get_rank):
        """Test check_stress_test_model when dataset_dir is None (line 3000)"""

        def getenv_side_effect(key, default=None):
            return "8118" if key == "MS_SCHED_PORT" else default

        mock_getenv.side_effect = getenv_side_effect

        monitor = StressTestModelMonitor(
            interval_steps=100,
            stress_model_dir='/path/to/model',
            stress_dataset_dir='/path/to/dataset'
        )

        # Set dataset_dir to None
        monitor.dataset_dir = None

        # Should return early without running stress test
        monitor.check_stress_test_model(current_step=100)

        # Should log error about dataset not found
        mock_logger.error.assert_called()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_rank', return_value=0)
    @patch('mindformers.core.callback.callback.os.getenv')
    @patch('mindformers.core.callback.callback.os.path.exists', return_value=True)
    @patch('mindformers.core.callback.callback.ms.communication.get_local_rank_size',
           return_value=8)
    @patch('mindformers.core.callback.callback.os.cpu_count', return_value=16)
    @patch('mindformers.core.callback.callback.barrier')
    @patch('mindformers.core.callback.callback.all_gather_into_tensor')
    @patch('mindformers.core.callback.callback.subprocess.Popen')
    @patch('mindformers.core.callback.callback.shlex.split', side_effect=lambda x: x.split())
    @patch('mindformers.core.callback.callback.logger')
    def test_check_stress_test_model_rank0_runs_subprocess(self, mock_logger, mock_shlex, mock_popen,
                                                           mock_all_gather, mock_barrier,
                                                           mock_cpu_count, mock_local_rank,
                                                           mock_exists, mock_getenv, mock_get_rank):
        """Test check_stress_test_model when rank_id % worker_num == 0 runs subprocess"""

        def getenv_side_effect(key, default=None):
            return "8118" if key == "MS_SCHED_PORT" else default

        mock_getenv.side_effect = getenv_side_effect

        monitor = StressTestModelMonitor(
            interval_steps=100,
            stress_model_dir='/path/to/model',
            stress_dataset_dir='/path/to/dataset',
            compare_interval_steps=None  # Skip interval comparison
        )

        # Mock subprocess behavior
        mock_process = Mock()
        mock_process.poll.side_effect = [None, 0]  # First call returns None, second returns 0
        mock_process.returncode = 0
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        mock_popen.return_value = mock_process

        # Mock all_gather_into_tensor result
        mock_tensor = Mock()
        mock_tensor.asnumpy.return_value = np.array([[1.0, 2.0], [1.0, 2.0]])
        mock_all_gather.return_value = (mock_tensor, None)

        # Mock readlog and extract methods
        monitor.readlog = Mock(return_value="Training step 10")
        monitor.extract_last_step_result = Mock(return_value=Mock())

        monitor.check_stress_test_model(current_step=100)

        # Should call Popen to start subprocess
        mock_popen.assert_called()
        # Should call barrier for synchronization
        mock_barrier.assert_called()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_rank', return_value=1)
    @patch('mindformers.core.callback.callback.os.getenv')
    @patch('mindformers.core.callback.callback.os.path.exists', return_value=True)
    @patch('mindformers.core.callback.callback.ms.communication.get_local_rank_size',
           return_value=8)
    @patch('mindformers.core.callback.callback.barrier')
    @patch('mindformers.core.callback.callback.all_gather_into_tensor')
    @patch('mindformers.core.callback.callback.logger')
    def test_check_stress_test_model_non_rank0_skips_subprocess(self, mock_logger, mock_all_gather,
                                                                mock_barrier, mock_local_rank,
                                                                mock_exists, mock_getenv, mock_get_rank):
        """Test check_stress_test_model when rank_id % worker_num != 0 skips subprocess"""

        def getenv_side_effect(key, default=None):
            return "8118" if key == "MS_SCHED_PORT" else default

        mock_getenv.side_effect = getenv_side_effect

        monitor = StressTestModelMonitor(
            interval_steps=100,
            stress_model_dir='/path/to/model',
            stress_dataset_dir='/path/to/dataset',
            compare_interval_steps=None
        )

        # Mock all_gather_into_tensor result
        mock_tensor = Mock()
        mock_tensor.asnumpy.return_value = np.array([[1.0, 2.0], [1.0, 2.0]])
        mock_all_gather.return_value = (mock_tensor, None)

        monitor.extract_last_step_result = Mock(return_value=Mock())

        monitor.check_stress_test_model(current_step=100)

        # Should call barrier (synchronization happens regardless of rank)
        mock_barrier.assert_called()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_rank', return_value=0)
    @patch('mindformers.core.callback.callback.os.getenv')
    @patch('mindformers.core.callback.callback.os.path.exists', return_value=True)
    @patch('mindformers.core.callback.callback.ms.communication.get_local_rank_size',
           return_value=8)
    @patch('mindformers.core.callback.callback.os.cpu_count', return_value=16)
    @patch('mindformers.core.callback.callback.barrier')
    @patch('mindformers.core.callback.callback.all_gather_into_tensor')
    @patch('mindformers.core.callback.callback.subprocess.Popen')
    @patch('mindformers.core.callback.callback.shlex.split', side_effect=lambda x: x.split())
    @patch('mindformers.core.callback.callback.logger')
    def test_check_stress_test_model_with_compare_interval_steps(self, mock_logger, mock_shlex, mock_popen,
                                                                 mock_all_gather, mock_barrier,
                                                                 mock_cpu_count, mock_local_rank,
                                                                 mock_exists, mock_getenv, mock_get_rank):
        """Test check_stress_test_model with compare_interval_steps set"""

        def getenv_side_effect(key, default=None):
            return "8118" if key == "MS_SCHED_PORT" else default

        mock_getenv.side_effect = getenv_side_effect

        monitor = StressTestModelMonitor(
            interval_steps=100,
            stress_model_dir='/path/to/model',
            stress_dataset_dir='/path/to/dataset',
            compare_interval_steps=10  # Set interval comparison
        )

        # Mock subprocess behavior
        mock_process = Mock()
        mock_process.poll.side_effect = [None, 0]
        mock_process.returncode = 0
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        mock_popen.return_value = mock_process

        # Mock all_gather_into_tensor result for interval comparison
        mock_interval_tensor = Mock()
        mock_interval_tensor.asnumpy.return_value = np.array([[[1, 10, 2.5, 1.2]], [[1, 10, 2.5, 1.2]]])
        mock_last_tensor = Mock()
        mock_last_tensor.asnumpy.return_value = np.array([[1.0, 2.0], [1.0, 2.0]])
        mock_all_gather.side_effect = [(mock_interval_tensor, None), (mock_last_tensor, None)]

        # Mock extract methods to return valid results
        mock_interval_result = Mock()
        monitor.extract_interval_step_results = Mock(return_value=(mock_interval_result, 100))
        monitor.extract_last_step_result = Mock(return_value=Mock())
        monitor.readlog = Mock(return_value="Training step 10")
        monitor.compare_gathered_results = Mock(return_value=True)

        monitor.check_stress_test_model(current_step=100)

        # Should call compare_gathered_results for interval comparison
        monitor.compare_gathered_results.assert_called()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_rank', return_value=0)
    @patch('mindformers.core.callback.callback.os.getenv')
    @patch('mindformers.core.callback.callback.os.path.exists', return_value=True)
    @patch('mindformers.core.callback.callback.ms.communication.get_local_rank_size',
           return_value=8)
    @patch('mindformers.core.callback.callback.os.cpu_count', return_value=16)
    @patch('mindformers.core.callback.callback.barrier')
    @patch('mindformers.core.callback.callback.all_gather_into_tensor')
    @patch('mindformers.core.callback.callback.subprocess.Popen')
    @patch('mindformers.core.callback.callback.shlex.split', side_effect=lambda x: x.split())
    @patch('mindformers.core.callback.callback.logger')
    def test_check_stress_test_model_interval_results_none(self, mock_logger, mock_shlex, mock_popen,
                                                           mock_all_gather, mock_barrier,
                                                           mock_cpu_count, mock_local_rank,
                                                           mock_exists, mock_getenv, mock_get_rank):
        """Test check_stress_test_model when interval_results is None"""

        def getenv_side_effect(key, default=None):
            return "8118" if key == "MS_SCHED_PORT" else default

        mock_getenv.side_effect = getenv_side_effect

        monitor = StressTestModelMonitor(
            interval_steps=100,
            stress_model_dir='/path/to/model',
            stress_dataset_dir='/path/to/dataset',
            compare_interval_steps=1000  # Large interval
        )

        # Mock subprocess behavior
        mock_process = Mock()
        mock_process.poll.side_effect = [None, 0]
        mock_process.returncode = 0
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        mock_popen.return_value = mock_process

        # Mock all_gather_into_tensor result
        mock_tensor = Mock()
        mock_tensor.asnumpy.return_value = np.array([[1.0, 2.0], [1.0, 2.0]])
        mock_all_gather.return_value = (mock_tensor, None)

        # Mock extract_interval_step_results to return None (interval too large)
        monitor.extract_interval_step_results = Mock(return_value=(None, 50))
        monitor.extract_last_step_result = Mock(return_value=Mock())
        monitor.readlog = Mock(return_value="Training step 10")

        monitor.check_stress_test_model(current_step=100)

        # Should log warning about interval being larger than total steps
        mock_logger.warning.assert_called()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_rank', return_value=0)
    @patch('mindformers.core.callback.callback.os.getenv')
    @patch('mindformers.core.callback.callback.os.path.exists', return_value=True)
    @patch('mindformers.core.callback.callback.ms.communication.get_local_rank_size',
           return_value=8)
    @patch('mindformers.core.callback.callback.os.cpu_count', return_value=16)
    @patch('mindformers.core.callback.callback.barrier')
    @patch('mindformers.core.callback.callback.all_gather_into_tensor')
    @patch('mindformers.core.callback.callback.subprocess.Popen')
    @patch('mindformers.core.callback.callback.shlex.split', side_effect=lambda x: x.split())
    @patch('mindformers.core.callback.callback.logger')
    def test_check_stress_test_model_results_match(self, mock_logger, mock_shlex, mock_popen,
                                                   mock_all_gather, mock_barrier,
                                                   mock_cpu_count, mock_local_rank,
                                                   mock_exists, mock_getenv, mock_get_rank):
        """Test check_stress_test_model when all results match"""

        def getenv_side_effect(key, default=None):
            return "8118" if key == "MS_SCHED_PORT" else default

        mock_getenv.side_effect = getenv_side_effect

        monitor = StressTestModelMonitor(
            interval_steps=100,
            stress_model_dir='/path/to/model',
            stress_dataset_dir='/path/to/dataset',
            compare_interval_steps=None
        )

        # Mock subprocess behavior
        mock_process = Mock()
        mock_process.poll.side_effect = [None, 0]
        mock_process.returncode = 0
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        mock_popen.return_value = mock_process

        # All results match
        mock_tensor = Mock()
        mock_tensor.asnumpy.return_value = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        mock_all_gather.return_value = (mock_tensor, None)

        monitor.extract_last_step_result = Mock(return_value=Mock())
        monitor.readlog = Mock(return_value="Training step 10")

        monitor.check_stress_test_model(current_step=100)

        # Should log STRESS TEST PASSED
        info_calls = [str(call) for call in mock_logger.info.call_args_list]
        passed_logged = any('STRESS TEST PASSED' in str(call) for call in info_calls)
        assert passed_logged

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_rank', return_value=0)
    @patch('mindformers.core.callback.callback.os.getenv')
    @patch('mindformers.core.callback.callback.os.path.exists', return_value=True)
    @patch('mindformers.core.callback.callback.ms.communication.get_local_rank_size',
           return_value=8)
    @patch('mindformers.core.callback.callback.os.cpu_count', return_value=16)
    @patch('mindformers.core.callback.callback.barrier')
    @patch('mindformers.core.callback.callback.all_gather_into_tensor')
    @patch('mindformers.core.callback.callback.subprocess.Popen')
    @patch('mindformers.core.callback.callback.shlex.split', side_effect=lambda x: x.split())
    @patch('mindformers.core.callback.callback.logger')
    def test_check_stress_test_model_results_mismatch(self, mock_logger, mock_shlex, mock_popen,
                                                      mock_all_gather, mock_barrier,
                                                      mock_cpu_count, mock_local_rank,
                                                      mock_exists, mock_getenv, mock_get_rank):
        """Test check_stress_test_model when results don't match"""

        def getenv_side_effect(key, default=None):
            return "8118" if key == "MS_SCHED_PORT" else default

        mock_getenv.side_effect = getenv_side_effect

        monitor = StressTestModelMonitor(
            interval_steps=100,
            stress_model_dir='/path/to/model',
            stress_dataset_dir='/path/to/dataset',
            compare_interval_steps=None
        )

        # Mock subprocess behavior
        mock_process = Mock()
        mock_process.poll.side_effect = [None, 0]
        mock_process.returncode = 0
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        mock_popen.return_value = mock_process

        # Results don't match - different values
        mock_tensor = Mock()
        mock_tensor.asnumpy.return_value = np.array([[1.0, 2.0], [1.5, 2.5], [1.0, 2.0]])
        mock_all_gather.return_value = (mock_tensor, None)

        monitor.extract_last_step_result = Mock(return_value=Mock())
        monitor.readlog = Mock(return_value="Training step 10")

        monitor.check_stress_test_model(current_step=100)

        # Should log STRESS TEST FAILED warning
        warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
        failed_logged = any('STRESS TEST FAILED' in str(call) for call in warning_calls)
        assert failed_logged

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_rank', return_value=0)
    @patch('mindformers.core.callback.callback.os.getenv')
    @patch('mindformers.core.callback.callback.os.path.exists', return_value=True)
    @patch('mindformers.core.callback.callback.ms.communication.get_local_rank_size',
           return_value=8)
    @patch('mindformers.core.callback.callback.os.cpu_count', return_value=16)
    @patch('mindformers.core.callback.callback.barrier')
    @patch('mindformers.core.callback.callback.all_gather_into_tensor')
    @patch('mindformers.core.callback.callback.subprocess.Popen')
    @patch('mindformers.core.callback.callback.shlex.split', side_effect=lambda x: x.split())
    @patch('mindformers.core.callback.callback.logger')
    def test_check_stress_test_model_subprocess_error(self, mock_logger, mock_shlex, mock_popen,
                                                      mock_all_gather, mock_barrier,
                                                      mock_cpu_count, mock_local_rank,
                                                      mock_exists, mock_getenv, mock_get_rank):
        """Test check_stress_test_model when subprocess returns error"""

        def getenv_side_effect(key, default=None):
            return "8118" if key == "MS_SCHED_PORT" else default

        mock_getenv.side_effect = getenv_side_effect

        monitor = StressTestModelMonitor(
            interval_steps=100,
            stress_model_dir='/path/to/model',
            stress_dataset_dir='/path/to/dataset',
            compare_interval_steps=None
        )

        # Mock subprocess with error
        mock_process = Mock()
        mock_process.poll.side_effect = [None, 1]  # Returns non-zero exit code
        mock_process.returncode = 1  # Error
        mock_process.stderr = Mock()
        mock_process.stderr.read.return_value = b'Error message'
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        mock_popen.return_value = mock_process

        # Mock all_gather_into_tensor result
        mock_tensor = Mock()
        mock_tensor.asnumpy.return_value = np.array([[1.0, 2.0], [1.0, 2.0]])
        mock_all_gather.return_value = (mock_tensor, None)

        monitor.extract_last_step_result = Mock(return_value=Mock())
        monitor.readlog = Mock(return_value="Training step 10")

        monitor.check_stress_test_model(current_step=100)

        # Should log warning about subprocess error
        warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
        error_logged = any('error occurred' in str(call).lower() for call in warning_calls)
        assert error_logged

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
