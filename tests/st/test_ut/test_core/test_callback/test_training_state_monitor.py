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
from unittest.mock import Mock, patch

import numpy as np
import pytest

from mindformers.core.callback.callback import TrainingStateMonitor

# pylint: disable=protected-access
# pylint: disable=unused-argument   # for mock logic


class TestTrainingStateMonitor:
    """Test TrainingStateMonitor class"""

    def setup_method(self):
        """Set up test fixtures for each test method."""

        # Mock context.get_auto_parallel_context to return appropriate values
        def mock_get_context(key):
            if key == "pipeline_stages":
                return 1  # Return integer for pipeline_stages
            if key == "dump_local_norm_path":
                return None  # No dump path
            return None

        with patch('mindformers.core.callback.callback.context.get_auto_parallel_context',
                   side_effect=mock_get_context), \
                patch('mindformers.core.callback.callback.get_real_group_size', return_value=1), \
                patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=None):
            self.monitor = TrainingStateMonitor(
                origin_epochs=10,
                dataset_size=100,
                step_interval=1
            )
        # Initialize dump_path to None to avoid finish_pattern attribute error
        self.monitor.dump_path = None

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_init(self):
        """Test initialization"""
        assert self.monitor.origin_epochs == 10
        assert self.monitor.steps_per_epoch == 100
        assert self.monitor.step_interval == 1

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('time.time')
    def test_on_train_epoch_begin(self, mock_time):
        """Test on_train_epoch_begin"""
        mock_time.return_value = 12345.0
        run_context = Mock()
        self.monitor.on_train_epoch_begin(run_context)
        assert self.monitor.epoch_time == 12345.0
        assert self.monitor.run_context == run_context

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('time.time')
    def test_on_train_step_begin(self, mock_time):
        """Test on_train_step_begin"""
        mock_time.return_value = 67890.0
        run_context = Mock()
        run_context.original_args.return_value = Mock()
        self.monitor.on_train_step_begin(run_context)
        assert self.monitor.step_time == 67890.0
        assert self.monitor.run_context == run_context

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('time.time')
    @patch('mindformers.core.callback.callback.set_auto_parallel_context')
    @patch('mindformers.core.callback.callback.get_auto_parallel_context')
    def test_on_train_step_end_basic(self, mock_get_parallel, mock_set_parallel, mock_time):
        """Test on_train_step_end basic flow"""
        mock_time.side_effect = [1000.0, 1000.1]  # start, end

        def get_context_side_effect(attr):
            if attr == "parallel_mode":
                return "stand_alone"
            if attr == "full_batch":
                return False
            if attr == "dump_local_norm_path":
                return None
            return None

        mock_get_parallel.side_effect = get_context_side_effect

        run_context = Mock()
        cb_params = Mock()
        cb_params.cur_step_num = 1
        cb_params.batch_num = 100
        cb_params.cur_epoch_num = 1
        cb_params.dataset_sink_mode = False
        cb_params.net_outputs = Mock()  # loss
        cb_params.initial_step = 0

        # Mock get method for cb_params to behave like dict for 'initial_step'
        def mock_get(key, default=None):
            if key == 'initial_step':
                return cb_params.initial_step
            return getattr(cb_params, key, default)

        cb_params.get.side_effect = mock_get

        run_context.original_args.return_value = cb_params

        with patch.object(self.monitor, '_get_loss_output', return_value=(0.5, 1.0, 0.1)):
            self.monitor.step_time = 1000.0
            self.monitor.on_train_step_end(run_context)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.check_device_local_loss')
    def test_boundary_check(self, mock_check_loss):
        """Test _boundary_check"""
        self.monitor.check_for_nan_in_loss_and_grad = True
        cb_params = Mock()

        # Case 1: Normal
        with patch.object(self.monitor, '_get_loss_output', return_value=(0.5, 1.0, 0.1)):
            self.monitor._boundary_check(cb_params)

        # Case 2: NaN loss
        # We need to simulate np.isnan check.
        # If _get_loss_output returns nan, _check_nan_or_inf checks it
        # using np.any(np.isnan(indicator))
        with patch.object(self.monitor, '_get_loss_output', return_value=(float('nan'), 1.0, 0.1)):
            with pytest.raises(ValueError):
                self.monitor._boundary_check(cb_params)


class TestTrainingStateMonitorExtended:
    """Extended tests for TrainingStateMonitor"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=1)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=None)
    def test_global_norm_spike_detection(self, *mocks):
        """Test global norm spike detection"""

        config = {
            'global_norm_spike_threshold': 10.0,
            'global_norm_spike_count_threshold': 3
        }

        # Mock context.get_auto_parallel_context to return appropriate values
        def mock_get_context(key):
            if key == "pipeline_stages":
                return 1  # Return integer for pipeline_stages
            if key == "dump_local_norm_path":
                return None  # No dump path
            return None

        with patch('mindformers.core.callback.callback.context.get_auto_parallel_context',
                   side_effect=mock_get_context):
            monitor = TrainingStateMonitor(
                origin_epochs=10,
                dataset_size=100,
                config=config
            )

            assert monitor.global_norm_spike_threshold == 10.0
            assert monitor.global_norm_spike_count_threshold == 3

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=1)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=None)
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    def test_dump_path_initialization(self, *mocks):
        """Test dump_path initialization"""

        # Mock context.get_auto_parallel_context to return appropriate values
        def mock_get_context(key):
            if key == "pipeline_stages":
                return 1  # Return integer for pipeline_stages
            if key == "dump_local_norm_path":
                return "/tmp/test_path"  # Return a path to trigger dump_path initialization
            return None

        with patch('mindformers.core.callback.callback.context.get_auto_parallel_context',
                   side_effect=mock_get_context), \
                patch('mindformers.core.callback.callback.get_auto_parallel_context',
                      side_effect=mock_get_context):
            monitor = TrainingStateMonitor(origin_epochs=10, dataset_size=100)
            assert monitor.dump_path is not None
            assert monitor.dump_path == "/tmp/test_path/rank_0"


class TestTrainingStateMonitorAbnormalGlobalNorm:
    """Test TrainingStateMonitor.abnormal_global_norm_check"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=8)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=None)
    @patch('mindformers.core.callback.callback.get_rank', return_value=0)
    @patch('mindformers.core.callback.callback.barrier_world')
    @patch('mindformers.core.callback.callback.ms.runtime.synchronize')
    @patch('mindformers.core.callback.callback.logger')
    @patch('mindformers.core.callback.callback.os.path.exists', return_value=False)
    @patch('mindformers.core.callback.callback.os.makedirs')
    @patch('builtins.open', create=True)
    @patch('mindformers.core.callback.callback.set_safe_mode_for_file_or_dir')
    @patch('mindformers.core.callback.callback.context.get_auto_parallel_context', return_value=1)
    def test_abnormal_global_norm_check_first_spike(
            self, mock_context, mock_safe_mode, mock_open, mock_makedirs,
            mock_exists, mock_logger, mock_sync,
            mock_barrier, mock_rank, *mocks):
        """Test abnormal_global_norm_check when first spike occurs"""

        config = {
            'check_for_global_norm': True,
            'global_norm_spike_threshold': 10.0,
            'global_norm_spike_count_threshold': 3
        }

        monitor = TrainingStateMonitor(
            origin_epochs=10,
            dataset_size=100,
            config=config
        )

        cb_params = Mock()
        cb_params.cur_step_num = 5
        cb_params.batch_num = 100
        cb_params.dataset_sink_mode = False
        cb_params.cur_epoch_num = 1
        cb_params.get.return_value = None

        # Mock net_outputs with high global_norm
        # Need to make global_norm support >= comparison
        mock_global_norm = Mock()
        mock_global_norm.item.return_value = 15.0
        # Mock __ge__ to support >= comparison
        mock_global_norm.__ge__ = Mock(return_value=True)
        cb_params.net_outputs = (Mock(), False, 1024.0, 0.001, mock_global_norm)

        # Should raise RuntimeError on first spike
        with pytest.raises(RuntimeError) as exc_info:
            monitor.abnormal_global_norm_check(cb_params)

        assert "TREError" in str(exc_info.value)
        mock_barrier.assert_called_once()
        mock_sync.assert_called_once()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=8)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=None)
    @patch('mindformers.core.callback.callback.logger')
    @patch('mindformers.core.callback.callback.context.get_auto_parallel_context', return_value=1)
    def test_abnormal_global_norm_check_use_skip_data(self, mock_context, mock_logger, *mocks):
        """Test abnormal_global_norm_check with use_skip_data_by_global_norm"""

        config = {
            'check_for_global_norm': False,
            'global_norm_spike_threshold': 10.0,
            'global_norm_spike_count_threshold': 3
        }

        monitor = TrainingStateMonitor(
            origin_epochs=10,
            dataset_size=100,
            config=config,
            use_skip_data_by_global_norm=True
        )

        cb_params = Mock()
        cb_params.cur_step_num = 5
        cb_params.batch_num = 100
        cb_params.dataset_sink_mode = False
        cb_params.cur_epoch_num = 1
        cb_params.get.return_value = None
        cb_params.optimizer = Mock()
        cb_params.optimizer.global_step = 5

        # Mock net_outputs with high global_norm
        mock_global_norm = Mock()
        mock_global_norm.item.return_value = 15.0
        # Mock __ge__ to support >= comparison (returns True for high norm)
        mock_global_norm.__ge__ = Mock(return_value=True)
        cb_params.net_outputs = (Mock(), False, 1024.0, 0.001, mock_global_norm)

        # First spike - should log but not raise
        monitor.abnormal_global_norm_check(cb_params)
        assert monitor.global_norm_spike_count == 1

        # Second spike
        monitor.abnormal_global_norm_check(cb_params)
        assert monitor.global_norm_spike_count == 2

        # Third spike - should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            monitor.abnormal_global_norm_check(cb_params)

        assert "consecutive times greater than threshold" in str(exc_info.value)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=8)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=None)
    @patch('mindformers.core.callback.callback.context.get_auto_parallel_context', return_value=1)
    def test_abnormal_global_norm_check_reset_count(self, mock_context, *mocks):
        """Test that global_norm_spike_count resets when norm is normal"""

        config = {
            'check_for_global_norm': False,
            'global_norm_spike_threshold': 10.0,
            'global_norm_spike_count_threshold': 3
        }

        monitor = TrainingStateMonitor(
            origin_epochs=10,
            dataset_size=100,
            config=config,
            use_skip_data_by_global_norm=True
        )

        cb_params = Mock()
        cb_params.cur_step_num = 5
        cb_params.batch_num = 100
        cb_params.dataset_sink_mode = False
        cb_params.cur_epoch_num = 1
        cb_params.get.return_value = None
        cb_params.optimizer = Mock()
        cb_params.optimizer.global_step = 5

        # High global_norm
        mock_high_norm = Mock()
        mock_high_norm.item.return_value = 15.0
        # Mock __ge__ to return True (high norm >= threshold)
        mock_high_norm.__ge__ = Mock(return_value=True)
        cb_params.net_outputs = (Mock(), False, 1024.0, 0.001, mock_high_norm)

        monitor.abnormal_global_norm_check(cb_params)
        assert monitor.global_norm_spike_count == 1

        # Normal global_norm - should reset count
        mock_normal_norm = Mock()
        mock_normal_norm.item.return_value = 5.0
        # Mock __ge__ to return False (normal norm < threshold)
        mock_normal_norm.__ge__ = Mock(return_value=False)
        cb_params.net_outputs = (Mock(), False, 1024.0, 0.001, mock_normal_norm)

        monitor.abnormal_global_norm_check(cb_params)
        assert monitor.global_norm_spike_count == 0


class TestTrainingStateMonitorCalcMethods:
    """Test TrainingStateMonitor calculation methods"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=8)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=Mock())
    @patch('mindformers.core.callback.callback._get_weight_norm', return_value=2.5)
    @patch('mindformers.core.callback.callback.context.get_auto_parallel_context', return_value=1)
    def test_calc_weight_state(self, mock_context, mock_get_weight_norm, *mocks):
        """Test _calc_weight_state method"""

        config = {
            'weight_state_format': ['log', 'tensorboard']
        }

        monitor = TrainingStateMonitor(
            origin_epochs=10,
            dataset_size=100,
            config=config
        )

        cb_params = Mock()
        cb_params.cur_step_num = 5
        cb_params.network = Mock()
        cb_params.network.network = Mock()

        monitor._calc_weight_state(cb_params)

        mock_get_weight_norm.assert_called_once()
        monitor.tensor_writer.add_scalar.assert_called()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=8)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=Mock())
    @patch('mindformers.core.callback.callback.context.get_auto_parallel_context', return_value=1)
    def test_calc_throughput_linearity(self, mock_context, *mocks):
        """Test _calc_throughput_linearity method"""

        config = {
            'throughput_baseline': 100.0
        }

        monitor = TrainingStateMonitor(
            origin_epochs=10,
            dataset_size=100,
            config=config,
            global_batch_size=32
        )

        cb_params = Mock()
        cb_params.cur_step_num = 5

        # per_step_seconds = 100ms
        monitor._calc_throughput_linearity(cb_params, 100.0)

        # throughput = 32 / 8 / (100/1000) = 4 / 0.1 = 40
        # linearity = 40 / 100 = 0.4
        monitor.tensor_writer.add_scalar.assert_called()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=8)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=Mock())
    @patch('mindformers.core.callback.callback.get_device_local_loss')
    @patch('mindformers.core.callback.callback.context.get_auto_parallel_context', return_value=1)
    def test_calc_device_local_loss(self, mock_context, mock_get_loss, *mocks):
        """Test _calc_device_local_loss method"""

        config = {
            'device_local_loss_format': ['log', 'tensorboard']
        }

        monitor = TrainingStateMonitor(
            origin_epochs=10,
            dataset_size=100,
            config=config
        )

        # Mock device local loss
        mock_loss_tensor = Mock()
        mock_loss_tensor.asnumpy.return_value = np.array(0.5)
        mock_get_loss.return_value = {'lm': mock_loss_tensor}

        cb_params = Mock()
        cb_params.cur_step_num = 5

        monitor._calc_device_local_loss(cb_params)

        mock_get_loss.assert_called_once()
        monitor.tensor_writer.add_scalar.assert_called()


class TestTrainingStateMonitorStableRank:
    """Test TrainingStateMonitor stable rank calculation"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=8)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=Mock())
    @patch('mindformers.core.callback.callback.ms.runtime.empty_cache')
    @patch('mindformers.core.callback.callback.context.get_auto_parallel_context', return_value=1)
    def test_do_stable_rank(self, mock_context, mock_empty_cache, *mocks):
        """Test _do_stable_rank method"""

        config = {
            'stable_rank_config': {
                'format': ['log'],
                'step_interval': 10
            }
        }

        monitor = TrainingStateMonitor(
            origin_epochs=10,
            dataset_size=100,
            config=config
        )

        cb_params = Mock()
        cb_params.cur_step_num = 10
        cb_params.train_network = Mock()
        cb_params.train_network.network = Mock()
        cb_params.train_network.network.trainable_params.return_value = []

        with patch.object(monitor, '_calc_stable_rank'):
            monitor._do_stable_rank(cb_params)

        mock_empty_cache.assert_called_once()
        assert monitor.sr_last_print_time == 10

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=8)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=Mock())
    @patch('mindformers.core.callback.callback.context.get_auto_parallel_context')
    def test_calc_stable_rank_standalone(self, mock_get_context, *mocks):
        """Test _calc_stable_rank in standalone mode"""

        # Mock context to return 1 for pipeline_stages during init,
        # then 'stand_alone' for parallel_mode
        def context_side_effect(key):
            if key == "pipeline_stages":
                return 1
            if key == "parallel_mode":
                return "stand_alone"
            return None

        mock_get_context.side_effect = context_side_effect

        config = {
            'stable_rank_config': {
                'format': ['log'],
                'step_interval': 10
            }
        }

        monitor = TrainingStateMonitor(
            origin_epochs=10,
            dataset_size=100,
            config=config
        )

        # Mock trainable params
        mock_param = Mock()
        mock_param.name = 'layer.weight'
        mock_param.ndim = 2

        cb_params = Mock()
        cb_params.cur_step_num = 10
        cb_params.train_network = Mock()
        cb_params.train_network.network = Mock()
        cb_params.train_network.network.trainable_params.return_value = [mock_param]

        with patch.object(monitor, '_print_stable_rank'):
            monitor._calc_stable_rank(cb_params)
            monitor._print_stable_rank.assert_called_once()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=8)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=Mock())
    @patch('mindformers.core.callback.callback.context.get_auto_parallel_context')
    def test_calc_stable_rank_parallel_no_aggregation(self, mock_get_context, *mocks):
        """Test _calc_stable_rank in parallel mode without aggregation"""

        # Mock context to return 1 for pipeline_stages during init,
        # then 'semi_auto_parallel' for parallel_mode
        def context_side_effect(key):
            if key == "pipeline_stages":
                return 1
            if key == "parallel_mode":
                return "semi_auto_parallel"
            return None

        mock_get_context.side_effect = context_side_effect

        config = {
            'stable_rank_config': {
                'format': ['log'],
                'step_interval': 10,
                'do_aggregation': False
            }
        }

        monitor = TrainingStateMonitor(
            origin_epochs=10,
            dataset_size=100,
            config=config
        )

        # Mock trainable params
        mock_param = Mock()
        mock_param.name = 'layer.weight'
        mock_param.ndim = 2

        cb_params = Mock()
        cb_params.cur_step_num = 10
        cb_params.train_network = Mock()
        cb_params.train_network.network = Mock()
        cb_params.train_network.network.trainable_params.return_value = [mock_param]

        with patch.object(monitor, '_get_remove_redundancy_param_names',
                          return_value=['layer.weight']):
            with patch.object(monitor, '_print_stable_rank'):
                monitor._calc_stable_rank(cb_params)
                monitor._print_stable_rank.assert_called_once()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=8)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=Mock())
    @patch('mindformers.core.callback.callback.context.get_auto_parallel_context')
    @patch('mindformers.core.callback.callback.Tensor')
    @patch('mindformers.core.callback.callback._get_merged_param_data')
    def test_calc_stable_rank_parallel_with_aggregation(self, mock_merged_data, mock_tensor,
                                                        mock_get_context, *mocks):
        """Test _calc_stable_rank in parallel mode with aggregation"""

        # Mock context to return 1 for pipeline_stages during init,
        # then 'semi_auto_parallel' for parallel_mode
        def context_side_effect(key):
            if key == "pipeline_stages":
                return 1
            if key == "parallel_mode":
                return "semi_auto_parallel"
            return None

        mock_get_context.side_effect = context_side_effect

        config = {
            'stable_rank_config': {
                'format': ['log'],
                'step_interval': 10,
                'do_aggregation': True,
                'target': ['layer.*']
            }
        }

        monitor = TrainingStateMonitor(
            origin_epochs=10,
            dataset_size=100,
            config=config
        )

        # Mock trainable params
        mock_param = Mock()
        mock_param.name = 'layer.weight'
        mock_param.ndim = 2
        mock_param.data = Mock()
        mock_param.data.asnumpy.return_value = np.array([[1, 2], [3, 4]])

        cb_params = Mock()
        cb_params.cur_step_num = 10
        cb_params.train_network = Mock()
        cb_params.train_network.network = Mock()
        cb_params.train_network.network.trainable_params.return_value = [mock_param]
        cb_params.train_network.parameter_layout_dict = {'layer.weight': Mock()}

        mock_merged_data.return_value = Mock()

        with patch.object(monitor, '_get_remove_redundancy_param_names',
                          return_value=['layer.weight']):
            with patch.object(monitor, '_get_single_params', return_value={0: ['layer.weight']}):
                with patch.object(monitor, '_get_redundancy_removed_print', return_value=True):
                    with patch.object(monitor, '_print_stable_rank'):
                        monitor._calc_stable_rank(cb_params)
                        # Should call _print_stable_rank with merged data
                        monitor._print_stable_rank.assert_called()


class TestTrainingStateMonitorCheckSrTarget:
    """Test TrainingStateMonitor._check_sr_target"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=8)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=None)
    @patch('mindformers.core.callback.callback.context.get_auto_parallel_context', return_value=1)
    def test_check_sr_target_match(self, mock_context, *mocks):
        """Test _check_sr_target with matching pattern"""

        config = {
            'stable_rank_config': {
                'format': ['log'],
                'target': ['layer\\..*', 'attention\\..*']
            }
        }

        monitor = TrainingStateMonitor(
            origin_epochs=10,
            dataset_size=100,
            config=config
        )

        # Should match
        assert monitor._check_sr_target('layer.weight')
        assert monitor._check_sr_target('attention.query')

        # Should not match
        assert not monitor._check_sr_target('other.weight')

        # Cache should work - second call should use cached result
        assert monitor._check_sr_target('layer.weight')
        assert 'layer.weight' in monitor.sr_target_cache


class TestTrainingStateMonitorPrintStableRank:
    """Test TrainingStateMonitor._print_stable_rank and related methods"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=8)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=Mock())
    @patch('mindformers.core.callback.callback._get_stable_rank')
    @patch('mindformers.core.callback.callback.context.get_auto_parallel_context', return_value=1)
    def test_print_stable_rank_2d_tensor(
            self, mock_context, mock_get_stable_rank,
            mock_tensorboard, mock_group_size):
        """Test _print_stable_rank with 2D tensor"""

        mock_get_stable_rank.return_value = (2.5, 3.0)

        config = {
            'stable_rank_config': {
                'format': ['log', 'tensorboard'],
                'step_interval': 10
            }
        }

        monitor = TrainingStateMonitor(
            origin_epochs=10,
            dataset_size=100,
            config=config
        )

        # Create a 2D mock parameter
        mock_param = Mock()
        mock_param.ndim = 2
        mock_param.asnumpy.return_value = np.random.randn(10, 10)

        monitor._print_stable_rank('test_layer', mock_param, 10)

        mock_get_stable_rank.assert_called_once()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=8)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=Mock())
    @patch('mindformers.core.callback.callback._get_stable_rank')
    @patch('mindformers.core.callback.callback.context.get_auto_parallel_context', return_value=1)
    def test_print_stable_rank_3d_moe_all_mode(self, mock_context, mock_get_stable_rank,
                                               mock_tensorboard, mock_group_size):
        """Test _print_stable_rank with 3D tensor in MoE 'all' mode"""

        # Return arrays for multiple experts
        mock_get_stable_rank.return_value = (
            np.array([2.5, 2.6, 2.7]),
            np.array([3.0, 3.1, 3.2])
        )

        config = {
            'stable_rank_config': {
                'format': ['log'],
                'step_interval': 10,
                'moe_show_mode': 'all'
            }
        }

        monitor = TrainingStateMonitor(
            origin_epochs=10,
            dataset_size=100,
            config=config
        )

        # Create a 3D mock parameter (for MoE)
        mock_param = Mock()
        mock_param.ndim = 3
        mock_param.asnumpy.return_value = np.random.randn(3, 10, 10)

        monitor._print_stable_rank('moe_layer', mock_param, 10)

        mock_get_stable_rank.assert_called_once()


class TestTrainingStateMonitorDumpMethods:
    """Test TrainingStateMonitor dump methods"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=8)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=Mock())
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.glob.glob', return_value=[])
    def test_parse_step_no_files(self, mock_glob, mock_rank, *mocks):
        """Test _parse_step with no dump files"""

        config = {
            'dump_path': '/tmp/dump',
            'finish_pattern': 'finish_*'
        }

        # Mock both get_auto_parallel_context and context.get_auto_parallel_context
        def get_context_side_effect(key):
            if key == "dump_local_norm_path":
                return '/tmp/dump'
            if key == "pipeline_stages":
                return 1
            return None

        with patch('mindformers.core.callback.callback.get_auto_parallel_context',
                   side_effect=get_context_side_effect), \
                patch('mindformers.core.callback.callback.context.get_auto_parallel_context',
                      side_effect=get_context_side_effect):

            monitor = TrainingStateMonitor(
                origin_epochs=10,
                dataset_size=100,
                config=config
            )

        monitor.dump_key = {}
        monitor._parse_step()

        # Should not add any keys when no files found
        assert len(monitor.dump_key) == 0

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=8)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=Mock())
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.os.listdir', return_value=[])
    def test_dump_data_in_step_empty(self, mock_listdir, mock_rank, *mocks):
        """Test _dump_data_in_step with empty directory"""

        config = {
            'dump_path': '/tmp/dump'
        }

        # Mock both get_auto_parallel_context and context.get_auto_parallel_context
        def get_context_side_effect(key):
            if key == "dump_local_norm_path":
                return '/tmp/dump'
            if key == "pipeline_stages":
                return 1
            return None

        with patch('mindformers.core.callback.callback.get_auto_parallel_context',
                   side_effect=get_context_side_effect), \
                patch('mindformers.core.callback.callback.context.get_auto_parallel_context',
                      side_effect=get_context_side_effect):

            monitor = TrainingStateMonitor(
                origin_epochs=10,
                dataset_size=100,
                config=config
            )

        monitor.dump_key = {0: 0, 1: 10}
        monitor.dump_step = 1
        monitor._parse_step = Mock()

        # Should not raise error with empty directory
        monitor._dump_data_in_step(1)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=8)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=Mock())
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    def test_dump_local_loss(self, mock_rank, *mocks):
        """Test _dump_local_loss method"""

        config = {
            'local_loss_format': ['log', 'tensorboard']
        }

        # Mock both get_auto_parallel_context and context.get_auto_parallel_context
        def get_context_side_effect(key):
            if key == "dump_local_norm_path":
                return None
            if key == "pipeline_stages":
                return 1
            return None

        with patch('mindformers.core.callback.callback.get_auto_parallel_context',
                   side_effect=get_context_side_effect), \
                patch('mindformers.core.callback.callback.context.get_auto_parallel_context',
                      side_effect=get_context_side_effect):

            monitor = TrainingStateMonitor(
                origin_epochs=10,
                dataset_size=100,
                config=config
            )

        monitor.dump_step = 10
        monitor._output = Mock()

        local_losses = {
            'main': [np.array(0.5), np.array(0.6)],
            'aux': [np.array(0.1), np.array(0.2)]
        }

        monitor._dump_local_loss(local_losses)

        # Should call _output for each loss
        assert monitor._output.call_count > 0

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=8)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=Mock())
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.get_rank', return_value=0)
    def test_dump_max_attention_logit(self, mock_get_rank, mock_real_rank, *mocks):
        """Test _dump_max_attention_logit method"""

        config = {
            'max_attention_logit_format': ['log']
        }

        # Mock both get_auto_parallel_context and context.get_auto_parallel_context
        def get_context_side_effect(key):
            if key == "dump_local_norm_path":
                return None
            if key == "pipeline_stages":
                return 1
            return None

        with patch('mindformers.core.callback.callback.get_auto_parallel_context',
                   side_effect=get_context_side_effect), \
                patch('mindformers.core.callback.callback.context.get_auto_parallel_context',
                      side_effect=get_context_side_effect):

            monitor = TrainingStateMonitor(
                origin_epochs=10,
                dataset_size=100,
                config=config,
                micro_batch_num=2,
                tensor_model_parallel_size=1
            )

        monitor._output = Mock()

        # Mock cb_params with optimizer parameters
        cb_params = Mock()
        cb_params.cur_step_num = 10
        cb_params.optimizer = Mock()

        # Mock train_network to avoid infinite loop in get_real_models
        mock_network = Mock(spec=['get_max_attention_logit'])
        mock_tensor = Mock()
        mock_tensor.asnumpy.return_value = np.array([[1.5, 2.0, 2.5]])
        mock_tensor.squeeze.return_value = np.array([1.5, 2.0, 2.5])
        mock_network.get_max_attention_logit.return_value = {'layer.max_logits': mock_tensor}
        cb_params.train_network = mock_network

        # Mock parameter with max_logits_val in name
        mock_param = Mock()
        mock_param.name = 'layer.max_logits_val'
        mock_tensor_param = Mock()
        mock_tensor_param.asnumpy.return_value = np.array([[1.5, 2.0, 2.5]])
        mock_param.value.return_value = mock_tensor_param

        cb_params.optimizer._parameters = [mock_param]

        monitor._dump_max_attention_logit(cb_params)

        # Should call _output
        assert monitor._output.call_count > 0

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=8)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=Mock())
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback._get_optimizer_state',
           return_value={'layer.weight': 2.5})
    def test_dump_optimizer_state(self, mock_get_opt_state, mock_rank, *mocks):
        """Test _dump_optimizer_state method"""

        config = {
            'optimizer_state_format': ['log']
        }

        # Mock both get_auto_parallel_context and context.get_auto_parallel_context
        def get_context_side_effect(key):
            if key == "dump_local_norm_path":
                return None
            if key == "pipeline_stages":
                return 1
            return None

        with patch('mindformers.core.callback.callback.get_auto_parallel_context',
                   side_effect=get_context_side_effect), \
                patch('mindformers.core.callback.callback.context.get_auto_parallel_context',
                      side_effect=get_context_side_effect):

            monitor = TrainingStateMonitor(
                origin_epochs=10,
                dataset_size=100,
                config=config
            )

        monitor._output = Mock()
        monitor._check_param_name = Mock(return_value=True)

        # Mock cb_params with optimizer
        cb_params = Mock()
        cb_params.cur_step_num = 10
        cb_params.optimizer = Mock()
        cb_params.optimizer.moment1 = Mock()
        cb_params.optimizer.moment2 = Mock()

        monitor._dump_optimizer_state(cb_params)

        # Should call _output for adam_m and adam_v
        assert monitor._output.call_count > 0

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=8)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=Mock())
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.get_output_root_path',
           return_value='/tmp/test_output')
    @patch('mindformers.core.callback.callback.os.listdir', return_value=['file1.txt', '.nfs123'])
    @patch('mindformers.core.callback.callback.os.remove')
    def test_clear_dump_path(self, mock_remove, mock_listdir, mock_output_path, mock_rank, *mocks):
        """Test _clear_dump_path method"""

        config = {
            'dump_path': '/tmp/dump'
        }

        # Mock both get_auto_parallel_context and context.get_auto_parallel_context
        def get_context_side_effect(key):
            if key == "dump_local_norm_path":
                return '/tmp/dump'
            if key == "pipeline_stages":
                return 1
            return None

        # Mock os.path.exists to return appropriate values for different paths
        def exists_side_effect(path):
            # Return True only for dump_path, False for global_norm_record_path
            if '/tmp/dump' in path:
                return True
            return False

        with patch('mindformers.core.callback.callback.get_auto_parallel_context',
                   side_effect=get_context_side_effect), \
                patch('mindformers.core.callback.callback.context.get_auto_parallel_context',
                      side_effect=get_context_side_effect), \
                patch('mindformers.core.callback.callback.os.path.exists',
                      side_effect=exists_side_effect):

            monitor = TrainingStateMonitor(
                origin_epochs=10,
                dataset_size=100,
                config=config
            )

        monitor._clear_dump_path()

        # Should remove file1.txt but not .nfs123
        mock_remove.assert_called_once()

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
