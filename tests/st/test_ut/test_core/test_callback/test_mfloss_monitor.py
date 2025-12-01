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
"""Test test_mfloss_monitor.py"""
from unittest.mock import Mock, patch
import unittest
import numpy as np
import mindspore as ms
import pytest
from mindspore import Tensor
from mindformers.core.callback.callback import MFLossMonitor


class TestMFLossMonitor(unittest.TestCase):
    """Test MFLossMonitor class"""

    def setUp(self):
        """Setup before tests"""
        self.monitor = MFLossMonitor(
            per_print_times=1,
            global_batch_size=32,
            dataset_size=100
        )

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_init_defaults(self):
        """Test MFLossMonitor default initialization"""
        self.assertEqual(self.monitor.per_print_times, 1)
        self.assertEqual(self.monitor.global_batch_size, 32)
        self.assertEqual(self.monitor.steps_per_epoch, 100)
        self.assertEqual(self.monitor.last_print_time, 0)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_init_custom_values(self):
        """Test MFLossMonitor custom initialization"""
        monitor = MFLossMonitor(
            learning_rate=0.01,
            per_print_times=10,
            micro_batch_num=2,
            micro_batch_interleave_num=2,
            gradient_accumulation_steps=4
        )

        self.assertEqual(monitor.per_print_times, 10)
        self.assertEqual(monitor.mirco_size, 2)
        self.assertEqual(monitor.micro_batch_interleave_num, 2)
        self.assertEqual(monitor.gradient_accumulation_steps, 4)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('time.time')
    def test_on_train_epoch_begin(self, mock_time):
        """Test on_train_epoch_begin callback"""
        mock_time.return_value = 1000.0
        mock_run_context = Mock()

        self.monitor.on_train_epoch_begin(mock_run_context)

        self.assertEqual(self.monitor.loss_list, [])
        self.assertEqual(self.monitor.epoch_time, 1000.0)
        self.assertEqual(self.monitor.run_context, mock_run_context)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('time.time')
    def test_on_train_step_begin(self, mock_time):
        """Test on_train_step_begin callback"""
        mock_time.return_value = 1000.0
        mock_run_context = Mock()

        self.monitor.on_train_step_begin(mock_run_context)

        self.assertEqual(self.monitor.step_time, 1000.0)
        self.assertEqual(self.monitor.run_context, mock_run_context)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_fix_loss_for_parallel_no_adjustment(self):
        """Test _fix_loss_for_parallel without adjustment"""
        with patch('mindspore.context.get_auto_parallel_context') as mock_get_context:
            mock_get_context.return_value = 1  # pipeline_stages = 1

            original_loss = 1.0
            # pylint: disable=W0212
            fixed_loss = self.monitor._fix_loss_for_parallel(original_loss)

            self.assertEqual(fixed_loss, original_loss)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_fix_loss_for_parallel_with_pipeline(self):
        """Test _fix_loss_for_parallel with pipeline parallel adjustment"""
        self.monitor.mirco_size = 2
        self.monitor.calculate_per_token_loss = False

        with patch('mindspore.context.get_auto_parallel_context') as mock_get_context:
            mock_get_context.return_value = 2  # pipeline_stages = 2

            original_loss = 2.0
            # pylint: disable=W0212
            fixed_loss = self.monitor._fix_loss_for_parallel(original_loss)

            # Should divide by micro_size
            self.assertEqual(fixed_loss, 1.0)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_can_calculate_model_flops_train_mode(self):
        """Test _can_calculate_model_flops in train mode"""
        mock_cb_params = Mock()
        mock_cb_params.mode = 'train'
        mock_cb_params.train_network = Mock()
        mock_cb_params.train_network.current_phase = 'train_phase'

        with patch('mindspore.get_context') as mock_get_context:
            mock_get_context.return_value = ms.GRAPH_MODE

            # pylint: disable=W0212
            result = self.monitor._can_calculate_model_flops(mock_cb_params)

            self.assertTrue(result)
            self.assertEqual(self.monitor.current_phase, 'train_phase')

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_can_calculate_model_flops_invalid_mode(self):
        """Test _can_calculate_model_flops with invalid mode"""
        mock_cb_params = Mock()
        mock_cb_params.mode = 'invalid'

        # pylint: disable=W0212
        result = self.monitor._can_calculate_model_flops(mock_cb_params)

        self.assertFalse(result)


class TestMFLossMonitorIntegration(unittest.TestCase):
    """MFLossMonitor integration tests"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.core.callback.callback.get_real_group_size')
    @patch('time.time')
    def test_on_train_step_end_basic_flow(self, mock_time, mock_get_group_size):
        """Test on_train_step_end basic flow"""
        mock_time.return_value = 1000.0
        mock_get_group_size.return_value = 1

        # Create monitor
        monitor = MFLossMonitor(
            per_print_times=1,
            global_batch_size=32,
            dataset_size=100,
            origin_epochs=1,
            device_num=1
        )
        mock_time.return_value = 1001.0
        # Create run context
        mock_run_context = Mock()
        mock_cb_params = Mock()
        mock_cb_params.net_outputs = Tensor(np.array([0.5]))
        mock_cb_params.cur_step_num = 1
        mock_cb_params.cur_epoch_num = 1
        mock_cb_params.batch_num = 100
        mock_cb_params.dataset_sink_mode = False
        mock_cb_params.initial_step = 0
        mock_cb_params.mode = 'train'
        mock_run_context.original_args.return_value = mock_cb_params

        # Mock auto parallel context
        with patch('mindspore.get_auto_parallel_context') as mock_auto_parallel, \
                patch('mindspore.context.get_auto_parallel_context') as mock_context_auto_parallel:
            mock_auto_parallel.return_value = 'data_parallel'
            mock_context_auto_parallel.return_value = 1  # pipeline_stages

            # Execute test
            monitor.on_train_step_end(mock_run_context)

            # Verify loss is recorded
            self.assertEqual(len(monitor.loss_list), 1)
            self.assertEqual(monitor.loss_list[0], 0.5)


if __name__ == '__main__':
    unittest.main()
