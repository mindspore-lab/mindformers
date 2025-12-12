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
import builtins
from unittest.mock import Mock, patch
import unittest
import numpy as np
import mindspore as ms
import pytest
from mindspore import Tensor
from mindformers.core.callback.callback import MFLossMonitor

# pylint: disable=protected-access
# pylint: disable=unused-argument   # for mock logic


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


class TestMFLossMonitorBasic:
    """Test MFLossMonitor basic functionality"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=1)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=None)
    @patch('mindformers.core.callback.callback.get_tensorboard_args', return_value={})
    def test_init_defaults(self, *mocks):
        """Test MFLossMonitor default initialization"""

        monitor = MFLossMonitor(per_print_times=1, global_batch_size=32, dataset_size=100)
        assert monitor.per_print_times == 1
        assert monitor.global_batch_size == 32
        assert monitor.steps_per_epoch == 100

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=1)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=None)
    @patch('mindformers.core.callback.callback.get_tensorboard_args', return_value={})
    def test_init_custom_values(self, *mocks):
        """Test MFLossMonitor custom initialization"""

        monitor = MFLossMonitor(
            learning_rate=0.01,
            per_print_times=10,
            micro_batch_num=2,
            micro_batch_interleave_num=2,
            gradient_accumulation_steps=4
        )
        assert monitor.per_print_times == 10
        assert monitor.mirco_size == 2
        assert monitor.micro_batch_interleave_num == 2
        assert monitor.gradient_accumulation_steps == 4

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_tensorboard_args', return_value={})
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=None)
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=1)
    @patch('time.time')
    def test_on_train_epoch_begin(self, mock_time, *mocks):
        """Test on_train_epoch_begin callback"""

        mock_time.return_value = 1000.0
        monitor = MFLossMonitor()
        mock_run_context = Mock()

        monitor.on_train_epoch_begin(mock_run_context)

        assert monitor.loss_list == []
        assert monitor.epoch_time == 1000.0
        assert monitor.run_context == mock_run_context

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_tensorboard_args', return_value={})
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=None)
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=1)
    @patch('time.time')
    def test_on_train_step_begin(self, mock_time, *mocks):
        """Test on_train_step_begin callback"""

        mock_time.return_value = 1000.0
        monitor = MFLossMonitor()
        mock_run_context = Mock()

        monitor.on_train_step_begin(mock_run_context)

        assert monitor.step_time == 1000.0
        assert monitor.run_context == mock_run_context


class TestMFLossMonitorExtended:
    """Extended tests for MFLossMonitor"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=1)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=None)
    @patch('mindformers.core.callback.callback.get_tensorboard_args', return_value={})
    def test_fix_loss_for_parallel_pipeline(self, *mocks):
        """Test _fix_loss_for_parallel with pipeline stages"""

        monitor = MFLossMonitor(
            micro_batch_num=2,
            gradient_accumulation_steps=2,
            calculate_per_token_loss=False
        )

        # Mock both context.get_auto_parallel_context and get_auto_parallel_context
        with patch('mindspore.context.get_auto_parallel_context', return_value=2), \
                patch('mindspore.get_auto_parallel_context', return_value='not_zero_bubble_v'):
            loss = 8.0
            fixed_loss = monitor._fix_loss_for_parallel(loss, print_warning=False)

            # When pipeline_stages=2: loss = 8.0 / mirco_size(2) = 4.0
            # When gradient_accumulation_steps=2: loss = 4.0 / 2 = 2.0
            assert fixed_loss == 2.0

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=1)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=None)
    @patch('mindformers.core.callback.callback.get_tensorboard_args', return_value={})
    def test_fix_loss_for_parallel_gradient_accumulation(self, *mocks):
        """Test _fix_loss_for_parallel with gradient accumulation only"""

        monitor = MFLossMonitor(
            micro_batch_num=1,  # No pipeline division
            gradient_accumulation_steps=2,
            calculate_per_token_loss=False
        )

        # Mock pipeline_stages=1 (no pipeline)
        with patch('mindspore.context.get_auto_parallel_context', return_value=1), \
                patch('mindspore.get_auto_parallel_context', return_value='not_zero_bubble_v'):
            loss = 8.0
            fixed_loss = monitor._fix_loss_for_parallel(loss, print_warning=False)

            # When pipeline_stages=1: no division by mirco_size
            # When gradient_accumulation_steps=2: loss = 8.0 / 2 = 4.0
            assert fixed_loss == 4.0

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=1)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=None)
    @patch('mindformers.core.callback.callback.get_tensorboard_args', return_value={})
    def test_fix_loss_for_parallel_no_pipeline(self, *mocks):
        """Test _fix_loss_for_parallel without pipeline stages"""

        monitor = MFLossMonitor(
            micro_batch_num=2,
            gradient_accumulation_steps=2,
            calculate_per_token_loss=False
        )

        with patch('mindspore.context.get_auto_parallel_context', return_value=1), \
                patch('mindspore.get_auto_parallel_context', return_value='data_parallel'):
            loss = 8.0
            fixed_loss = monitor._fix_loss_for_parallel(loss)
            # When pipeline_stages=1: no division by mirco_size
            # When gradient_accumulation_steps=2: loss = 8.0 / 2 = 4.0
            assert fixed_loss == 4.0

    @patch('time.time', return_value=1000.0)
    @patch('mindformers.core.callback.callback.get_tensorboard_args',
           return_value={'log_loss_scale_to_tensorboard': True})
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=Mock())
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=1)
    def test_print_output_info_with_tensorboard(self, *mocks):
        """Test print_output_info with tensorboard enabled"""

        monitor = MFLossMonitor(learning_rate=0.001, global_batch_size=32)
        monitor.tensor_writer = Mock()

        cb_params = Mock()
        cb_params.dataset_sink_mode = False
        cb_params.optimizer = Mock()
        cb_params.optimizer.global_step = 10
        cb_params.train_network = Mock()
        cb_params.train_network.phase = 'train'
        cb_params.train_network.set_train = Mock()

        monitor.print_output_info(
            cb_params, 1, 10, 100.0, 1, 100, 0.5, 100.0,
            False, 1024.0, 3600, 10.0, 2.5, None, None, None
        )

        # Verify tensorboard writer was called
        assert monitor.tensor_writer.add_scalar.call_count > 0


class TestMFLossMonitorOnTrainStepEnd:
    """Test MFLossMonitor.on_train_step_end comprehensive coverage"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=1)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=None)
    @patch('mindformers.core.callback.callback.get_tensorboard_args', return_value={})
    @patch('mindformers.core.callback.callback.get_auto_parallel_context')
    @patch('mindformers.core.callback.callback.set_auto_parallel_context')
    @patch('time.time')
    def test_on_train_step_end_with_separate_loss(
            self, mock_time, mock_set_context, mock_get_context, *mocks):
        """Test on_train_step_end with print_separate_loss enabled"""

        mock_time.return_value = 1000.0

        def get_context_side_effect(x, *args):
            return {
                'parallel_mode': 'stand_alone',
                'full_batch': False
            }.get(x, None)

        mock_get_context.side_effect = get_context_side_effect

        monitor = MFLossMonitor(
            origin_epochs=10,
            dataset_size=100,
            global_batch_size=32,
            print_separate_loss=True,
            is_moe_model=True
        )
        monitor.step_time = 999.0

        run_context = Mock()
        cb_params = Mock()
        cb_params.cur_step_num = 1
        cb_params.batch_num = 100
        cb_params.cur_epoch_num = 1
        cb_params.dataset_sink_mode = False
        cb_params.net_outputs = (0.5, False, 1024.0, 0.001, 2.5)
        cb_params.get.return_value = None
        run_context.original_args.return_value = cb_params

        separate_loss_mock = (np.array([0.3]), np.array([0.1]), np.array([0.1]))
        loss_patch = 'mindformers.core.callback.callback._get_separate_loss'
        with patch(loss_patch, return_value=separate_loss_mock):
            monitor.on_train_step_end(run_context)

        assert len(monitor.loss_list) == 1

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=8)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=None)
    @patch('mindformers.core.callback.callback.get_tensorboard_args', return_value={})
    @patch('mindformers.core.callback.callback.get_auto_parallel_context')
    @patch('mindformers.core.callback.callback.set_auto_parallel_context')
    @patch('time.time')
    @patch('mindformers.core.callback.callback.check_arf_status', return_value=True)
    def test_on_train_step_end_with_arf_status(
            self, mock_arf, mock_time, mock_set_context, mock_get_context, *mocks):
        """Test on_train_step_end with ARF status check"""

        mock_time.return_value = 1000.0

        def get_context_side_effect(x, *args):
            return {
                'parallel_mode': 'stand_alone',
                'full_batch': False
            }.get(x, None)

        mock_get_context.side_effect = get_context_side_effect

        monitor = MFLossMonitor(
            origin_epochs=10,
            dataset_size=100,
            global_batch_size=32
        )
        monitor.step_time = 999.0
        monitor.mf_support = True
        monitor.mf_calculated = False

        run_context = Mock()
        cb_params = Mock()
        cb_params.cur_step_num = 1
        cb_params.batch_num = 100
        cb_params.cur_epoch_num = 1
        cb_params.dataset_sink_mode = False
        cb_params.net_outputs = 0.5
        cb_params.get.return_value = None
        cb_params.mode = 'train'
        cb_params.train_network = Mock()
        cb_params.train_network.current_phase = 'train_phase'
        run_context.original_args.return_value = cb_params

        with patch.object(monitor, '_calculate_model_flops'):
            monitor.on_train_step_end(run_context)


class TestMFLossMonitorCalculateFlops:
    """Test MFLossMonitor._calculate_model_flops"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=8)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=None)
    @patch('mindformers.core.callback.callback.get_tensorboard_args', return_value={})
    @patch('mindformers.core.callback.callback.flops_collection')
    @patch('mindformers.core.callback.callback.auto_parallel_context')
    @patch('mindformers.core.callback.callback.get_group_size', return_value=8)
    def test_calculate_model_flops_standalone(
            self, mock_group_size, mock_auto_context, mock_flops, *mocks):
        """Test _calculate_model_flops in standalone mode"""

        monitor = MFLossMonitor()
        monitor.current_phase = 'train_phase'

        mock_flops.return_value = (1000000.0, 0, 500000.0, 0, False)
        mock_auto_context.return_value.get_pipeline_stages.return_value = 1
        mock_auto_context.return_value.get_parallel_mode.return_value = 'stand_alone'

        monitor._calculate_model_flops()

        assert monitor.mf_calculated
        assert monitor.full_model_flops == 1000000.0

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=8)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=None)
    @patch('mindformers.core.callback.callback.get_tensorboard_args', return_value={})
    @patch('mindformers.core.callback.callback.flops_collection')
    def test_calculate_model_flops_runtime_error(self, mock_flops, *mocks):
        """Test _calculate_model_flops with RuntimeError"""

        monitor = MFLossMonitor()
        monitor.current_phase = 'train_phase'
        monitor.mf_support = True

        mock_flops.side_effect = RuntimeError("Flops calculation failed")

        monitor._calculate_model_flops()

        assert not monitor.mf_support


class TestMFLossMonitorPrintOutputInfo:
    """Test MFLossMonitor.print_output_info comprehensive coverage"""

    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=1)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=Mock())
    @patch('mindformers.core.callback.callback.get_tensorboard_args', return_value={
        'log_loss_scale_to_tensorboard': True,
        'log_timers_to_tensorboard': True
    })
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_group_size', return_value=8)
    def test_print_output_info_with_all_tensorboard_options(self, mock_group_size, *mocks):
        """Test print_output_info with all tensorboard options enabled"""

        monitor = MFLossMonitor(learning_rate=0.001, global_batch_size=32)
        monitor.tensor_writer = Mock()
        monitor.mf_calculated = True
        monitor.full_model_flops = 1000000.0

        cb_params = Mock()
        cb_params.dataset_sink_mode = False
        cb_params.optimizer = Mock()
        cb_params.optimizer.global_step = 10
        cb_params.train_network = Mock()
        cb_params.train_network.phase = 'train'
        cb_params.train_network.set_train = Mock()

        monitor.print_output_info(
            cb_params, 1, 10, 100.0, 1, 100, 0.5, 100.0,
            False, 1024.0, 3600, 10.0, 2.5, None, None, None
        )

        # Verify tensorboard writer was called for various metrics
        assert monitor.tensor_writer.add_scalar.call_count > 5

    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=1)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=Mock())
    @patch('mindformers.core.callback.callback.get_tensorboard_args', return_value={
        'log_timers_to_tensorboard': True
    })
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_group_size', return_value=8)
    @patch('mindformers.core.callback.callback.is_legacy_model', return_value=False)
    def test_print_output_info_with_separate_loss(self, mock_legacy, mock_group_size, *mocks):
        """Test print_output_info with separate loss (MoE and MTP)"""

        monitor = MFLossMonitor(
            learning_rate=0.001,
            global_batch_size=32,
            print_separate_loss=True,
            is_moe_model=True,
            is_mtp_model=True
        )
        # Explicitly set print_separate_loss to True (in case it was reset during init)
        monitor.print_separate_loss = True
        # Ensure tensor_writer is a Mock and tensorboard config is set
        monitor.tensor_writer = Mock()
        monitor.tensorboard = {'log_timers_to_tensorboard': True}

        cb_params = Mock()
        cb_params.dataset_sink_mode = True
        cb_params.optimizer = Mock()
        cb_params.optimizer.global_step = 10
        cb_params.train_network = Mock()
        cb_params.train_network.phase = 'train'

        # Test with separate losses
        lm_loss = np.array([0.3])
        aux_loss = np.array([0.1])
        mtp_loss = np.array([0.05])

        monitor.print_output_info(
            cb_params, 1, 10, 100.0, 1, 100, 0.5, 100.0,
            False, 1024.0, 3600, 10.0, 2.5, lm_loss, aux_loss, mtp_loss
        )

        # Verify separate loss was logged to tensorboard
        # Check if add_scalar was called with the expected tags
        call_args = monitor.tensor_writer.add_scalar.call_args_list
        tags_called = [call[0][0] for call in call_args]  # Extract first positional argument (tag)

        assert 'lm-loss' in tags_called, f"'lm-loss' not found in {tags_called}"
        assert 'mtp-loss' in tags_called, f"'mtp-loss' not found in {tags_called}"
        assert 'load-balancing-loss' in tags_called, \
            f"'load-balancing-loss' not found in {tags_called}"


class TestMFLossMonitorGetPipelineGroup:
    """Test MFLossMonitor._get_pipeline_group"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_rank', return_value=2)
    @patch('mindformers.core.callback.callback.auto_parallel_context')
    @patch('mindformers.core.callback.callback.get_group_size', return_value=8)
    def test_get_pipeline_group(self, mock_group_size, mock_auto_context, mock_get_rank):
        """Test _get_pipeline_group calculation"""

        mock_auto_context.return_value.get_pipeline_stages.return_value = 2

        rank_list, rank_list_str = MFLossMonitor._get_pipeline_group()

        # With rank=2, stage_nums=2, device_nums=8
        # per_stage_device_nums = 8 // 2 = 4
        # local_stage_rank_id = 2 % 4 = 2
        # rank_list = [2 + 0*4, 2 + 1*4] = [2, 6]
        assert rank_list == [2, 6]
        assert rank_list_str == "2-6"


class TestMFLossMonitorCanCalculateFlops:
    """Test MFLossMonitor._can_calculate_model_flops"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=1)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=None)
    @patch('mindformers.core.callback.callback.get_tensorboard_args', return_value={})
    @patch('mindformers.core.callback.callback.ms.get_context', return_value=0)  # GRAPH_MODE
    @patch('mindformers.core.callback.callback.is_legacy_model', return_value=True)
    def test_can_calculate_flops_train_mode(self, mock_legacy, mock_get_context, *mocks):
        """Test _can_calculate_model_flops in train mode"""

        monitor = MFLossMonitor()
        monitor.is_moe_model = False

        cb_params = Mock()
        cb_params.mode = 'train'
        cb_params.train_network = Mock()
        cb_params.train_network.current_phase = 'train_phase'

        result = monitor._can_calculate_model_flops(cb_params)

        assert result
        assert monitor.current_phase == 'train_phase'

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=1)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=None)
    @patch('mindformers.core.callback.callback.get_tensorboard_args', return_value={})
    @patch('mindformers.core.callback.callback.ms.get_context', return_value=1)  # PYNATIVE_MODE
    @patch('mindformers.core.callback.callback.logger')
    def test_can_calculate_flops_pynative_mode(self, mock_logger, mock_get_context, *mocks):
        """Test _can_calculate_model_flops in pynative mode (should fail)"""

        monitor = MFLossMonitor()

        cb_params = Mock()
        cb_params.mode = 'train'
        cb_params.train_network = Mock()

        result = monitor._can_calculate_model_flops(cb_params)

        assert not result
        mock_logger.warning.assert_called()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=1)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=None)
    @patch('mindformers.core.callback.callback.get_tensorboard_args', return_value={})
    @patch('mindformers.core.callback.callback.logger')
    def test_can_calculate_flops_invalid_mode(self, mock_logger, *mocks):
        """Test _can_calculate_model_flops with invalid mode"""

        monitor = MFLossMonitor()

        cb_params = Mock()
        cb_params.mode = 'predict'  # Invalid mode

        result = monitor._can_calculate_model_flops(cb_params)

        assert not result
        mock_logger.warning.assert_called()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=1)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=None)
    @patch('mindformers.core.callback.callback.get_tensorboard_args', return_value={})
    @patch('mindformers.core.callback.callback.ms.get_context', return_value=0)
    @patch('mindformers.core.callback.callback.logger')
    def test_can_calculate_flops_no_current_phase(self, mock_logger, mock_get_context, *mocks):
        """Test _can_calculate_model_flops when network has no current_phase"""

        monitor = MFLossMonitor()

        cb_params = Mock()
        cb_params.mode = 'train'
        cb_params.train_network = Mock(spec=[])  # No current_phase attribute

        result = monitor._can_calculate_model_flops(cb_params)

        assert not result
        mock_logger.warning.assert_called()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=1)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=None)
    @patch('mindformers.core.callback.callback.get_tensorboard_args', return_value={})
    @patch('mindformers.core.callback.callback.ms.get_context', return_value=0)
    @patch('mindformers.core.callback.callback.is_legacy_model', return_value=False)
    @patch('mindformers.core.callback.callback.logger')
    def test_can_calculate_flops_moe_model_non_legacy(
            self, mock_logger, mock_legacy, mock_get_context, *mocks):
        """Test _can_calculate_model_flops with MoE model in non-legacy mode"""

        monitor = MFLossMonitor()
        monitor.is_moe_model = True

        cb_params = Mock()
        cb_params.mode = 'train'
        cb_params.train_network = Mock()
        cb_params.train_network.current_phase = 'train_phase'

        result = monitor._can_calculate_model_flops(cb_params)

        assert not result
        mock_logger.warning.assert_called()


class TestMFLossMonitorPrintOutputInfoLearningRate:
    """Test MFLossMonitor.print_output_info learning rate scenarios"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=1)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=None)
    @patch('mindformers.core.callback.callback.get_tensorboard_args', return_value={})
    @patch('mindformers.core.callback.callback.ms.context.get_context', return_value='CPU')
    @patch('mindformers.core.callback.callback.logger')
    def test_print_output_info_lr_schedule_cpu(self, mock_logger, mock_get_context, *mocks):
        """Test print_output_info with LearningRateSchedule on CPU"""

        lr_schedule = Mock(spec=['__call__'])
        monitor = MFLossMonitor(learning_rate=lr_schedule, global_batch_size=32)
        monitor.print_warning_flag = True

        cb_params = Mock()
        cb_params.dataset_sink_mode = False
        cb_params.optimizer = Mock()
        cb_params.optimizer.global_step = 10

        monitor.print_output_info(
            cb_params, 1, 10, 100.0, 1, 100, 0.5, 100.0,
            False, 1024.0, 3600, 10.0, 2.5, None, None, None
        )

        # Should log warning about CPU not supported
        mock_logger.warning.assert_called()
        assert not monitor.print_warning_flag

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=1)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=None)
    @patch('mindformers.core.callback.callback.get_tensorboard_args', return_value={})
    @patch('mindformers.core.callback.callback.ms.context.get_context', return_value='Ascend')
    def test_print_output_info_lr_schedule_ascend(self, mock_get_context, *mocks):
        """Test print_output_info with LearningRateSchedule on Ascend"""

        # Create a simple mock that can be called
        lr_schedule = Mock()
        lr_result = Mock()
        lr_result.asnumpy.return_value = np.array(0.001)
        lr_schedule.return_value = lr_result

        monitor = MFLossMonitor(learning_rate=lr_schedule, global_batch_size=32)

        # Manually set the learning_rate to be recognized as LearningRateSchedule
        # by patching the isinstance check in print_output_info
        with patch('mindformers.core.callback.callback.isinstance') as mock_isinstance:
            # Default behavior: call the real isinstance
            def isinstance_side_effect(obj, classinfo):
                # Special handling for our lr_schedule object
                if obj is monitor.learning_rate:
                    # Check if classinfo is a tuple (for the first isinstance check)
                    if isinstance(classinfo, tuple):
                        return False  # Not (float, Tensor, np.ndarray)
                    return True  # Is LearningRateSchedule
                # For all other cases, use built-in isinstance
                return builtins.isinstance(obj, classinfo)

            mock_isinstance.side_effect = isinstance_side_effect

            cb_params = Mock()
            cb_params.dataset_sink_mode = False
            cb_params.optimizer = Mock()
            cb_params.optimizer.global_step = 10
            cb_params.train_network = Mock()
            cb_params.train_network.phase = 'train'
            cb_params.train_network.set_train = Mock()

            monitor.print_output_info(
                cb_params, 1, 10, 100.0, 1, 100, 0.5, 100.0,
                False, 1024.0, 3600, 10.0, 2.5, None, None, None
            )

            # Verify set_train was called to temporarily disable training
            cb_params.train_network.set_train.assert_called()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=1)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=None)
    @patch('mindformers.core.callback.callback.get_tensorboard_args', return_value={})
    @patch('mindformers.core.callback.callback.logger')
    def test_print_output_info_invalid_lr_type(self, mock_logger, *mocks):
        """Test print_output_info with invalid learning rate type"""

        # Use a list as learning rate (invalid type)
        monitor = MFLossMonitor(learning_rate=[0.01, 0.02], global_batch_size=32)
        monitor.print_warning_flag = True

        cb_params = Mock()
        cb_params.dataset_sink_mode = False
        cb_params.optimizer = Mock()
        cb_params.optimizer.global_step = 10

        monitor.print_output_info(
            cb_params, 1, 10, 100.0, 1, 100, 0.5, 100.0,
            False, 1024.0, 3600, 10.0, 2.5, None, None, None
        )

        # Should log warning about invalid type
        mock_logger.warning.assert_called()
        assert not monitor.print_warning_flag

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=1)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=None)
    @patch('mindformers.core.callback.callback.get_tensorboard_args', return_value={})
    @patch('mindformers.core.callback.callback.logger')
    def test_print_output_info_no_lr(self, mock_logger, *mocks):
        """Test print_output_info without learning rate"""

        monitor = MFLossMonitor(learning_rate=None, global_batch_size=32)
        monitor.print_warning_flag = True

        cb_params = Mock()
        cb_params.dataset_sink_mode = False
        cb_params.optimizer = Mock()
        cb_params.optimizer.global_step = 10

        monitor.print_output_info(
            cb_params, 1, 10, 100.0, 1, 100, 0.5, 100.0,
            False, 1024.0, 3600, 10.0, 2.5, None, None, None
        )

        # Should log warning about missing learning rate
        mock_logger.warning.assert_called()
        assert not monitor.print_warning_flag


class TestMFLossMonitorCalculateFlopsWithPipeline:
    """Test MFLossMonitor._calculate_model_flops with pipeline parallel"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=8)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=None)
    @patch('mindformers.core.callback.callback.get_tensorboard_args', return_value={})
    @patch('mindformers.core.callback.callback.flops_collection')
    @patch('mindformers.core.callback.callback.auto_parallel_context')
    @patch('mindformers.core.callback.callback.get_group_size', return_value=8)
    @patch('mindformers.core.callback.callback.create_group')
    @patch('mindformers.core.callback.callback.AllReduceNet')
    @patch('mindformers.core.callback.callback.Tensor')
    def test_calculate_flops_with_pipeline_dynamic_shape(self, mock_tensor, mock_allreduce_net,
                                                         mock_create_group, mock_group_size,
                                                         mock_auto_context, mock_flops, *mocks):
        """Test _calculate_model_flops with pipeline and dynamic shape"""

        monitor = MFLossMonitor()
        monitor.current_phase = 'train_phase'

        mock_flops.return_value = (1000000.0, 0, 500000.0, 0, True)  # is_dynamic_shape=True
        mock_auto_context.return_value.get_pipeline_stages.return_value = 2

        # Mock AllReduceNet to return is_dynamic_shape > 0
        mock_allreduce_instance = Mock()
        mock_result = Mock()
        mock_result.asnumpy.return_value = [1]  # is_dynamic_shape > 0
        mock_allreduce_instance.return_value = mock_result
        mock_allreduce_net.return_value = mock_allreduce_instance

        monitor._calculate_model_flops()

        # Should set mf_support to False due to dynamic shape
        assert not monitor.mf_support

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=8)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=None)
    @patch('mindformers.core.callback.callback.get_tensorboard_args', return_value={})
    @patch('mindformers.core.callback.callback.flops_collection')
    @patch('mindformers.core.callback.callback.auto_parallel_context')
    @patch('mindformers.core.callback.callback.get_group_size', return_value=8)
    @patch('mindformers.core.callback.callback.create_group')
    @patch('mindformers.core.callback.callback.AllReduceNet')
    @patch('mindformers.core.callback.callback.Tensor')
    def test_calculate_flops_with_pipeline_success(self, mock_tensor, mock_allreduce_net,
                                                   mock_create_group, mock_group_size,
                                                   mock_auto_context, mock_flops, *mocks):
        """Test _calculate_model_flops with pipeline parallel success"""

        monitor = MFLossMonitor()
        monitor.current_phase = 'train_phase'

        mock_flops.return_value = (1000000.0, 0, 500000.0, 0, False)  # is_dynamic_shape=False
        mock_auto_context.return_value.get_pipeline_stages.return_value = 2
        mock_auto_context.return_value.get_parallel_mode.return_value = 'semi_auto_parallel'

        # Mock AllReduceNet
        mock_allreduce_instance = Mock()

        # First call: is_dynamic_shape check
        mock_is_dynamic_result = Mock()
        mock_is_dynamic_result.asnumpy.return_value = [0]

        # Second call: flops aggregation
        mock_flops_result = Mock()
        mock_flops_result.asnumpy.return_value = [2000000.0]

        mock_allreduce_instance.side_effect = [mock_is_dynamic_result, mock_flops_result]
        mock_allreduce_net.return_value = mock_allreduce_instance

        monitor._calculate_model_flops()

        # Should aggregate flops across pipeline stages and divide by group size
        assert monitor.mf_calculated
        # 2000000.0 / 8 = 250000.0
        assert monitor.full_model_flops == 250000.0


class TestMFLossMonitorPrintOutputInfoDataSinkMode:
    """Test MFLossMonitor.print_output_info in dataset sink mode"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=1)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=None)
    @patch('mindformers.core.callback.callback.get_tensorboard_args', return_value={})
    def test_print_output_info_sink_mode(self, *mocks):
        """Test print_output_info in dataset sink mode"""

        monitor = MFLossMonitor(learning_rate=0.001, global_batch_size=32)

        cb_params = Mock()
        cb_params.dataset_sink_mode = True  # Sink mode
        cb_params.optimizer = Mock()
        cb_params.optimizer.global_step = 10

        monitor.print_output_info(
            cb_params, 1, 10, 100.0, 1, 100, 0.5, 100.0,
            False, 1024.0, 3600, 10.0, 2.5, None, None, None
        )

        # In sink mode, loss_info format is different
        # This test mainly ensures no errors occur


class TestMFLossMonitorMstxEnabled:
    """Test MFLossMonitor with mstx enabled"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=1)
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=None)
    @patch('mindformers.core.callback.callback.get_tensorboard_args', return_value={})
    @patch('mindformers.core.callback.callback._check_mspti_is_on', return_value=True)
    @patch('mindformers.core.callback.callback.ms.profiler.mstx')
    @patch('mindformers.core.callback.callback.ms.runtime')
    @patch('time.time')
    def test_on_train_step_with_mstx(self, mock_time, mock_runtime, mock_mstx, mock_mspti, *mocks):
        """Test on_train_step_begin and on_train_step_end with mstx enabled"""

        mock_time.return_value = 1000.0
        mock_mstx.range_start.return_value = 12345
        mock_runtime.current_stream.return_value = Mock()

        monitor = MFLossMonitor(origin_epochs=10, dataset_size=100, global_batch_size=32)

        run_context = Mock()
        cb_params = Mock()
        cb_params.cur_step_num = 5
        run_context.original_args.return_value = cb_params

        # Test on_train_step_begin
        monitor.on_train_step_begin(run_context)

        mock_mstx.range_start.assert_called_once()
        assert monitor.mstx_range_id == 12345

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
