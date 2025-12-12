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
"""Test test_checkpoint_monitor.py"""
import json
import os
import unittest
import shutil
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pytest
from mindspore import ModelCheckpoint
from mindformers.core.callback.callback import CheckpointMonitor

# pylint: disable=protected-access
# pylint: disable=unused-argument   # for mock logic


class TestCheckpointMonitor(unittest.TestCase):
    """Test cases for CheckpointMonitor class"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_global_batch_size_parameter(self):
        """Test global batch size parameter"""
        monitor = CheckpointMonitor(
            global_batch_size=64
        )

        self.assertEqual(monitor.global_batch_size, 64)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_parameter_validation(self):
        """Test parameter validation"""
        # Test invalid directory type
        with self.assertRaises(TypeError):
            CheckpointMonitor(directory=123)

        # Test invalid save_checkpoint_steps type
        with self.assertRaises(TypeError):
            CheckpointMonitor(save_checkpoint_steps="invalid")

        # Test invalid keep_checkpoint_max type
        with self.assertRaises(TypeError):
            CheckpointMonitor(keep_checkpoint_max="invalid")

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_inheritance_from_model_checkpoint(self):
        """Test that CheckpointMonitor properly inherits from ModelCheckpoint"""
        monitor = CheckpointMonitor()

        # Check that it's an instance of ModelCheckpoint
        self.assertIsInstance(monitor, ModelCheckpoint)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_remove_redundancy_parameter(self):
        """Test remove_redundancy parameter"""
        monitor = CheckpointMonitor(
            remove_redundancy=True
        )

        self.assertTrue(monitor.remove_redundancy)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_save_network_params_parameter(self):
        """Test save_network_params parameter"""
        monitor = CheckpointMonitor(
            save_network_params=True
        )

        self.assertTrue(monitor.save_network_params)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_save_trainable_params_parameter(self):
        """Test save_trainable_params parameter"""
        monitor = CheckpointMonitor(
            save_trainable_params=True
        )

        self.assertTrue(monitor.save_trainable_params)


class TestCheckpointMonitorExtended:
    """Extended tests for CheckpointMonitor"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.get_output_subpath', return_value='/tmp/checkpoint')
    def test_checkpoint_monitor_init(self, *mocks):
        """Test CheckpointMonitor initialization"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            save_checkpoint_steps=100,
            global_batch_size=32
        )

        assert monitor.global_batch_size == 32
        assert monitor.save_checkpoint_steps == 100

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.get_output_subpath', return_value='/tmp/checkpoint')
    def test_checkpoint_monitor_remove_redundancy(self, *mocks):
        """Test CheckpointMonitor with remove_redundancy parameter"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            remove_redundancy=True
        )

        assert monitor.need_remove_redundancy

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.get_output_subpath', return_value='/tmp/checkpoint')
    def test_checkpoint_monitor_save_network_params(self, *mocks):
        """Test CheckpointMonitor with save_network_params parameter"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            save_network_params=True
        )

        assert monitor.save_network_params

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.get_output_subpath', return_value='/tmp/checkpoint')
    def test_checkpoint_monitor_save_trainable_params(self, *mocks):
        """Test CheckpointMonitor with save_trainable_params parameter"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            save_trainable_params=True
        )

        assert monitor.save_trainable_params


class TestCheckpointMonitorHelpers:
    """Test CheckpointMonitor helper methods"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_output_subpath')
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    def test_record_last_ckpt_to_json(self, mock_get_real_rank, mock_get_output_subpath):
        """Test record_last_ckpt_to_json method"""

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_get_output_subpath.return_value = tmpdir

            monitor = CheckpointMonitor(prefix='TEST')
            monitor._directory = tmpdir
            monitor.meta_json = os.path.join(tmpdir, 'meta.json')

            monitor.record_last_ckpt_to_json(5, 10, 'test_ckpt.ckpt')

            # Verify file was created
            assert os.path.exists(monitor.meta_json)

            # Verify content
            with open(monitor.meta_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            assert data['last_epoch'] == 5
            assert data['last_step'] == 10
            assert data['last_ckpt_file'] == 'test_ckpt.ckpt'


class TestCheckpointMonitorSaveAndHealth:
    """Test CheckpointMonitor save and health check methods"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.get_rank', return_value=0)
    @patch('mindformers.core.callback.callback.get_group_size', return_value=8)
    @patch('mindformers.core.callback.callback.auto_parallel_context')
    @patch('mindformers.core.callback.callback.get_embedding_info', return_value=1.5)
    @patch('mindformers.core.callback.callback.AllReduceNet')
    @patch('mindformers.core.callback.callback.create_group')
    @patch('mindformers.core.callback.callback.ms.get_auto_parallel_context',
           return_value='semi_auto_parallel')
    @patch('mindformers.core.callback.callback.ms.set_auto_parallel_context')
    def test_get_checkpoint_health_info_healthy(
            self, mock_set_context, mock_get_parallel_mode,
            mock_create_group, mock_allreduce_net, mock_get_embedding,
            mock_auto_context, mock_group_size, mock_get_rank, mock_real_rank):
        """Test get_checkpoint_health_info when checkpoint is healthy"""

        mock_auto_context.return_value.get_pipeline_stages.return_value = 2

        # Mock AllReduce result
        mock_allreduce_instance = Mock()
        mock_health_tensor = Mock()
        mock_health_tensor.asnumpy.return_value = np.array([1.0])
        mock_allreduce_instance.return_value = mock_health_tensor
        mock_allreduce_net.return_value = mock_allreduce_instance

        # Mock create_group to avoid distributed communication initialization
        mock_create_group.return_value = None

        monitor = CheckpointMonitor(
            prefix='TEST',
            directory='./test_ckpt',
            embedding_size=4096,
            embedding_local_norm_threshold=1.0,
            use_checkpoint_health_monitor=True
        )

        cb_params = Mock()
        cb_params.cur_step_num = 10

        is_health = monitor.get_checkpoint_health_info(cb_params)

        assert is_health == 1

        # Verify create_group was called
        mock_create_group.assert_called_once()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.time.time', return_value=1000.0)
    @patch('mindformers.core.callback.callback.os.path.exists', return_value=True)
    @patch('mindformers.core.callback.callback.os.path.getmtime', return_value=1005.0)
    def test_print_savetime(self, mock_getmtime, mock_exists, mock_time):
        """Test print_savetime method"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            directory='./test_ckpt'
        )

        # Setup save_info_list
        monitor.save_info_list[10] = {
            'ckpt': {
                'save_start_time': 1000.0,
                'ckpt_file_path': '/path/to/ckpt.ckpt',
                'save_end_time': None
            },
            'network': {
                'save_start_time': None,
                'ckpt_file_path': None,
                'save_end_time': None
            },
            'trainable_params': {
                'save_start_time': None,
                'ckpt_file_path': None,
                'save_end_time': None
            }
        }

        monitor.print_savetime(10, 100)

        # Verify the save_end_time was set
        assert monitor.save_info_list[10]['ckpt']['save_end_time'] is not None


class TestCheckpointMonitorStepEndAndTrainEnd:
    """Test CheckpointMonitor.step_end and on_train_end"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    def test_step_end_legacy_format(self, mock_real_rank):
        """Test step_end with legacy format"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            directory='./test_ckpt',
            use_legacy_format=True
        )

        run_context = Mock()
        cb_params = Mock()
        cb_params.cur_step_num = 10
        run_context.original_args.return_value = cb_params

        # Should call parent's step_end
        with patch.object(CheckpointMonitor.__bases__[0], 'step_end') as mock_parent_step_end:
            monitor.step_end(run_context)
            mock_parent_step_end.assert_called_once()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    def test_on_train_end_new_format(self, mock_real_rank):
        """Test on_train_end with new format"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            directory='./test_ckpt',
            use_legacy_format=False,
            async_save=False
        )

        monitor._save_megatron_ckpt_file_format = Mock()

        run_context = Mock()
        cb_params = Mock()
        cb_params.cur_step_num = 100
        run_context.original_args.return_value = cb_params

        monitor.on_train_end(run_context)

        monitor._save_megatron_ckpt_file_format.assert_called_once_with(cb_params)


class TestCheckpointMonitorSaveCkpt:
    """Test CheckpointMonitor._save_ckpt method"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.check_arf_status', return_value=False)
    def test_save_ckpt_skip_same_step(self, mock_arf, mock_real_rank):
        """Test _save_ckpt skips when called twice for same step"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            directory='./test_ckpt'
        )

        cb_params = Mock()
        cb_params.cur_step_num = 10

        monitor._last_triggered_step = 10
        monitor.save_checkpoint = Mock()

        # Should return early without saving
        monitor._save_ckpt(cb_params, force_to_save=False)

        monitor.save_checkpoint.assert_not_called()


class TestCheckpointMonitorSaveCheckpoint:
    """Test CheckpointMonitor.save_checkpoint and related methods"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.logger')
    @patch('mindformers.core.callback.callback.get_rank', return_value=0)
    @patch('mindformers.core.callback.callback.time.time', return_value=1000.0)
    @patch('mindformers.core.callback.callback.os.path.join',
           side_effect=lambda *args: '/'.join(args))
    @patch('mindformers.core.callback.callback.set_safe_mode_for_file_or_dir')
    @patch('mindformers.core.callback.callback.json.dump')
    @patch('mindformers.core.callback.callback.json.load', return_value=[])
    @patch('builtins.open', create=True)
    @patch('mindformers.core.callback.callback.os.path.exists', return_value=False)
    def test_save_checkpoint_with_health_monitor(self, mock_exists, mock_open_file, mock_json_load,
                                                 mock_json_dump, mock_safe_mode, mock_path_join,
                                                 mock_time, mock_get_rank, mock_logger, mock_real_rank):
        """Test save_checkpoint with health monitoring enabled"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            directory='./test_ckpt',
            use_checkpoint_health_monitor=True,
            health_ckpts_record_dir='./health'
        )

        monitor.save_info_list[10] = {
            'ckpt': {'save_start_time': None, 'ckpt_file_path': None, 'save_end_time': None},
            'network': {'save_start_time': None, 'ckpt_file_path': None, 'save_end_time': None},
            'trainable_params': {
                'save_start_time': None, 'ckpt_file_path': None,
                'save_end_time': None}
        }

        cb_params = Mock()
        cb_params.cur_step_num = 10
        cb_params.cur_epoch_num = 1
        cb_params.batch_num = 100
        cb_params.optimizer = Mock()
        cb_params.optimizer.global_step = 10
        cb_params.train_network = Mock()

        monitor.get_checkpoint_health_info = Mock(return_value=1)
        monitor.remove_redundancy = Mock()
        monitor._manager = Mock()
        monitor._manager.ckpoint_num = 0

        # Should not raise error
        monitor.save_checkpoint(cb_params)

        # Verify health info was checked
        monitor.get_checkpoint_health_info.assert_called_once()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.time.time', return_value=1000.0)
    def test_filter_ckpt_not_save(self, mock_time, mock_real_rank):
        """Test _filter_ckpt_not_save method"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            directory='./test_ckpt'
        )

        monitor.filter_list = ['optimizer', 'temp']

        # Should filter out parameters starting with filter_list items
        assert not monitor._filter_ckpt_not_save('optimizer.weight', monitor.filter_list)
        assert not monitor._filter_ckpt_not_save('model.temp', monitor.filter_list)
        assert monitor._filter_ckpt_not_save('model.weight', monitor.filter_list)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.time.time', return_value=1000.0)
    @patch('mindformers.core.callback.callback.context.get_auto_parallel_context',
           return_value='stand_alone')
    @patch('mindformers.core.callback.callback.ms.save_checkpoint')
    def test_remove_redundancy_standalone(
            self, mock_save_ckpt, mock_context, mock_time, mock_real_rank):
        """Test remove_redundancy in standalone mode"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            directory='./test_ckpt',
            remove_redundancy=True
        )

        network = Mock()
        cur_file = './test.ckpt'
        append_dict = {}

        # In standalone mode, should use simple save
        monitor.remove_redundancy(network, cur_file, append_dict, None)

        mock_save_ckpt.assert_called_once()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.time.time', return_value=1000.0)
    def test_get_cur_dp(self, mock_time, mock_real_rank):
        """Test _get_cur_dp method"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            directory='./test_ckpt'
        )

        # Test with simple redundancy dict
        param_redundancy_dict = {
            'layer.weight': [(0, 1, 2, 3), (4, 5, 6, 7)],
            'layer.bias': [(0, 1, 2, 3), (4, 5, 6, 7)]
        }

        cur_dp = monitor._get_cur_dp(0, param_redundancy_dict)

        # Should return the tuple containing rank 0
        assert 0 in cur_dp


class TestCheckpointMonitorTFTSave:
    """Test CheckpointMonitor TFT save methods"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.ms.save_checkpoint')
    def test_tft_save_ckpt(self, mock_save_ckpt, mock_real_rank):
        """Test _tft_save_ckpt method"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            directory='./test_ckpt',
            remove_redundancy=True
        )

        param_layout_set = {'layer.weight', 'layer.bias'}
        save_param_names = {'layer.weight'}
        cur_file = './test_rank_0/ckpt_1.ckpt'
        append_dict = {'epoch_num': 1}
        network = Mock()

        monitor._tft_save_ckpt(param_layout_set, save_param_names, cur_file, append_dict, network)

        mock_save_ckpt.assert_called_once()
        # Verify choice_func filters correctly
        call_args = mock_save_ckpt.call_args
        choice_func = call_args[1].get('choice_func') if call_args[1] else None
        if choice_func:
            # layer.weight is in save_param_names
            assert choice_func('layer.weight')
            # layer.bias is in param_layout_set but not in save_param_names
            assert not choice_func('layer.bias')

    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.context.get_auto_parallel_context',
           return_value='semi_auto_parallel')
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.logger')
    @patch('mindformers.core.callback.callback.re.sub')
    def test_do_remove_redundancy_for_tft(
            self, mock_re_sub, mock_logger, mock_context, mock_real_rank):
        """Test _do_remove_redundancy_for_tft method"""

        def re_sub_side_effect(p, r, s):
            if 'rank_' in p:
                return s.replace('rank_0', f'rank_{r.split("_")[1]}')
            return s

        mock_re_sub.side_effect = re_sub_side_effect

        monitor = CheckpointMonitor(
            prefix='TEST',
            directory='./test_ckpt',
            remove_redundancy=True,
            checkpoint_format='ckpt'
        )

        monitor._tft_save_ckpt = Mock()
        monitor.record_last_ckpt_to_json = Mock()

        redundancy_info = (
            0,  # rank_id
            {'layer.weight': [(0, 1)]},  # param_redundancy_dict
            {0: {'layer.weight'}, 1: {'layer.weight'}},  # single_params
            {'layer.weight': Mock()}  # param_layout
        )
        cur_file = './test_rank_0/ckpt_1_10.ckpt'
        network = Mock()
        append_dict = {'epoch_num': 1}

        monitor._do_remove_redundancy_for_tft(redundancy_info, cur_file, network, append_dict)

        # Should call _tft_save_ckpt for each rank in cur_dp
        assert monitor._tft_save_ckpt.called
        # Should set __exception_save__ flag
        assert '__exception_save__' in append_dict

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.ms.save_checkpoint')
    def test_tft_save_ckpt_with_filter_list(self, mock_save_ckpt, mock_real_rank):
        """Test _tft_save_ckpt filters out items in filter_list"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            directory='./test_ckpt',
            remove_redundancy=True
        )

        param_layout_set = set()
        save_param_names = {'layer.weight', 'accu_grads.weight'}
        cur_file = './test.ckpt'
        append_dict = {}
        network = Mock()

        monitor._tft_save_ckpt(param_layout_set, save_param_names, cur_file, append_dict, network)

        # Verify choice_func filters out filter_list items
        call_args = mock_save_ckpt.call_args
        choice_func = call_args[1].get('choice_func') if call_args[1] else None
        if choice_func:
            # accu_grads should be filtered out
            assert not choice_func('accu_grads.weight')


class TestCheckpointMonitorSkipTrainableParams:
    """Test CheckpointMonitor._check_if_skip_trainable_params method"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.context.get_context', return_value=0)  # GRAPH_MODE
    @patch('mindformers.core.callback.callback.ms.get_auto_parallel_context',
           return_value='semi_auto_parallel')
    def test_skip_trainable_params_graph_mode_parallel(
            self, mock_parallel_ctx, mock_get_ctx, mock_real_rank):
        """Test _check_if_skip_trainable_params in graph mode with auto parallel"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            directory='./test_ckpt'
        )

        # Mock parameter that is not sliced
        mock_param = Mock()
        mock_param.sliced = False
        mock_param.has_init = False
        mock_param.param_info = Mock()
        mock_param.param_info.is_pipeline_shared_param = False

        result = monitor._check_if_skip_trainable_params(mock_param)
        # Should skip because sliced=False in graph mode + auto parallel
        assert result

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.context.get_context', return_value=0)  # GRAPH_MODE
    @patch('mindformers.core.callback.callback.ms.get_auto_parallel_context',
           return_value='semi_auto_parallel')
    def test_skip_trainable_params_has_init(self, mock_parallel_ctx, mock_get_ctx, mock_real_rank):
        """Test _check_if_skip_trainable_params with has_init=True"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            directory='./test_ckpt'
        )

        # Mock parameter with has_init=True
        mock_param = Mock()
        mock_param.sliced = True
        mock_param.has_init = True
        mock_param.param_info = Mock()
        mock_param.param_info.is_pipeline_shared_param = False

        result = monitor._check_if_skip_trainable_params(mock_param)
        # Should skip because has_init=True in graph mode + auto parallel
        assert result

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.context.get_context', return_value=0)  # GRAPH_MODE
    @patch('mindformers.core.callback.callback.ms.get_auto_parallel_context',
           return_value='semi_auto_parallel')
    def test_skip_trainable_params_pipeline_shared(
            self, mock_parallel_ctx, mock_get_ctx, mock_real_rank):
        """Test _check_if_skip_trainable_params with pipeline shared param"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            directory='./test_ckpt'
        )

        # Mock parameter that is pipeline shared
        mock_param = Mock()
        mock_param.sliced = True
        mock_param.has_init = False
        mock_param.param_info = Mock()
        mock_param.param_info.is_pipeline_shared_param = True

        result = monitor._check_if_skip_trainable_params(mock_param)
        # Should skip because is_pipeline_shared_param=True
        assert result

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.context.get_context',
           return_value=1)  # PYNATIVE_MODE
    @patch('mindformers.core.callback.callback.ms.get_auto_parallel_context',
           return_value='semi_auto_parallel')
    def test_skip_trainable_params_pynative_mode(
            self, mock_parallel_ctx, mock_get_ctx, mock_real_rank):
        """Test _check_if_skip_trainable_params in pynative mode"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            directory='./test_ckpt'
        )

        # Mock parameter
        mock_param = Mock()
        mock_param.sliced = False
        mock_param.has_init = False
        mock_param.param_info = Mock()
        mock_param.param_info.is_pipeline_shared_param = False

        result = monitor._check_if_skip_trainable_params(mock_param)
        # Should not skip in pynative mode (not graph mode)
        assert not result

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.context.get_context', return_value=0)  # GRAPH_MODE
    @patch('mindformers.core.callback.callback.ms.get_auto_parallel_context',
           return_value='stand_alone')
    def test_skip_trainable_params_standalone(
            self, mock_parallel_ctx, mock_get_ctx, mock_real_rank):
        """Test _check_if_skip_trainable_params in standalone mode"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            directory='./test_ckpt'
        )

        # Mock parameter
        mock_param = Mock()
        mock_param.sliced = False
        mock_param.has_init = False
        mock_param.param_info = Mock()
        mock_param.param_info.is_pipeline_shared_param = False

        result = monitor._check_if_skip_trainable_params(mock_param)
        # Should not skip in standalone mode
        assert not result


class TestCheckpointMonitorRemoveRedundancyBranches:
    """Test CheckpointMonitor.remove_redundancy method branches"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=8)
    @patch('mindformers.core.callback.callback.get_auto_parallel_context',
           return_value=1)  # 1 stage
    @patch('mindformers.core.callback.callback.context.get_auto_parallel_context',
           return_value='semi_auto_parallel')
    @patch('mindformers.core.callback.callback.get_parameter_redundancy')
    @patch('mindformers.core.callback.callback.remove_param_redundancy')
    @patch('mindformers.core.callback.callback.ms.save_checkpoint')
    @patch('mindformers.core.callback.callback.logger')
    def test_remove_redundancy_with_param_layout(self, mock_logger, mock_save_ckpt,
                                                 mock_remove_redundancy, mock_get_redundancy,
                                                 mock_context, mock_pp, mock_group_size, mock_rank):
        """Test remove_redundancy with param_layout dict"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            directory='./test_ckpt',
            remove_redundancy=True
        )

        network = Mock()
        train_network = Mock()
        train_network.parameter_layout_dict = {'layer.weight': Mock(), 'layer.bias': Mock()}

        mock_get_redundancy.return_value = {'layer.weight': [(0, 1, 2, 3)]}
        mock_remove_redundancy.return_value = {0: {'layer.weight'}}

        cur_file = './test.ckpt'
        append_dict = {}

        monitor.remove_redundancy(network, cur_file, append_dict, train_network)

        mock_save_ckpt.assert_called_once()
        mock_logger.info.assert_called()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=8)
    @patch('mindformers.core.callback.callback.get_auto_parallel_context', return_value=1)
    @patch('mindformers.core.callback.callback.context.get_auto_parallel_context',
           return_value='semi_auto_parallel')
    @patch('mindformers.core.callback.callback.get_parameter_redundancy')
    @patch('mindformers.core.callback.callback.remove_param_redundancy')
    @patch('mindformers.core.callback.callback.ms.save_checkpoint')
    @patch('mindformers.core.callback.callback.logger')
    def test_remove_redundancy_without_param_layout(self, mock_logger, mock_save_ckpt,
                                                    mock_remove_redundancy, mock_get_redundancy,
                                                    mock_context, mock_pp, mock_group_size, mock_rank):
        """Test remove_redundancy without param_layout dict"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            directory='./test_ckpt',
            remove_redundancy=True
        )

        network = Mock()
        network.parameter_layout_dict = None

        mock_get_redundancy.return_value = {'layer.weight': [(0, 1, 2, 3)]}
        mock_remove_redundancy.return_value = {0: {'layer.weight'}}

        cur_file = './test.ckpt'
        append_dict = {}

        monitor.remove_redundancy(network, cur_file, append_dict, None)

        mock_save_ckpt.assert_called_once()
        # Verify choice_func is used correctly
        call_args = mock_save_ckpt.call_args
        choice_func = call_args[1].get('choice_func') if call_args[1] else None
        if choice_func:
            assert choice_func('layer.weight')

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=8)
    @patch('mindformers.core.callback.callback.get_auto_parallel_context', return_value=1)
    @patch('mindformers.core.callback.callback.context.get_auto_parallel_context',
           return_value='semi_auto_parallel')
    @patch('mindformers.core.callback.callback.get_parameter_redundancy')
    @patch('mindformers.core.callback.callback.remove_param_redundancy')
    @patch('mindformers.core.callback.callback.logger')
    def test_remove_redundancy_exception_save(self, mock_logger, mock_remove_redundancy, mock_get_redundancy,
                                              mock_context, mock_pp, mock_group_size, mock_rank):
        """Test remove_redundancy with __exception_save__ in append_dict"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            directory='./test_ckpt',
            remove_redundancy=True,
            checkpoint_format='ckpt'
        )

        network = Mock()
        train_network = Mock()
        train_network.parameter_layout_dict = {'layer.weight': Mock()}

        mock_get_redundancy.return_value = {'layer.weight': [(0,)]}
        mock_remove_redundancy.return_value = {0: {'layer.weight'}}

        monitor._do_remove_redundancy_for_tft = Mock()

        cur_file = './test_rank_0/ckpt_1_10.ckpt'
        append_dict = {'__exception_save__': True, 'epoch_num': 1}

        monitor.remove_redundancy(network, cur_file, append_dict, train_network)

        # Should call _do_remove_redundancy_for_tft
        monitor._do_remove_redundancy_for_tft.assert_called_once()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.get_real_group_size', return_value=8)
    @patch('mindformers.core.callback.callback.get_auto_parallel_context', return_value=1)
    @patch('mindformers.core.callback.callback.context.get_auto_parallel_context',
           return_value='semi_auto_parallel')
    @patch('mindformers.core.callback.callback.get_parameter_redundancy')
    @patch('mindformers.core.callback.callback.remove_param_redundancy')
    @patch('mindformers.core.callback.callback.ms.save_checkpoint')
    @patch('mindformers.core.callback.callback.logger')
    def test_remove_redundancy_all_params_non_redundant_warning(self, mock_logger, mock_save_ckpt,
                                                                mock_remove_redundancy, mock_get_redundancy,
                                                                mock_context, mock_pp, mock_group_size, mock_rank):
        """Test remove_redundancy logs warning when all params are non-redundant"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            directory='./test_ckpt',
            remove_redundancy=True
        )

        network = Mock()
        train_network = Mock()
        # param_layout.keys() returns same as save_param_names
        param_layout_dict = {'layer.weight': Mock()}
        train_network.parameter_layout_dict = param_layout_dict

        mock_get_redundancy.return_value = {'layer.weight': [(0,)]}
        mock_remove_redundancy.return_value = {0: param_layout_dict.keys()}

        cur_file = './test.ckpt'
        append_dict = {}

        monitor.remove_redundancy(network, cur_file, append_dict, train_network)

        # Should log warning about non-redundant params
        mock_logger.warning.assert_called()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.context.get_auto_parallel_context',
           return_value='stand_alone')
    @patch('mindformers.core.callback.callback.ms.save_checkpoint')
    def test_remove_redundancy_no_config(self, mock_save_ckpt, mock_context, mock_rank):
        """Test remove_redundancy when remove_redundancy config is False"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            directory='./test_ckpt',
            remove_redundancy=False
        )

        network = Mock()
        cur_file = './test.ckpt'
        append_dict = {}

        monitor.remove_redundancy(network, cur_file, append_dict, None)

        # Should call ms.save_checkpoint without redundancy removal
        mock_save_ckpt.assert_called_once()


class TestCheckpointMonitorSaveCheckpointNetwork:
    """Test CheckpointMonitor.save_checkpoint_network method"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.time.time', return_value=1000.0)
    @patch('mindformers.core.callback.callback.os.makedirs')
    @patch('mindformers.core.callback.callback.context.get_context',
           return_value=1)  # PYNATIVE_MODE
    @patch('mindformers.core.callback.callback.ms.get_auto_parallel_context',
           return_value='stand_alone')
    def test_save_checkpoint_network_trainable_params(self, mock_parallel_ctx, mock_get_ctx,
                                                      mock_makedirs, mock_time, mock_rank):
        """Test save_checkpoint_network with save_trainable_params=True"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            directory='./test_ckpt',
            save_trainable_params=True
        )

        monitor.save_info_list[10] = {
            'ckpt': {'save_start_time': None, 'ckpt_file_path': None, 'save_end_time': None},
            'network': {'save_start_time': None, 'ckpt_file_path': None, 'save_end_time': None},
            'trainable_params': {
                'save_start_time': None, 'ckpt_file_path': None,
                'save_end_time': None}
        }

        # Create mock parameter
        mock_param = Mock()
        mock_param.name = 'layer.weight'
        mock_param.sliced = True
        mock_param.has_init = False
        mock_param.param_info = Mock()
        mock_param.param_info.is_pipeline_shared_param = False
        mock_param.data = Mock()
        mock_param.data.asnumpy.return_value = np.array([1.0, 2.0])

        # Mock network - need optimizer to be non-None so save_obj becomes mock_network.network
        mock_network = Mock()
        mock_network.network = Mock()
        mock_network.network.trainable_params.return_value = [mock_param]
        mock_network.network.init_parameters_data = Mock()
        mock_network.network.parameter_layout_dict = {}
        mock_network.optimizer = Mock()  # Non-None so save_obj = save_obj.network

        cb_params = Mock()
        cb_params.network = mock_network
        cb_params.train_network = Mock()
        cb_params.cur_step_num = 10
        cb_params.cur_epoch_num = 1
        cb_params.batch_num = 100

        monitor._trainable_manager = Mock()
        monitor._trainable_manager.ckpoint_num = 0
        monitor.need_remove_extra_ckpt = False
        monitor.remove_redundancy = Mock()

        monitor.save_checkpoint_network(cb_params)

        # Should call remove_redundancy with param list
        monitor.remove_redundancy.assert_called_once()
        # Verify save_start_time was set
        assert monitor.save_info_list[10]['trainable_params']['save_start_time'] == 1000.0

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.time.time', return_value=1000.0)
    @patch('mindformers.core.callback.callback.os.makedirs')
    def test_save_checkpoint_network_network_params(self, mock_makedirs, mock_time, mock_rank):
        """Test save_checkpoint_network with save_network_params=True"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            directory='./test_ckpt',
            save_network_params=True
        )

        monitor.save_info_list[10] = {
            'ckpt': {'save_start_time': None, 'ckpt_file_path': None, 'save_end_time': None},
            'network': {'save_start_time': None, 'ckpt_file_path': None, 'save_end_time': None},
            'trainable_params': {
                'save_start_time': None, 'ckpt_file_path': None,
                'save_end_time': None}
        }

        mock_network = Mock()
        mock_network.network = Mock()
        mock_network.optimizer = None

        cb_params = Mock()
        cb_params.network = mock_network
        cb_params.train_network = Mock()
        cb_params.cur_step_num = 10
        cb_params.cur_epoch_num = 1
        cb_params.batch_num = 100

        monitor._network_manager = Mock()
        monitor._network_manager.ckpoint_num = 0
        monitor.need_remove_extra_ckpt = True
        monitor.remove_redundancy = Mock()

        monitor.save_checkpoint_network(cb_params)

        # Should call remove_redundancy
        monitor.remove_redundancy.assert_called_once()
        # Should remove oldest ckpt file since need_remove_extra_ckpt is True
        monitor._network_manager.remove_oldest_ckpoint_file.assert_called_once()
        # need_remove_extra_ckpt should be reset to False
        assert not monitor.need_remove_extra_ckpt

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.time.time', return_value=1000.0)
    @patch('mindformers.core.callback.callback.os.makedirs')
    @patch('mindformers.core.callback.callback.context.get_context', return_value=0)  # GRAPH_MODE
    @patch('mindformers.core.callback.callback.ms.get_auto_parallel_context',
           return_value='semi_auto_parallel')
    @patch('mindformers.core.callback.callback._get_merged_param_data')
    def test_save_checkpoint_network_trainable_params_merged(
            self, mock_merged_data, mock_parallel_ctx,
            mock_get_ctx, mock_makedirs,
            mock_time, mock_rank):
        """Test save_checkpoint_network merges param data in auto parallel"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            directory='./test_ckpt',
            save_trainable_params=True
        )

        monitor.save_info_list[10] = {
            'ckpt': {'save_start_time': None, 'ckpt_file_path': None, 'save_end_time': None},
            'network': {'save_start_time': None, 'ckpt_file_path': None, 'save_end_time': None},
            'trainable_params': {
                'save_start_time': None, 'ckpt_file_path': None,
                'save_end_time': None}
        }

        # Create mock parameter that should be saved
        mock_param = Mock()
        mock_param.name = 'layer.weight'
        mock_param.sliced = True
        mock_param.has_init = False
        mock_param.param_info = Mock()
        mock_param.param_info.is_pipeline_shared_param = False
        mock_param.data = Mock()
        mock_param.data.asnumpy.return_value = np.array([1.0, 2.0])

        # Mock network with parameter_layout_dict
        mock_network = Mock()
        mock_network.network = Mock()
        mock_network.network.trainable_params.return_value = [mock_param]
        mock_network.network.init_parameters_data = Mock()
        mock_network.network.parameter_layout_dict = {'layer.weight': Mock()}
        mock_network.optimizer = Mock()  # Non-None so save_obj = save_obj.network

        mock_merged_data.return_value = Mock()

        cb_params = Mock()
        cb_params.network = mock_network
        cb_params.train_network = Mock()
        cb_params.cur_step_num = 10
        cb_params.cur_epoch_num = 1
        cb_params.batch_num = 100

        monitor._trainable_manager = Mock()
        monitor._trainable_manager.ckpoint_num = 0
        monitor.need_remove_extra_ckpt = True
        monitor.remove_redundancy = Mock()

        monitor.save_checkpoint_network(cb_params)

        # Should call _get_merged_param_data
        mock_merged_data.assert_called_once()
        # Should remove oldest file
        monitor._trainable_manager.remove_oldest_ckpoint_file.assert_called_once()


class TestCheckpointMonitorMegatronFormat:
    """Test CheckpointMonitor._save_megatron_ckpt_file_format method"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.get_all_sharded_tensor')
    @patch('mindformers.core.callback.callback.save_checkpoint')
    def test_save_megatron_ckpt_file_format_basic(
            self, mock_save_ckpt, mock_get_sharded, mock_rank):
        """Test _save_megatron_ckpt_file_format basic functionality"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            directory='./test_ckpt',
            use_legacy_format=False,
            save_optimizer=True,
            global_batch_size=64
        )

        monitor._last_triggered_step = 0
        monitor._append_step_num = 0

        mock_get_sharded.return_value = {'layer.weight': Mock()}

        cb_params = Mock()
        cb_params.cur_step_num = 10
        cb_params.cur_epoch_num = 1
        cb_params.network = Mock()
        cb_params.network.network = Mock()
        cb_params.network.optimizer = Mock()
        cb_params.network.optimizer.global_step = 10
        cb_params.network.network.parameters_dict.return_value = {'layer.weight': Mock()}
        cb_params.net_outputs = (Mock(), Mock(), 1024)  # loss, overflow, loss_scale

        monitor._save_megatron_ckpt_file_format(cb_params)

        # Should call save_checkpoint
        mock_save_ckpt.assert_called_once()
        # Should update _last_triggered_step
        assert monitor._last_triggered_step == 10
        # Verify common_info was set
        assert monitor.common_info.step_num == 10
        assert monitor.common_info.epoch_num == 1
        assert monitor.common_info.global_batch_size == 64

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.get_all_sharded_tensor')
    @patch('mindformers.core.callback.callback.save_checkpoint')
    def test_save_megatron_ckpt_file_format_skip_same_step(
            self, mock_save_ckpt, mock_get_sharded, mock_rank):
        """Test _save_megatron_ckpt_file_format skips when same step"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            directory='./test_ckpt',
            use_legacy_format=False
        )

        monitor._last_triggered_step = 10  # Same as cur_step_num

        cb_params = Mock()
        cb_params.cur_step_num = 10

        monitor._save_megatron_ckpt_file_format(cb_params)

        # Should not call save_checkpoint
        mock_save_ckpt.assert_not_called()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.get_all_sharded_tensor')
    @patch('mindformers.core.callback.callback.save_checkpoint')
    def test_save_megatron_ckpt_file_format_no_optimizer(
            self, mock_save_ckpt, mock_get_sharded, mock_rank):
        """Test _save_megatron_ckpt_file_format without optimizer"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            directory='./test_ckpt',
            use_legacy_format=False,
            save_optimizer=False,
            global_batch_size=32
        )

        monitor._last_triggered_step = 0
        monitor._append_step_num = 5

        mock_get_sharded.return_value = {}

        cb_params = Mock()
        cb_params.cur_step_num = 10
        cb_params.cur_epoch_num = 2
        cb_params.network = Mock()
        cb_params.network.network = Mock()
        cb_params.network.optimizer = Mock()
        cb_params.network.optimizer.global_step = 15
        cb_params.network.network.parameters_dict.return_value = {'layer.weight': Mock()}
        cb_params.net_outputs = (Mock(), Mock())  # No loss_scale

        monitor._save_megatron_ckpt_file_format(cb_params)

        # Should call save_checkpoint with optimizer=None
        mock_save_ckpt.assert_called_once()
        call_kwargs = mock_save_ckpt.call_args[1]
        assert call_kwargs.get('optimizer') is None
        # Verify step_num includes append_step_num
        assert monitor.common_info.step_num == 15  # 5 + 10
        # loss_scale should be None since net_outputs doesn't have 3 elements
        assert monitor.common_info.loss_scale is None

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.get_all_sharded_tensor')
    @patch('mindformers.core.callback.callback.save_checkpoint')
    def test_save_megatron_ckpt_file_format_with_async(
            self, mock_save_ckpt, mock_get_sharded, mock_rank):
        """Test _save_megatron_ckpt_file_format with async save"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            directory='./test_ckpt',
            use_legacy_format=False,
            async_save=True,
            save_optimizer=True
        )

        monitor._last_triggered_step = 0
        monitor._append_step_num = 0

        mock_get_sharded.return_value = {}

        cb_params = Mock()
        cb_params.cur_step_num = 20
        cb_params.cur_epoch_num = 1
        cb_params.network = Mock()
        cb_params.network.network = Mock()
        cb_params.network.optimizer = Mock()
        cb_params.network.optimizer.global_step = 20
        cb_params.network.network.parameters_dict.return_value = {}
        cb_params.net_outputs = (Mock(), Mock(), 2048, Mock())  # Has loss_scale

        monitor._save_megatron_ckpt_file_format(cb_params)

        # Should pass async_save_manager to save_checkpoint
        mock_save_ckpt.assert_called_once()
        call_kwargs = mock_save_ckpt.call_args[1]
        assert call_kwargs.get('async_save_manager') is not None
        # Verify loss_scale was extracted
        assert monitor.common_info.loss_scale == 2048.0

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    @patch('mindformers.core.callback.callback.get_all_sharded_tensor')
    @patch('mindformers.core.callback.callback.save_checkpoint')
    def test_save_megatron_ckpt_file_format_filter_func(
            self, mock_save_ckpt, mock_get_sharded, mock_rank):
        """Test _save_megatron_ckpt_file_format uses filter_func when save_optimizer=False"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            directory='./test_ckpt',
            use_legacy_format=False,
            save_optimizer=False
        )

        monitor._last_triggered_step = 0
        monitor._append_step_num = 0

        # Track what filter_func was passed to get_all_sharded_tensor and test it
        filter_func_tested = [False]

        def capture_and_test_filter_func(*args, **kwargs):
            # Get the filter_func and test it immediately
            filter_func = kwargs.get('filter_func')
            assert filter_func is not None, "filter_func should be passed"
            assert callable(filter_func), "filter_func should be callable"
            # Filter should only allow params in parameters_dict
            assert filter_func('layer.weight'), "filter_func should allow 'layer.weight'"
            # 'optimizer.state' is not in parameters_dict, so filter returns False
            assert not filter_func('optimizer.state'), "filter_func should reject 'optimizer.state'"
            filter_func_tested[0] = True
            return {}

        mock_get_sharded.side_effect = capture_and_test_filter_func

        cb_params = Mock()
        cb_params.cur_step_num = 10
        cb_params.cur_epoch_num = 1
        cb_params.network = Mock()
        cb_params.network.network = Mock()
        cb_params.network.optimizer = Mock()
        cb_params.network.optimizer.global_step = 10
        # Only include network params, not optimizer state
        cb_params.network.network.parameters_dict.return_value = {'layer.weight': Mock()}
        cb_params.net_outputs = ()

        monitor._save_megatron_ckpt_file_format(cb_params)

        # Verify that the filter_func was actually tested
        assert filter_func_tested[0], "filter_func should have been tested"


class TestCheckpointMonitorGetCurDpEdgeCases:
    """Test CheckpointMonitor._get_cur_dp edge cases"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    def test_get_cur_dp_no_matching_rank(self, mock_rank):
        """Test _get_cur_dp when rank not in any group"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            directory='./test_ckpt'
        )

        # Rank 0 not in any group - returns empty tuple (initial min_value)
        param_redundancy_dict = {
            'layer.weight': [(1, 2, 3, 4)],
        }

        cur_dp = monitor._get_cur_dp(0, param_redundancy_dict)

        # When rank is not in any group, returns empty tuple (initial min_value)
        assert cur_dp == ()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    def test_get_cur_dp_skip_accu_grads(self, mock_rank):
        """Test _get_cur_dp skips accu_grads parameters"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            directory='./test_ckpt'
        )

        param_redundancy_dict = {
            'accu_grads.layer.weight': [(0, 1, 2, 3)],
            'inputs.data': [(0, 1)],
            'layer.weight': [(0, 1)],
        }

        cur_dp = monitor._get_cur_dp(0, param_redundancy_dict)

        # Should skip accu_grads and inputs, use layer.weight group
        assert 0 in cur_dp
        assert 1 in cur_dp

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_real_rank', return_value=0)
    def test_get_cur_dp_conflicting_groups(self, mock_rank):
        """Test _get_cur_dp with conflicting groups"""

        monitor = CheckpointMonitor(
            prefix='TEST',
            directory='./test_ckpt'
        )

        # Conflicting groups: rank 0 is in (0, 1) but also in (0, 2, 3)
        # where (0, 2, 3) is not a subset of (0, 1)
        param_redundancy_dict = {
            'layer1.weight': [(0, 1)],
            'layer2.weight': [(0, 2, 3)],
        }

        cur_dp = monitor._get_cur_dp(0, param_redundancy_dict)

        # Should return single rank when conflicts exist
        assert cur_dp == (0,)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
