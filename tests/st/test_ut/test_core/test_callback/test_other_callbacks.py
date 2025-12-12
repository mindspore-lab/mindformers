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

import pytest

from mindformers.core.callback.callback import (
    EvalCallBack,
    MaxLogitsMonitor,
    MoEDropRateCallback,
    StressDetectCallBack,
    SummaryMonitor,
    TopkBiasBalanceCallback,
    TrainCallBack
)

# pylint: disable=unused-argument   # for mock logic


class TestSummaryMonitor:
    """Test SummaryMonitor class"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.SummaryCollector')
    @patch('mindformers.core.callback.callback.get_output_subpath')
    @patch('mindformers.core.callback.callback.get_real_rank')
    def test_init(self, mock_get_real_rank, mock_get_output_subpath, mock_summary_collector):
        """Test initialization"""
        mock_get_real_rank.return_value = 0
        mock_get_output_subpath.return_value = "/tmp/summary"

        SummaryMonitor(summary_dir=None)

        mock_summary_collector.assert_called_once()
        _, kwargs = mock_summary_collector.call_args
        assert kwargs['summary_dir'] == "/tmp/summary"


class TestEvalCallBack:
    """Test EvalCallBack class"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_on_train_epoch_end(self):
        """Test on_train_epoch_end callback"""
        eval_func = Mock()
        callback = EvalCallBack(eval_func, epoch_interval=2)

        run_context = Mock()
        cb_params = Mock()

        # Epoch 1: no eval
        cb_params.cur_epoch_num = 1
        run_context.original_args.return_value = cb_params
        callback.on_train_epoch_end(run_context)
        eval_func.assert_not_called()

        # Epoch 2: eval
        cb_params.cur_epoch_num = 2
        run_context.original_args.return_value = cb_params
        callback.on_train_epoch_end(run_context)
        eval_func.assert_called_once()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_on_train_step_end(self):
        """Test on_train_step_end callback"""
        eval_func = Mock()
        callback = EvalCallBack(eval_func, step_interval=10, epoch_interval=-1)

        run_context = Mock()
        cb_params = Mock()

        # Step 5: no eval
        cb_params.cur_step_num = 5
        run_context.original_args.return_value = cb_params
        callback.on_train_step_end(run_context)
        eval_func.assert_not_called()

        # Step 10: eval
        cb_params.cur_step_num = 10
        run_context.original_args.return_value = cb_params
        callback.on_train_step_end(run_context)
        eval_func.assert_called_once()


class TestTrainCallBack:
    """Test TrainCallBack class"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_stop_step(self):
        """Test stop_step functionality"""
        callback = TrainCallBack(stop_step=10)
        run_context = Mock()
        cb_params = Mock()
        run_context.original_args.return_value = cb_params

        # Step 5
        cb_params.cur_step_num = 5
        callback.on_train_step_end(run_context)
        run_context.request_stop.assert_not_called()

        # Step 10
        cb_params.cur_step_num = 10
        callback.on_train_step_end(run_context)
        run_context.request_stop.assert_called_once()


class TestStressDetectCallBack:
    """Test StressDetectCallBack class"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.stress_detect')
    def test_stress_detect(self, mock_stress_detect):
        """Test stress detection functionality"""
        callback = StressDetectCallBack(detection_interval=10, num_detections=2, dataset_size=100)
        run_context = Mock()
        cb_params = Mock()
        run_context.original_args.return_value = cb_params

        # Step 5: no detect
        cb_params.cur_step_num = 5
        callback.on_train_step_end(run_context)
        mock_stress_detect.assert_not_called()

        # Step 10: detect
        cb_params.cur_step_num = 10
        mock_stress_detect.return_value = 0
        callback.on_train_step_end(run_context)
        assert mock_stress_detect.call_count == 2


class TestMaxLogitsMonitor:
    """Test MaxLogitsMonitor"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_on_train_step_end(self):
        """Test on_train_step_end callback"""
        callback = MaxLogitsMonitor()
        run_context = Mock()
        cb_params = Mock()

        # Create a network structure where 'network' attribute chain terminates
        # The leaf node MUST NOT have a 'network' attribute to break the loop.
        leaf_network = Mock()
        del leaf_network.network
        leaf_network.reset_max_attention_logit = Mock()

        # intermediate network
        network = Mock()
        network.network = leaf_network

        cb_params.train_network = network

        run_context.original_args.return_value = cb_params

        with patch('mindformers.core.callback.callback.get_auto_parallel_context') \
                as mock_get_parallel:
            mock_get_parallel.return_value = "stand_alone"
            callback.on_train_step_end(run_context)

        leaf_network.reset_max_attention_logit.assert_called_once()


class TestTopkBiasBalanceCallback:
    """Test TopkBiasBalanceCallback"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindspore.context.get_auto_parallel_context')
    @patch('mindformers.core.callback.callback.get_tensorboard_writer')
    @patch('mindformers.core.callback.callback.get_tensorboard_args')
    def test_update_topk_bias(self, mock_args, mock_writer, mock_get_parallel):
        """Test topk bias update functionality"""
        mock_args.return_value = {'log_expert_load_to_tensorboard': False}
        mock_get_parallel.return_value = 1  # pipeline stages

        # We need to mock P.Assign etc, which are used in __init__
        with patch('mindspore.ops.operations.Assign'), \
                patch('mindspore.ops.operations.Sub'), \
                patch('mindspore.ops.operations.Add'), \
                patch('mindspore.ops.operations.Sign'), \
                patch('mindspore.ops.operations.Mul'), \
                patch('mindspore.ops.operations.Div'):
            callback = TopkBiasBalanceCallback(balance_via_topk_bias=True, expert_num=2)

        # Setup network structure for _update_topk_bias logic
        # Ensure leaf_network does not have 'network' attribute to terminate loop
        leaf_network = Mock()
        del leaf_network.network

        layer = Mock()
        router_inner = Mock()
        mock_expert_load = Mock()
        mock_expert_load.sum.return_value = 2.0
        router_inner.expert_load.value.return_value = mock_expert_load
        router_inner.topk_bias.value.return_value = Mock()

        router = Mock()
        router.router = router_inner

        routed_experts = Mock()
        routed_experts.router = router

        feed_forward = Mock()
        feed_forward.routed_experts = routed_experts

        layer.feed_forward = feed_forward
        leaf_network.model.layers = [layer]

        network = Mock()
        network.network = leaf_network

        run_context = Mock()
        cb_params = Mock()
        cb_params.train_network = network
        run_context.original_args.return_value = cb_params

        ctx_patch = 'mindformers.core.callback.callback.get_auto_parallel_context'
        with patch(ctx_patch, return_value="stand_alone"):
            callback.on_train_step_end(run_context)


class TestMoEDropRateCallback:
    """Test MoEDropRateCallback"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_callback_droprate(self):
        """Test MoEDropRateCallback - simplified version that just verifies initialization"""
        # Test initialization
        callback = MoEDropRateCallback(expert_num=8, capacity_factor=1.1, num_layers=1, mtp_depth=0)

        # Verify basic attributes
        assert callback.capacity_factor_over_expert_num == 1.1 / 8
        assert callback.num_layers == 1

        # Test with mock network that has no routed_experts (skip the callback logic)
        leaf_network = Mock()
        del leaf_network.network

        layer = Mock()
        # Make feed_forward not have routed_experts attribute
        layer.feed_forward = Mock(spec=[])

        leaf_network.model.layers = [layer]

        network = Mock()
        network.network = leaf_network

        run_context = Mock()
        cb_params = Mock()
        cb_params.train_network = network
        run_context.original_args.return_value = cb_params

        # Mock to avoid entering the complex logic
        ctx_patch = 'mindformers.core.callback.callback.get_auto_parallel_context'
        with patch(ctx_patch, return_value="stand_alone"):
            # This should not raise any errors
            callback.on_train_step_end(run_context)


class TestTopkBiasBalanceCallbackExtended:
    """Extended tests for TopkBiasBalanceCallback"""

    @patch('mindformers.core.callback.callback.get_tensorboard_args',
           return_value={'log_expert_load_to_tensorboard': True})
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_tensorboard_writer', return_value=Mock())
    def test_log_expert_load_to_tensorboard(self, *mocks):
        """Test logging expert load to tensorboard"""

        callback = TopkBiasBalanceCallback(
            balance_via_topk_bias=False,
            topk_bias_update_rate=0.01,
            expert_num=8,
            micro_batch_num=2,
            gradient_accumulation_steps=4
        )

        assert callback.tensor_writer is not None

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
