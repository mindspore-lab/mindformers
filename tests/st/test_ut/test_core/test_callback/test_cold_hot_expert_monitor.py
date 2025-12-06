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

from mindformers.core.callback.callback import ColdHotExpertMonitor

# pylint: disable=unused-argument   # for mock logic


class TestColdHotExpertMonitorExtended:
    """Extended tests for ColdHotExpertMonitor"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('os.getenv')
    def test_get_attribute_by_path(self, mock_getenv):
        """Test get_attribute_by_path method"""

        def getenv_side_effect(x, default=None):
            if x == "RANK_ID":
                return "0"
            if x == "RANK_SIZE":
                return "8"
            return default

        mock_getenv.side_effect = getenv_side_effect

        moe_config = Mock()
        moe_config.update_step = 10
        moe_config.expert_num = 8
        moe_config.hot_expert_num = 1
        moe_config.moe_module_name = "model.layers"

        monitor = ColdHotExpertMonitor(
            moe_config=moe_config,
            hidden_size=128,
            ffn_hidden_size=512,
            expert_parallel=1,
            model_parallel=1,
            save_checkpoint_steps=100
        )

        # Create mock object with nested attributes
        obj = Mock()
        obj.model.layers = [Mock(), Mock()]

        result = monitor.get_attribute_by_path(obj, "model.layers")
        assert len(result) == 2


class TestColdHotExpertMonitorBasic:
    """Test ColdHotExpertMonitor basic functionality"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('os.getenv')
    def test_init_basic(self, mock_getenv):
        """Test ColdHotExpertMonitor initialization"""

        def getenv_side_effect(x, default=None):
            if x == "RANK_ID":
                return "0"
            if x == "RANK_SIZE":
                return "8"
            return default

        mock_getenv.side_effect = getenv_side_effect

        moe_config = Mock()
        moe_config.update_step = 10
        moe_config.expert_num = 8
        moe_config.hot_expert_num = 2
        moe_config.moe_module_name = "model.layers"

        monitor = ColdHotExpertMonitor(
            moe_config=moe_config,
            hidden_size=128,
            ffn_hidden_size=512,
            expert_parallel=2,
            model_parallel=2,
            save_checkpoint_steps=100
        )

        assert monitor.update_step == 10
        assert monitor.expert_num == 8
        assert monitor.hot_expert_num == 2
        assert monitor.local_expert_num == 4  # 8 / 2


class TestColdHotExpertMonitorStepEnd:
    """Test ColdHotExpertMonitor.on_train_step_end and expert switching"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_rank', return_value=0)
    @patch('mindformers.core.callback.callback.os.getenv')
    @patch('mindformers.core.callback.callback.time.time')
    def test_on_train_step_end_switch_experts(self, mock_time, mock_getenv, mock_get_rank):
        """Test on_train_step_end triggers expert switching"""

        def getenv_side_effect(key, default=None):
            if key == "RANK_ID":
                return "0"
            if key == "RANK_SIZE":
                return "8"
            return default

        mock_getenv.side_effect = getenv_side_effect

        # Mock time.time() to return incrementing values
        time_counter = [100.0]

        def time_side_effect():
            result = time_counter[0]
            time_counter[0] += 1.0
            return result

        mock_time.side_effect = time_side_effect

        # Use Mock object instead of dict to support attribute access
        moe_config = Mock()
        moe_config.expert_num = 8
        moe_config.hot_expert_num = 1
        moe_config.moe_module_name = 'network.blocks'
        moe_config.update_step = 10

        monitor = ColdHotExpertMonitor(
            moe_config=moe_config,
            hidden_size=4096,
            ffn_hidden_size=16384,
            expert_parallel=1,
            model_parallel=1,
            save_checkpoint_steps=10
        )

        # Mock the train_network and blocks
        run_context = Mock()
        cb_params = Mock()
        cb_params.cur_step_num = 10
        cb_params.train_network = Mock()

        # Mock blocks structure
        mock_block = Mock()
        mock_block.output.hot_expert_index.value.return_value = [np.array([0])]

        monitor.get_attribute_by_path = Mock(return_value=[mock_block])
        monitor.return_back_hot_expert = Mock()
        monitor.switch_hot_expert = Mock()

        run_context.original_args.return_value = cb_params

        monitor.on_train_step_end(run_context)

        monitor.switch_hot_expert.assert_called()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_rank', return_value=0)
    @patch('mindformers.core.callback.callback.os.getenv')
    def test_return_back_hot_expert_single(self, mock_getenv, mock_get_rank):
        """Test return_back_hot_expert with single hot expert"""

        def getenv_side_effect(key, default=None):
            if key == "RANK_ID":
                return "0"
            if key == "RANK_SIZE":
                return "8"
            return default

        mock_getenv.side_effect = getenv_side_effect

        # Use Mock object instead of dict to support attribute access
        moe_config = Mock()
        moe_config.expert_num = 8
        moe_config.hot_expert_num = 1
        moe_config.moe_module_name = 'network.blocks'
        moe_config.update_step = 10

        monitor = ColdHotExpertMonitor(
            moe_config=moe_config,
            hidden_size=4096,
            ffn_hidden_size=16384,
            expert_parallel=1,
            model_parallel=1,
            save_checkpoint_steps=10
        )

        # Mock block with hot expert - need to support subscript access
        mock_block = Mock()

        # old_hot_expert_index[0] needs to be subscriptable
        # value()[0] should return an array-like object that supports indexing
        mock_hot_expert_index = np.array([0])  # Use numpy array for proper indexing support
        mock_block.output.hot_expert_index.value.return_value = [mock_hot_expert_index]

        # Create mock arrays that support subscript assignment
        # For weight arrays - simple list
        mock_weight_array = [Mock() for _ in range(8)]

        # For bias arrays - need nested structure that supports bias[0][ffn_index][0] = value
        # Create a list of lists where each inner list contains Mock objects
        mock_bias_inner = [[Mock()] for _ in range(8)]
        mock_bias_array = [mock_bias_inner]

        mock_block.output.ffn.mapping.weight = mock_weight_array
        mock_block.output.ffn.mapping.bias = mock_bias_array
        mock_block.output.ffn.projection.weight = [Mock() for _ in range(8)]
        mock_block.output.ffn.projection.bias = [[[Mock()] for _ in range(8)]]

        mock_block.output.mlp.mapping.weight = Mock()
        mock_block.output.mlp.mapping.bias = Mock()
        mock_block.output.mlp.projection.weight = Mock()
        mock_block.output.mlp.projection.bias = Mock()

        # Should not raise error
        monitor.return_back_hot_expert(mock_block)


class TestColdHotExpertMonitorSwitchExpert:
    """Test ColdHotExpertMonitor.switch_hot_expert method"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_rank', return_value=0)
    @patch('mindformers.core.callback.callback.os.getenv')
    def test_switch_hot_expert_single_no_change(self, mock_getenv, mock_get_rank):
        """Test switch_hot_expert when expert doesn't change"""

        def getenv_side_effect(key, default=None):
            if key == "RANK_ID":
                return "0"
            if key == "RANK_SIZE":
                return "8"
            return default

        mock_getenv.side_effect = getenv_side_effect

        moe_config = Mock()
        moe_config.expert_num = 8
        moe_config.hot_expert_num = 1
        moe_config.moe_module_name = 'network.blocks'
        moe_config.update_step = 10

        monitor = ColdHotExpertMonitor(
            moe_config=moe_config,
            hidden_size=4096,
            ffn_hidden_size=16384,
            expert_parallel=1,
            model_parallel=1,
            save_checkpoint_steps=10
        )

        # Mock block where old and new expert are the same
        mock_block = Mock()

        # Old expert index - should be array-like supporting indexing
        # value()[0] should return an array, and then [0] accesses first element
        old_expert_array = np.array([0])  # Array that supports [0] indexing
        mock_block.output.hot_expert_index.value.return_value = [old_expert_array]

        # New expert index (same as old)
        # cumsum_value.value() returns a tensor
        mock_cumsum = Mock()

        # Create a mock tensor that supports slicing and indexing
        # new_expert_index[0:1] should return np.array([0])
        # new_expert_index[1:8] should return an array
        def mock_getitem(self, key):
            # Need to accept self parameter since this is bound as a method
            if isinstance(key, slice):
                # Handle slicing
                if key.start == 0 and key.stop == 1:  # hot_expert_num = 1
                    return np.array([0])
                return np.array(list(range(key.start or 0, key.stop or 8)))
            # Handle single index
            return 0

        mock_expert_indices = Mock()
        mock_expert_indices.__getitem__ = mock_getitem
        mock_cumsum.topk.return_value = (Mock(), mock_expert_indices)
        mock_block.output.router.router.cumsum_value.value.return_value = mock_cumsum

        # Should return early without switching (since old and new expert are the same)
        monitor.switch_hot_expert(mock_block, cur_step_num=2)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
