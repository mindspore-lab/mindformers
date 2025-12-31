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

from mindformers.core.callback.callback import ExpertMigrateCallback
from mindformers.parallel_core.transformer_config import TransformerConfig


# pylint: disable=unused-argument   # for mock logic


class TestExpertMigrateCallback:
    """Test ExpertMigrateCallback"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_rank', return_value=0)
    def test_init(self, mock_rank):
        """Test ExpertMigrateCallback initialization and on_train_step_end."""
        config = TransformerConfig(kv_channels=1, num_layers=1)
        config.pipeline_model_parallel_size = 1
        config.data_parallel_size = 1
        config.tensor_model_parallel_size = 1
        config.expert_model_parallel_size = 1
        config.num_layers = 2
        config.mtp_num_layers = 0
        config.num_moe_experts = 8

        callback = ExpertMigrateCallback(config=config, print_expert_load=True)

        run_context = Mock()
        cb_params = Mock()

        real_network = Mock()
        # Need to ensure loop terminates: while hasattr(network, 'network')
        # We can just not give it a 'network' attribute
        del real_network.network

        layer = Mock()
        layer.pipeline_stage = 0
        layer.mlp.experts.num_tokens_per_expert = Mock()
        layer.mlp.num_local_experts = 8
        layer.mlp.expert_load_history.asnumpy.return_value = np.zeros(8)

        real_network.model.decoder.layers = [layer, layer]

        cb_params.train_network = real_network
        cb_params.optimizer = Mock()
        run_context.original_args.return_value = cb_params

        ctx_patch = 'mindformers.core.callback.callback.get_auto_parallel_context'
        with patch(ctx_patch, return_value="stand_alone"):
            callback.on_train_step_end(run_context)

        layer.mlp.update_expert_load_history.assert_called()


class TestExpertMigrateCallbackExtended:
    """Extended tests for ExpertMigrateCallback"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_rank', return_value=0)
    def test_expert_migrate_with_mtp_layers(self, mock_rank):
        """Test ExpertMigrateCallback with MTP layers"""

        config = TransformerConfig(kv_channels=1, num_layers=1)
        config.pipeline_model_parallel_size = 1
        config.data_parallel_size = 1
        config.tensor_model_parallel_size = 1
        config.expert_model_parallel_size = 1
        config.num_layers = 2
        config.mtp_num_layers = 1
        config.num_moe_experts = 4

        callback = ExpertMigrateCallback(config=config, print_expert_load=False)

        assert callback.mtp_num_layers == 1
        assert callback.num_layers == 2

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
