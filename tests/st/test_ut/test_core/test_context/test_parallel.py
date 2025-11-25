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
"""Test parallel.py"""
# pylint: disable=protected-access
from unittest.mock import patch

import pytest
import mindspore.context as ms_context

from mindformers.core.context.parallel import ParallelOperator
from mindformers.tools.register import MindFormerConfig
from mindformers.modules.transformer.transformer import TransformerOpParallelConfig


@pytest.fixture(name="mock_config")
def fixture_mock_config():
    """Create a mock config for testing."""
    config = MindFormerConfig(
        use_parallel=False,
        parallel={},
        parallel_config={}
    )
    return config


@pytest.fixture(name="mock_config_with_parallel")
def fixture_mock_config_with_parallel():
    """Create a mock config with parallel enabled."""
    config = MindFormerConfig(
        use_parallel=True,
        parallel={
            'parallel_mode': 'semi_auto_parallel',
            'full_batch': True
        },
        parallel_config={
            'data_parallel': 2,
            'model_parallel': 2,
            'pipeline_stage': 1
        }
    )
    return config


class TestParallelOperator:
    """Test ParallelOperator class."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_init_without_parallel(self, mock_config):
        """
        Feature: Test ParallelOperator initialization without parallel.
        Description: Test that ParallelOperator can be initialized with use_parallel=False.
        Expectation: ParallelOperator is initialized successfully.
        """
        operator = ParallelOperator(mock_config)
        assert operator.parallel_ctx is not None
        assert operator.parallel is not None
        assert not hasattr(operator, 'config')

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_init_with_parallel(self, mock_config_with_parallel):
        """
        Feature: Test ParallelOperator initialization with parallel.
        Description: Test that ParallelOperator can be initialized with use_parallel=True.
        Expectation: ParallelOperator is initialized successfully.
        """
        operator = ParallelOperator(mock_config_with_parallel)
        assert operator.parallel_ctx is not None
        assert operator.parallel is not None

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_set_pipeline_stage(self):
        """
        Feature: Test _set_pipeline_stage method.
        Description: Test pipeline stage setting logic.
        Expectation: Pipeline stage is set correctly.
        """
        config = MindFormerConfig(
            use_parallel=True,
            parallel={
                'auto_pipeline': False
            },
            parallel_config={
                'pipeline_stage': 2
            }
        )
        operator = ParallelOperator(config)
        assert operator.parallel.pipeline_stage == 2

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_set_pipeline_stage_with_auto_pipeline(self):
        """
        Feature: Test _set_pipeline_stage with auto_pipeline=True.
        Description: Test that auto_pipeline raises ValueError.
        Expectation: ValueError is raised when auto_pipeline is True.
        """
        config = MindFormerConfig(
            use_parallel=True,
            parallel={
                'auto_pipeline': True
            },
            parallel_config={
                'pipeline_stage': 2
            }
        )
        with pytest.raises(ValueError) as exc_info:
            ParallelOperator(config)
        assert "Automatic pipeline stage is unavailable" in str(exc_info.value)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_set_pipeline_stage_single_stage(self):
        """
        Feature: Test _set_pipeline_stage with single stage.
        Description: Test pipeline stage with value 1.
        Expectation: Pipeline stages is not set when final_stages <= 1.
        """
        config = MindFormerConfig(
            use_parallel=True,
            parallel={
                'auto_pipeline': False
            },
            parallel_config={
                'pipeline_stage': 1
            }
        )
        operator = ParallelOperator(config)
        assert operator.parallel.pipeline_stage == 1

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_set_pipeline_stage_none(self):
        """
        Feature: Test _set_pipeline_stage with None pipeline_stage.
        Description: Test pipeline stage with None value.
        Expectation: Pipeline stage defaults to 1.
        """
        config = MindFormerConfig(
            use_parallel=True,
            parallel={
                'auto_pipeline': False
            },
            parallel_config={}
        )
        operator = ParallelOperator(config)
        assert operator.parallel.pipeline_stage == 1

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_get_parallel_ctx_config_with_full_batch(self):
        """
        Feature: Test _get_parallel_ctx_config with full_batch.
        Description: Test that full_batch is set to False for non-parallel modes.
        Expectation: full_batch is False when parallel_mode is not SEMI_AUTO_PARALLEL or AUTO_PARALLEL.
        """
        config = MindFormerConfig(
            use_parallel=False,
            parallel={
                'parallel_mode': 'stand_alone',
                'full_batch': True
            },
            parallel_config={}
        )
        operator = ParallelOperator(config)
        assert operator.parallel_ctx.get('full_batch') is False

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_get_parallel_ctx_config_with_semi_auto_parallel(self):
        """
        Feature: Test _get_parallel_ctx_config with SEMI_AUTO_PARALLEL.
        Description: Test that full_batch is preserved for SEMI_AUTO_PARALLEL mode.
        Expectation: full_batch remains True for SEMI_AUTO_PARALLEL mode.
        """
        config = MindFormerConfig(
            use_parallel=False,
            parallel={
                'parallel_mode': ms_context.ParallelMode.SEMI_AUTO_PARALLEL,
                'full_batch': True
            },
            parallel_config={}
        )
        operator = ParallelOperator(config)
        assert operator.parallel_ctx.get('full_batch') is True

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_get_parallel_ctx_config_with_auto_parallel(self):
        """
        Feature: Test _get_parallel_ctx_config with AUTO_PARALLEL.
        Description: Test that full_batch is preserved for AUTO_PARALLEL mode.
        Expectation: full_batch remains True for AUTO_PARALLEL mode.
        """
        config = MindFormerConfig(
            use_parallel=False,
            parallel={
                'parallel_mode': ms_context.ParallelMode.AUTO_PARALLEL,
                'full_batch': True
            },
            parallel_config={}
        )
        operator = ParallelOperator(config)
        assert operator.parallel_ctx.get('full_batch') is True

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_get_parallel_ctx_config_without_full_batch(self):
        """
        Feature: Test _get_parallel_ctx_config without full_batch.
        Description: Test that function works without full_batch key.
        Expectation: Function executes successfully.
        """
        config = MindFormerConfig(
            use_parallel=False,
            parallel={
                'parallel_mode': 'stand_alone'
            },
            parallel_config={}
        )
        operator = ParallelOperator(config)
        assert 'full_batch' not in operator.parallel_ctx or operator.parallel_ctx.get('full_batch') is None

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_get_parallel_config_dict(self):
        """
        Feature: Test _get_parallel_config with dict.
        Description: Test that dict parallel_config is handled correctly.
        Expectation: ParallelConfig is created from dict.
        """
        config = MindFormerConfig(
            use_parallel=False,
            parallel={},
            parallel_config={
                'data_parallel': 2,
                'model_parallel': 4
            }
        )
        operator = ParallelOperator(config)
        assert operator.parallel.data_parallel == 2
        assert operator.parallel.model_parallel == 4

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_get_parallel_config_transformer_op(self):
        """
        Feature: Test _get_parallel_config with TransformerOpParallelConfig.
        Description: Test that TransformerOpParallelConfig is converted to dict.
        Expectation: ParallelConfig is created from TransformerOpParallelConfig.
        """
        transformer_config = TransformerOpParallelConfig(
            data_parallel=2,
            model_parallel=4
        )
        config = MindFormerConfig(
            use_parallel=False,
            parallel={},
            parallel_config=transformer_config
        )
        operator = ParallelOperator(config)
        assert operator.parallel.data_parallel == 2
        assert operator.parallel.model_parallel == 4

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.context.parallel.init')
    @patch('mindformers.core.context.parallel.get_group_size')
    @patch('mindformers.core.context.parallel.get_rank')
    @patch('mindformers.core.context.parallel.context.reset_auto_parallel_context')
    @patch('mindformers.core.context.parallel.context.get_auto_parallel_context')
    def test_init_communication_success(
            self, mock_get_context, mock_reset_context, mock_get_rank, mock_get_group_size, mock_init):
        """
        Feature: Test init_communication method success.
        Description: Test that communication is initialized successfully.
        Expectation: Communication is initialized and rank/device_num are returned.
        """
        mock_get_rank.return_value = 0
        mock_get_group_size.return_value = 8
        mock_get_context.return_value = "semi_auto_parallel"

        config = MindFormerConfig(
            use_parallel=True,
            parallel={
                'parallel_mode': 'semi_auto_parallel'
            },
            parallel_config={}
        )
        operator = ParallelOperator(config)
        rank, device_num = operator.init_communication()

        assert rank == 0
        assert device_num == 8
        mock_init.assert_called_once()
        mock_get_group_size.assert_called_once()
        mock_reset_context.assert_called_once()
        assert operator.parallel_ctx['device_num'] == 8

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.context.parallel.init')
    def test_init_communication_failure(self, mock_init):
        """
        Feature: Test init_communication method failure.
        Description: Test that communication initialization failure is handled.
        Expectation: Exception is raised with appropriate error message.
        """
        mock_init.side_effect = Exception("Communication init failed")

        config = MindFormerConfig(
            use_parallel=True,
            parallel={},
            parallel_config={}
        )
        operator = ParallelOperator(config)

        with pytest.raises(Exception) as exc_info:
            operator.init_communication()
        assert "Communication init failed" in str(exc_info.value)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.context.parallel.context.set_auto_parallel_context')
    def test_set_ms_auto_parallel_context_with_full_batch_false(self, mock_set_context):
        """
        Feature: Test _set_ms_auto_parallel_context with full_batch=False.
        Description: Test dataset_strategy conversion from list to tuple.
        Expectation: dataset_strategy is converted to tuple when full_batch is False.
        """
        parallel_ctx = {
            'full_batch': False,
            'dataset_strategy': [[1, 2], [1, 4]]
        }
        ParallelOperator._set_ms_auto_parallel_context(**parallel_ctx)
        mock_set_context.assert_called_once()
        call_args = mock_set_context.call_args[1]
        assert isinstance(call_args['dataset_strategy'], tuple)
        assert call_args['dataset_strategy'] == ((1, 2), (1, 4))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.context.parallel.context.set_auto_parallel_context')
    def test_set_ms_auto_parallel_context_with_non_list_dataset_strategy(self, mock_set_context):
        """
        Feature: Test _set_ms_auto_parallel_context with non-list dataset_strategy.
        Description: Test that non-list dataset_strategy is not converted.
        Expectation: Non-list dataset_strategy remains unchanged.
        """
        parallel_ctx = {
            'full_batch': False,
            'dataset_strategy': "data_parallel"
        }
        ParallelOperator._set_ms_auto_parallel_context(**parallel_ctx)
        mock_set_context.assert_called_once()
        call_args = mock_set_context.call_args[1]
        assert call_args.get('dataset_strategy') is None

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.context.parallel.context.get_auto_parallel_context')
    @patch('mindformers.core.context.parallel.set_algo_parameters')
    @patch('mindformers.core.context.parallel._set_multi_subgraphs')
    def test_set_ms_parallel_auto_parallel(self, mock_set_multi_subgraphs,
                                            mock_set_algo_parameters, mock_get_context):
        """
        Feature: Test _set_ms_parallel with auto_parallel mode.
        Description: Test algo parameters for auto_parallel mode.
        Expectation: elementwise_op_strategy_follow=False and fully_use_devices=False.
        """
        mock_get_context.return_value = "auto_parallel"
        ParallelOperator._set_ms_parallel()
        mock_set_algo_parameters.assert_called_once_with(
            elementwise_op_strategy_follow=False,
            fully_use_devices=False
        )
        mock_set_multi_subgraphs.assert_called_once()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.context.parallel.context.get_auto_parallel_context')
    @patch('mindformers.core.context.parallel.set_algo_parameters')
    @patch('mindformers.core.context.parallel._set_multi_subgraphs')
    def test_set_ms_parallel_semi_auto_parallel(self, mock_set_multi_subgraphs,
                                                 mock_set_algo_parameters, mock_get_context):
        """
        Feature: Test _set_ms_parallel with semi_auto_parallel mode.
        Description: Test algo parameters for semi_auto_parallel mode.
        Expectation: elementwise_op_strategy_follow=True and fully_use_devices=True.
        """
        mock_get_context.return_value = "semi_auto_parallel"
        ParallelOperator._set_ms_parallel()
        mock_set_algo_parameters.assert_called_once_with(
            elementwise_op_strategy_follow=True,
            fully_use_devices=True
        )
        mock_set_multi_subgraphs.assert_called_once()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.context.parallel.context.get_auto_parallel_context')
    @patch('mindformers.core.context.parallel.initialize_model_parallel')
    def test_set_manmul_parallel_stand_alone(self, mock_init_model_parallel, mock_get_context):
        """
        Feature: Test _set_manmul_parallel with stand_alone mode.
        Description: Test that model parallel is initialized for stand_alone mode.
        Expectation: initialize_model_parallel is called with correct parameters.
        """
        mock_get_context.return_value = "stand_alone"
        config = MindFormerConfig(
            use_parallel=False,
            parallel={},
            parallel_config={
                'model_parallel': 2,
                'data_parallel': 4,
                'pipeline_stage': 1,
                'expert_parallel': 1
            }
        )
        operator = ParallelOperator(config)
        operator._set_manmul_parallel()

        mock_init_model_parallel.assert_called_once_with(
            tensor_model_parallel_size=2,
            data_parallel_size=4,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            order="tp-ep-pp-dp"
        )

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.context.parallel.context.get_auto_parallel_context')
    @patch('mindformers.core.context.parallel.initialize_model_parallel')
    def test_set_manmul_parallel_not_stand_alone(self, mock_init_model_parallel, mock_get_context):
        """
        Feature: Test _set_manmul_parallel with non-stand_alone mode.
        Description: Test that model parallel is not initialized for non-stand_alone mode.
        Expectation: initialize_model_parallel is not called.
        """
        mock_get_context.return_value = "semi_auto_parallel"
        config = MindFormerConfig(
            use_parallel=False,
            parallel={},
            parallel_config={}
        )
        operator = ParallelOperator(config)
        operator._set_manmul_parallel()

        mock_init_model_parallel.assert_not_called()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.context.parallel.context.get_auto_parallel_context')
    @patch('mindformers.core.context.parallel.initialize_model_parallel')
    def test_set_manmul_parallel_with_all_parallel_types(self, mock_init_model_parallel, mock_get_context):
        """
        Feature: Test _set_manmul_parallel with all parallel types.
        Description: Test parallel strategy string generation.
        Expectation: Parallel strategy string includes all parallel types > 1.
        """
        mock_get_context.return_value = "stand_alone"
        config = MindFormerConfig(
            use_parallel=False,
            parallel={},
            parallel_config={
                'model_parallel': 2,
                'data_parallel': 4,
                'pipeline_stage': 2,
                'expert_parallel': 2
            }
        )
        operator = ParallelOperator(config)
        operator._set_manmul_parallel()

        mock_init_model_parallel.assert_called_once()
        call_args = mock_init_model_parallel.call_args[1]
        assert call_args['tensor_model_parallel_size'] == 2
        assert call_args['data_parallel_size'] == 4
        assert call_args['pipeline_model_parallel_size'] == 2
        assert call_args['expert_model_parallel_size'] == 2

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.context.parallel.context.get_auto_parallel_context')
    @patch('mindformers.core.context.parallel.initialize_model_parallel')
    def test_set_manmul_parallel_with_default_values(self, mock_init_model_parallel, mock_get_context):
        """
        Feature: Test _set_manmul_parallel with default values.
        Description: Test that default values are used when attributes are missing.
        Expectation: Default values (1) are used for missing attributes.
        """
        mock_get_context.return_value = "stand_alone"
        config = MindFormerConfig(
            use_parallel=False,
            parallel={},
            parallel_config={}
        )
        operator = ParallelOperator(config)
        operator._set_manmul_parallel()

        mock_init_model_parallel.assert_called_once()
        call_args = mock_init_model_parallel.call_args[1]
        assert call_args['tensor_model_parallel_size'] == 1
        assert call_args['data_parallel_size'] == 1
        assert call_args['pipeline_model_parallel_size'] == 1
        assert call_args['expert_model_parallel_size'] == 1
