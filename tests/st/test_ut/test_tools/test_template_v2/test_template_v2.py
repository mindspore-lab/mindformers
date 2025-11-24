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
"""Comprehensive Test Suite for Template V2 Module"""
import pytest

from mindformers.tools.register.llm_template_v2 import (
    TrainingConfig,
    RecomputeConfig,
    SwapConfig,
    TrainingParallelConfig,
    InferParallelConfig,
    ParallelContextConfig,
    InferParallelContextConfig,
    MonitorConfig,
    TensorBoardConfig,
    ProfileConfig,
    OptimizerConfig,
    LrScheduleConfig,
    TrainingGeneralConfig,
    InferGeneralConfig,
    ContextConfig,
    InferContextConfig,
    TrainingDatasetConfig,
    MegatronDataLoaderConfig,
    HFDataLoader,
    ConfigTemplate,
    CheckpointConfig
)


# ============================================================================
# Training Config Tests
# ============================================================================
class TestTrainingConfig:
    """Test TrainingConfig class"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_default_values(self):
        """Test default values of TrainingConfig"""
        default_config = TrainingConfig.default_value()
        assert default_config['micro_batch_size'] == 1
        assert default_config['global_batch_size'] == 512
        assert default_config['epochs'] == 1
        assert default_config['training_seed'] == 1234
        assert default_config['dataset_seed'] == 1234
        assert default_config['check_for_nan_in_loss_and_grad'] is False
        assert default_config['scale_sense'] == 1.0
        assert default_config['use_clip_grad'] is True
        assert default_config['max_grad_norm'] == 1.0
        assert default_config['gradient_accumulation_steps'] == 1
        assert default_config['resume_training'] is False
        assert default_config['sink_mode'] is True
        assert default_config['sink_size'] == 1

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_with_valid_config(self):
        """Test apply method with valid configuration"""
        config = {
            'micro_batch_size': 2,
            'global_batch_size': 64,
            'epochs': 3,
            'training_seed': 42
        }
        result = TrainingConfig.apply('training_args', config)
        assert result['micro_batch_size'] == 2
        assert result['global_batch_size'] == 64
        assert result['epochs'] == 3
        assert result['training_seed'] == 42

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_valid(self):
        """Test validate_config with valid configuration"""
        config = {
            'micro_batch_size': 4,
            'global_batch_size': 128,
            'epochs': 2,
            'training_seed': 100,
            'use_clip_grad': True,
            'max_grad_norm': 1.0,
            'sink_mode': True,
            'sink_size': 1
        }
        # Should not raise exception
        TrainingConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_micro_batch_size(self):
        """Test validate_config with invalid micro_batch_size"""
        config = {
            'micro_batch_size': 0,  # Should be >= 1
        }
        with pytest.raises(ValueError):
            TrainingConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_global_batch_size(self):
        """Test validate_config with invalid global_batch_size"""
        config = {
            'global_batch_size': 0,  # Should be >= 1
        }
        with pytest.raises(ValueError):
            TrainingConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_epochs(self):
        """Test validate_config with invalid epochs"""
        config = {
            'epochs': 0,  # Should be >= 1
        }
        with pytest.raises(ValueError):
            TrainingConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_sink_mode(self):
        """Test validate_config with invalid sink_mode"""
        config = {
            'sink_mode': False,  # Only True is supported
        }
        with pytest.raises(ValueError):
            TrainingConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_sink_size(self):
        """Test validate_config with invalid sink_size"""
        config = {
            'sink_size': 2,  # Only 1 is supported
        }
        with pytest.raises(ValueError):
            TrainingConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_type(self):
        """Test validate_config with invalid parameter type"""
        config = {
            'micro_batch_size': '1',  # Should be int
        }
        with pytest.raises(ValueError):
            TrainingConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_gradient_accumulation_steps(self):
        """Test gradient_accumulation_steps configuration"""
        config = {
            'gradient_accumulation_steps': 4
        }
        result = TrainingConfig.apply('training_args', config)
        assert result['gradient_accumulation_steps'] == 4

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_resume_training_config(self):
        """Test resume training related configuration"""
        config = {
            'resume_training': True,
            'data_skip_steps': 100,
            'ignore_data_skip': False
        }
        result = TrainingConfig.apply('training_args', config)
        assert result['resume_training'] is True
        assert result['data_skip_steps'] == 100
        assert result['ignore_data_skip'] is False


# ============================================================================
# Recompute and Swap Config Tests
# ============================================================================
class TestRecomputeConfig:
    """Test RecomputeConfig class"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_default_values(self):
        """Test default values of RecomputeConfig"""
        default_config = RecomputeConfig.default_value()
        assert default_config['recompute'] is False
        assert default_config['parallel_optimizer_comm_recompute'] is True
        assert default_config['mp_comm_recompute'] is True
        assert default_config['recompute_slice_activation'] is False
        assert default_config['select_recompute'] is False
        assert default_config['select_recompute_exclude'] is False
        assert default_config['select_comm_recompute'] is False
        assert default_config['select_comm_recompute_exclude'] is False

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_with_bool_recompute(self):
        """Test apply method with boolean recompute value"""
        config = {
            'recompute': True,
            'mp_comm_recompute': True
        }
        result = RecomputeConfig.apply('recompute_config', config)
        assert result['recompute'] is True
        assert result['mp_comm_recompute'] is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_with_list_recompute(self):
        """Test apply method with list recompute value"""
        config = {
            'recompute': ['layer1', 'layer2'],
            'select_recompute': ['layer3']
        }
        result = RecomputeConfig.apply('recompute_config', config)
        assert result['recompute'] == ['layer1', 'layer2']
        assert result['select_recompute'] == ['layer3']

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_valid_bool(self):
        """Test validate_config with valid boolean values"""
        config = {
            'recompute': True,
            'select_recompute': False,
            'parallel_optimizer_comm_recompute': True,
            'mp_comm_recompute': True
        }
        # Should not raise exception
        RecomputeConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_valid_list(self):
        """Test validate_config with valid list values"""
        config = {
            'recompute': ['layer1', 'layer2'],
            'select_recompute': ['layer3', 'layer4']
        }
        # Should not raise exception
        RecomputeConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_recompute_slice(self):
        """Test validate_config with invalid recompute_slice_activation"""
        config = {
            'recompute_slice_activation': True,  # Only False is supported
        }
        with pytest.raises(ValueError):
            RecomputeConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_type(self):
        """Test validate_config with invalid parameter type"""
        config = {
            'recompute': 'true',  # Should be bool or list
        }
        with pytest.raises(ValueError):
            RecomputeConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_select_recompute_exclude(self):
        """Test select_recompute_exclude configuration"""
        config = {
            'select_recompute': True,
            'select_recompute_exclude': ['layer1', 'layer2']
        }
        result = RecomputeConfig.apply('recompute_config', config)
        assert result['select_recompute'] is True
        assert result['select_recompute_exclude'] == ['layer1', 'layer2']


class TestSwapConfig:
    """Test SwapConfig class"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_default_values(self):
        """Test default values of SwapConfig"""
        default_config = SwapConfig.default_value()
        assert default_config['swap'] is False
        assert default_config['layer_swap'] is None
        assert default_config['op_swap'] is None
        assert default_config['default_prefetch'] == 1

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_with_valid_config(self):
        """Test apply method with valid configuration"""
        config = {
            'swap': True,
            'default_prefetch': 2
        }
        result = SwapConfig.apply('swap_config', config)
        assert result['swap'] is True
        assert result['default_prefetch'] == 2

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_with_layer_swap(self):
        """Test apply method with layer_swap configuration"""
        config = {
            'swap': True,
            'layer_swap': ['layer1', 'layer2']
        }
        result = SwapConfig.apply('swap_config', config)
        assert result['swap'] is True
        assert result['layer_swap'] == ['layer1', 'layer2']

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_with_op_swap(self):
        """Test apply method with op_swap configuration"""
        config = {
            'swap': True,
            'op_swap': ['matmul', 'conv']
        }
        result = SwapConfig.apply('swap_config', config)
        assert result['swap'] is True
        assert result['op_swap'] == ['matmul', 'conv']

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_valid(self):
        """Test validate_config with valid configuration"""
        config = {
            'swap': True,
            'default_prefetch': 1
        }
        # Should not raise exception
        SwapConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_with_none_values(self):
        """Test validate_config with None values"""
        config = {
            'swap': False,
            'layer_swap': None,
            'op_swap': None
        }
        # Should not raise exception
        SwapConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_type(self):
        """Test validate_config with invalid parameter type"""
        config = {
            'swap': 'true',  # Should be boolean
        }
        with pytest.raises(ValueError):
            SwapConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_prefetch_type(self):
        """Test validate_config with invalid default_prefetch type"""
        config = {
            'default_prefetch': '1',  # Should be int
        }
        with pytest.raises(ValueError):
            SwapConfig.validate_config(config)


# ============================================================================
# Parallel Config Tests
# ============================================================================
class TestTrainingParallelConfig:
    """Test TrainingParallelConfig class"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_default_values(self):
        """Test default values of TrainingParallelConfig"""
        default_config = TrainingParallelConfig.default_value()
        assert default_config['data_parallel_size'] is None
        assert default_config['tensor_model_parallel_size'] == 1
        assert default_config['pipeline_model_parallel_size'] == 1
        assert default_config['context_parallel_size'] == 1
        assert default_config['cp_comm_type'] == 'all_gather'
        assert default_config['expert_model_parallel_size'] == 1
        assert default_config['sequence_parallel'] is False
        assert default_config['gradient_aggregation_group'] == 1
        assert default_config['micro_batch_interleave_num'] == 1

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_with_valid_config(self):
        """Test apply method with valid configuration"""
        config = {
            'tensor_model_parallel_size': 2,
            'pipeline_model_parallel_size': 2,
            'sequence_parallel': True
        }
        result = TrainingParallelConfig.apply('training_parallel_config', config)
        assert result['tensor_model_parallel_size'] == 2
        assert result['pipeline_model_parallel_size'] == 2
        assert result['sequence_parallel'] is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_pipeline_parallel_config(self):
        """Test pipeline parallel configuration"""
        config = {
            'pipeline_parallel_config': {
                'pipeline_interleave': True,
                'pipeline_scheduler': '1f1b',
                'virtual_pipeline_model_parallel_size': 2,
                'pipeline_stage_offset': 0
            }
        }
        result = TrainingParallelConfig.apply('training_parallel_config', config)
        pp_config = result['pipeline_parallel_config']
        assert pp_config['pipeline_interleave'] is True
        assert pp_config['pipeline_scheduler'] == '1f1b'
        assert pp_config['virtual_pipeline_model_parallel_size'] == 2

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_optimizer_parallel_config(self):
        """Test optimizer parallel configuration"""
        config = {
            'optimizer_parallel_config': {
                'enable_parallel_optimizer': True,
                'optimizer_level': 'level1',
                'optimizer_weight_shard_size': 2
            }
        }
        result = TrainingParallelConfig.apply('training_parallel_config', config)
        opt_config = result['optimizer_parallel_config']
        assert opt_config['enable_parallel_optimizer'] is True
        assert opt_config['optimizer_level'] == 'level1'

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_valid(self):
        """Test validate_config with valid configuration"""
        config = {
            'tensor_model_parallel_size': 4,
            'pipeline_model_parallel_size': 2,
            'context_parallel_size': 1,
            'sequence_parallel': False,
            'gradient_aggregation_group': 1
        }
        # Should not raise exception
        TrainingParallelConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_pipeline_config(self):
        """Test validate_config with invalid pipeline configuration"""
        config = {
            'pipeline_parallel_config': {
                'pipeline_interleave': 'invalid',  # Should be boolean
                'pipeline_scheduler': '1f1b',
                'virtual_pipeline_model_parallel_size': 1
            }
        }
        with pytest.raises(ValueError):
            TrainingParallelConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_gradient_aggregation(self):
        """Test validate_config with invalid gradient_aggregation_group"""
        config = {
            'gradient_aggregation_group': 0,  # Should be >= 1
        }
        with pytest.raises(ValueError):
            TrainingParallelConfig.validate_config(config)


class TestInferParallelConfig:
    """Test InferParallelConfig class"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_default_values(self):
        """Test default values of InferParallelConfig"""
        default_config = InferParallelConfig.default_value()
        assert default_config['data_parallel_size'] is None
        assert default_config['tensor_model_parallel_size'] == 1
        assert default_config['pipeline_model_parallel_size'] == 1
        assert default_config['expert_model_parallel_size'] == 1

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_with_valid_config(self):
        """Test apply method with valid configuration"""
        config = {
            'tensor_model_parallel_size': 4,
            'expert_model_parallel_size': 2
        }
        result = InferParallelConfig.apply('infer_parallel_config', config)
        assert result['tensor_model_parallel_size'] == 4
        assert result['expert_model_parallel_size'] == 2

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_valid(self):
        """Test validate_config with valid configuration"""
        config = {
            'tensor_model_parallel_size': 2,
            'pipeline_model_parallel_size': 1,
            'expert_model_parallel_size': 1
        }
        # Should not raise exception
        InferParallelConfig.validate_config(config)


class TestParallelContextConfig:
    """Test ParallelContextConfig class"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_default_values(self):
        """Test default values of ParallelContextConfig"""
        default_config = ParallelContextConfig.default_value()
        assert default_config['parallel_mode'] == 1
        assert default_config['full_batch'] is False
        assert default_config['gradients_mean'] is False
        assert default_config['enable_alltoall'] is True
        assert default_config['search_mode'] == 'sharding_propagation'

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_with_valid_config(self):
        """Test apply method with valid configuration"""
        config = {
            'parallel_mode': 1,
            'enable_alltoall': True,
            'search_mode': 'sharding_propagation'
        }
        result = ParallelContextConfig.apply('parallel', config)
        assert result['parallel_mode'] == 1
        assert result['enable_alltoall'] is True
        assert result['search_mode'] == 'sharding_propagation'

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_valid(self):
        """Test validate_config with valid configuration"""
        config = {
            'parallel_mode': 1,
            'full_batch': False,
            'enable_alltoall': True,
            'search_mode': 'sharding_propagation'
        }
        # Should not raise exception
        ParallelContextConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_parallel_mode(self):
        """Test validate_config with invalid parallel_mode"""
        config = {
            'parallel_mode': 3,  # Should be 0, 1, or 2
        }
        with pytest.raises(ValueError):
            ParallelContextConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_search_mode(self):
        """Test validate_config with invalid search_mode"""
        config = {
            'search_mode': 'invalid_mode',
        }
        with pytest.raises(ValueError):
            ParallelContextConfig.validate_config(config)


class TestInferParallelContextConfig:
    """Test InferParallelContextConfig class"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_default_values(self):
        """Test default values of InferParallelContextConfig"""
        default_config = InferParallelContextConfig.default_value()
        assert default_config['parallel_mode'] == 'MANUAL_PARALLEL'
        assert default_config['enable_alltoall'] is False

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_with_valid_config(self):
        """Test apply method with valid configuration"""
        config = {
            'parallel_mode': 'MANUAL_PARALLEL',
            'enable_alltoall': False
        }
        result = InferParallelContextConfig.apply('infer_parallel_context', config)
        assert result['parallel_mode'] == 'MANUAL_PARALLEL'
        assert result['enable_alltoall'] is False

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_valid(self):
        """Test validate_config with valid configuration"""
        config = {
            'parallel_mode': 'MANUAL_PARALLEL',
            'enable_alltoall': False
        }
        # Should not raise exception
        InferParallelContextConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_parallel_mode(self):
        """Test validate_config with invalid parallel_mode"""
        config = {
            'parallel_mode': 'AUTO_PARALLEL',  # Only MANUAL_PARALLEL is supported
        }
        with pytest.raises(ValueError):
            InferParallelContextConfig.validate_config(config)


# ============================================================================
# Monitor Config Tests
# ============================================================================
class TestMonitorConfig:
    """Test MonitorConfig class"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_default_values(self):
        """Test default values of MonitorConfig"""
        default_config = MonitorConfig.default_value()
        assert default_config['dump_path'] == './dump'
        assert default_config['local_loss'] is None
        assert default_config['device_local_norm'] is None
        assert default_config['device_local_loss'] is None
        assert default_config['local_norm'] is None
        assert default_config['optimizer_params_state'] is None
        assert default_config['net_weight_params_state'] is None
        assert default_config['target_parameters'] == ['.*']
        assert default_config['target_parameters_invert'] is False
        assert default_config['embedding_local_norm'] is False
        assert default_config['step_interval'] == 1

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_with_valid_config(self):
        """Test apply method with valid configuration"""
        config = {
            'dump_path': './custom_dump',
            'local_loss': ['log', 'tensorboard'],
            'step_interval': 10
        }
        result = MonitorConfig.apply('monitor_config', config)
        assert result['dump_path'] == './custom_dump'
        assert result['local_loss'] == ['log', 'tensorboard']
        assert result['step_interval'] == 10

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_with_target_parameters(self):
        """Test apply method with target_parameters configuration"""
        config = {
            'target_parameters': ['.*embedding.*', '.*layer1.*'],
            'target_parameters_invert': True
        }
        result = MonitorConfig.apply('monitor_config', config)
        assert len(result['target_parameters']) == 2
        assert result['target_parameters_invert'] is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_valid(self):
        """Test validate_config with valid configuration"""
        config = {
            'dump_path': './dump',
            'step_interval': 1,
            'local_loss': ['log'],
            'embedding_local_norm': False
        }
        # Should not raise exception
        MonitorConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_type(self):
        """Test validate_config with invalid parameter type"""
        config = {
            'step_interval': '1',  # Should be int
        }
        with pytest.raises(ValueError):
            MonitorConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_target_params_type(self):
        """Test validate_config with invalid target_parameters type"""
        config = {
            'target_parameters': '.*',  # Should be list
        }
        with pytest.raises(ValueError):
            MonitorConfig.validate_config(config)


class TestTensorBoardConfig:
    """Test TensorBoardConfig class"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_default_values(self):
        """Test default values of TensorBoardConfig"""
        default_config = TensorBoardConfig.default_value()
        assert default_config['tensorboard_on'] is False
        assert default_config['tensorboard_dir'] == './tensorboard'
        assert default_config['tensorboard_queue_size'] == 10
        assert default_config['log_loss_scale_to_tensorboard'] is False
        assert default_config['log_timers_to_tensorboard'] is False
        assert default_config['log_expert_load_to_tensorboard'] is False

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_with_valid_config(self):
        """Test apply method with valid configuration"""
        config = {
            'tensorboard_on': True,
            'tensorboard_dir': './custom_tensorboard',
            'tensorboard_queue_size': 20
        }
        result = TensorBoardConfig.apply('tensorboard', config)
        assert result['tensorboard_on'] is True
        assert result['tensorboard_dir'] == './custom_tensorboard'
        assert result['tensorboard_queue_size'] == 20

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_with_logging_options(self):
        """Test apply method with logging options"""
        config = {
            'tensorboard_on': True,
            'log_loss_scale_to_tensorboard': True,
            'log_timers_to_tensorboard': True,
            'log_expert_load_to_tensorboard': True
        }
        result = TensorBoardConfig.apply('tensorboard', config)
        assert result['log_loss_scale_to_tensorboard'] is True
        assert result['log_timers_to_tensorboard'] is True
        assert result['log_expert_load_to_tensorboard'] is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_valid(self):
        """Test validate_config with valid configuration"""
        config = {
            'tensorboard_on': True,
            'tensorboard_dir': './tensorboard',
            'tensorboard_queue_size': 10
        }
        # Should not raise exception
        TensorBoardConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_type(self):
        """Test validate_config with invalid parameter type"""
        config = {
            'tensorboard_on': 'true',  # Should be boolean
        }
        with pytest.raises(ValueError):
            TensorBoardConfig.validate_config(config)


class TestProfileConfig:
    """Test ProfileConfig class"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_default_values(self):
        """Test default values of ProfileConfig"""
        default_config = ProfileConfig.default_value()
        assert default_config['profile_on'] is False
        assert default_config['profile_output'] is None
        assert default_config['profiler_level'] == 1
        assert default_config['profile_start_step'] == 1
        assert default_config['profile_stop_step'] == 10
        assert default_config['init_start_profile'] is False
        assert default_config['profile_rank_ids'] is None
        assert default_config['profile_pipeline'] is False
        assert default_config['profile_communication'] is False
        assert default_config['profile_memory'] is True
        assert default_config['with_stack'] is False
        assert default_config['data_simplification'] is False
        assert default_config['mstx'] is False
        assert default_config['use_llm_token_profile'] is False
        assert default_config['llm_token_profile_config'] is None

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_with_valid_config(self):
        """Test apply method with valid configuration"""
        config = {
            'profile_on': True,
            'profile_output': './profile_output',
            'profile_start_step': 5,
            'profile_stop_step': 15
        }
        result = ProfileConfig.apply('profile', config)
        assert result['profile_on'] is True
        assert result['profile_output'] == './profile_output'
        assert result['profile_start_step'] == 5
        assert result['profile_stop_step'] == 15

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_with_profile_options(self):
        """Test apply method with profile options"""
        config = {
            'profile_on': True,
            'profile_pipeline': True,
            'profile_communication': True,
            'profile_memory': True,
            'with_stack': True
        }
        result = ProfileConfig.apply('profile', config)
        assert result['profile_pipeline'] is True
        assert result['profile_communication'] is True
        assert result['profile_memory'] is True
        assert result['with_stack'] is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_with_rank_ids(self):
        """Test apply method with profile_rank_ids"""
        config = {
            'profile_on': True,
            'profile_rank_ids': [0, 1, 2, 3]
        }
        result = ProfileConfig.apply('profile', config)
        assert result['profile_rank_ids'] == [0, 1, 2, 3]

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_valid(self):
        """Test validate_config with valid configuration"""
        config = {
            'profile_on': True,
            'profiler_level': 1,
            'profile_start_step': 1,
            'profile_stop_step': 10
        }
        # Should not raise exception
        ProfileConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_profiler_level(self):
        """Test validate_config with invalid profiler_level"""
        config = {
            'profiler_level': 2,  # Only 1 is supported
        }
        with pytest.raises(ValueError):
            ProfileConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_type(self):
        """Test validate_config with invalid parameter type"""
        config = {
            'profile_on': 'true',  # Should be boolean
        }
        with pytest.raises(ValueError):
            ProfileConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_llm_token_profile_config(self):
        """Test llm_token_profile_config configuration"""
        config = {
            'use_llm_token_profile': True,
            'llm_token_profile_config': {
                'profile_range': [0, 100]
            }
        }
        result = ProfileConfig.apply('profile', config)
        assert result['use_llm_token_profile'] is True
        assert result['llm_token_profile_config'] is not None


# ============================================================================
# Optimizer and Learning Rate Config Tests
# ============================================================================
class TestOptimizerConfig:
    """Test OptimizerConfig class"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_default_values(self):
        """Test default values of OptimizerConfig"""
        default_config = OptimizerConfig.default_value()
        assert default_config['type'] == 'AdamW'
        assert default_config['betas'] == [0.9, 0.999]
        assert default_config['learning_rate'] == 5.e-5
        assert default_config['eps'] == 1.e-8
        assert default_config['weight_decay'] == 0.0

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_with_valid_config(self):
        """Test apply method with valid configuration"""
        config = {
            'type': 'AdamW',
            'betas': [0.9, 0.95],
            'learning_rate': 1.e-4,
            'weight_decay': 0.01
        }
        result = OptimizerConfig.apply('optimizer', config)
        assert result['type'] == 'AdamW'
        assert result['betas'] == [0.9, 0.95]
        assert result['learning_rate'] == 1.e-4
        assert result['weight_decay'] == 0.01

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_missing_required_type(self):
        """Test apply method missing required type key"""
        config = {
            'learning_rate': 1.e-4
        }
        # Missing 'type' which is required
        with pytest.raises(KeyError):
            OptimizerConfig.apply('optimizer', config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_with_additional_params(self):
        """Test apply method with additional parameters"""
        config = {
            'type': 'AdamW',
            'use_fused': True,
            'amsgrad': False,
            'swap': False
        }
        result = OptimizerConfig.apply('optimizer', config)
        assert result['type'] == 'AdamW'
        assert result['use_fused'] is True
        assert result['amsgrad'] is False

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_pma_adamw_config(self):
        """Test PmaAdamW optimizer configuration"""
        config = {
            'type': 'PmaAdamW',
            'fused_num': 10,
            'interleave_step': 1000,
            'fused_algo': 'ema',
            'ema_alpha': 0.2
        }
        result = OptimizerConfig.apply('optimizer', config)
        assert result['type'] == 'PmaAdamW'
        assert result['fused_num'] == 10
        assert result['interleave_step'] == 1000
        assert result['fused_algo'] == 'ema'
        assert result['ema_alpha'] == 0.2


class TestLrScheduleConfig:
    """Test LrScheduleConfig class"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_default_values(self):
        """Test default values of LrScheduleConfig"""
        default_config = LrScheduleConfig.default_value()
        assert default_config['type'] == 'CosineWithWarmUpLR'
        assert default_config['learning_rate'] == 5.e-5
        assert default_config['lr_end'] == 0.
        assert default_config['warmup_ratio'] == 0.
        assert default_config['total_steps'] == -1

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_constant_warmup_lr(self):
        """Test apply method with ConstantWarmUpLR"""
        config = {
            'type': 'ConstantWarmUpLR',
            'learning_rate': 1.e-6,
            'warmup_ratio': 0.1
        }
        result = LrScheduleConfig.apply('lr_schedule', config)
        assert result['type'] == 'ConstantWarmUpLR'
        assert result['learning_rate'] == 1.e-6
        assert result['warmup_ratio'] == 0.1

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_linear_warmup_lr(self):
        """Test apply method with LinearWithWarmUpLR"""
        config = {
            'type': 'LinearWithWarmUpLR',
            'learning_rate': 1.e-6,
            'warmup_lr_init': 0.,
            'warmup_ratio': 0.05
        }
        result = LrScheduleConfig.apply('lr_schedule', config)
        assert result['type'] == 'LinearWithWarmUpLR'
        assert result['warmup_lr_init'] == 0.

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_cosine_warmup_lr(self):
        """Test apply method with CosineWithWarmUpLR"""
        config = {
            'type': 'CosineWithWarmUpLR',
            'learning_rate': 1.e-4,
            'lr_end': 1.e-7,
            'warmup_ratio': 0.1,
            'total_steps': 10000
        }
        result = LrScheduleConfig.apply('lr_schedule', config)
        assert result['type'] == 'CosineWithWarmUpLR'
        assert result['lr_end'] == 1.e-7
        assert result['total_steps'] == 10000

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_missing_required_type(self):
        """Test apply method missing required type key"""
        config = {
            'learning_rate': 1.e-4
        }
        # Missing 'type' which is required
        with pytest.raises(KeyError):
            LrScheduleConfig.apply('lr_schedule', config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_polynomial_lr(self):
        """Test apply method with PolynomialWithWarmUpLR"""
        config = {
            'type': 'PolynomialWithWarmUpLR',
            'learning_rate': 1.e-4,
            'power': 1.0,
            'lr_end': 0.
        }
        result = LrScheduleConfig.apply('lr_schedule', config)
        assert result['type'] == 'PolynomialWithWarmUpLR'
        assert result['power'] == 1.0

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_cosine_annealing_warm_restarts(self):
        """Test apply method with CosineAnnealingWarmRestarts"""
        config = {
            'type': 'CosineAnnealingWarmRestarts',
            'base_lr': 1.e-6,
            't_0': 10,
            't_mult': 1.
        }
        result = LrScheduleConfig.apply('lr_schedule', config)
        assert result['type'] == 'CosineAnnealingWarmRestarts'
        assert result['base_lr'] == 1.e-6
        assert result['t_0'] == 10


# ============================================================================
# General Config Tests
# ============================================================================
class TestTrainingGeneralConfig:
    """Test TrainingGeneralConfig class"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_default_values(self):
        """Test default values of TrainingGeneralConfig"""
        default_config = TrainingGeneralConfig.default_value()
        assert default_config['run_mode'] is None
        assert default_config['output_dir'] == "./output"
        assert default_config['use_parallel'] is False
        assert default_config['use_legacy'] is False
        assert default_config['pretrained_model_dir'] == ""
        assert default_config['train_precision_sync'] is False

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_with_valid_config(self):
        """Test apply method with valid configuration"""
        config = {
            'run_mode': 'train',
            'output_dir': './custom_output',
            'use_parallel': True
        }
        result = TrainingGeneralConfig.apply('training_general_config', config)
        assert result['run_mode'] == 'train'
        assert result['output_dir'] == './custom_output'
        assert result['use_parallel'] is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_with_none_config(self):
        """Test apply method with None configuration"""
        result = TrainingGeneralConfig.apply('training_general_config', None)
        assert result['run_mode'] is None
        assert result['output_dir'] == "./output"

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_with_empty_config(self):
        """Test apply method with empty configuration"""
        result = TrainingGeneralConfig.apply('training_general_config', {})
        assert result['run_mode'] is None

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_valid_run_mode(self):
        """Test validate_config with valid run_mode"""
        config = {
            'run_mode': 'train',
            'output_dir': './output',
            'use_parallel': True,
            'use_legacy': False
        }
        # Should not raise exception
        TrainingGeneralConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_run_mode(self):
        """Test validate_config with invalid run_mode"""
        config = {
            'run_mode': 'invalid_mode',
            'output_dir': './output'
        }
        with pytest.raises(ValueError):
            TrainingGeneralConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_use_legacy(self):
        """Test validate_config with invalid use_legacy value"""
        config = {
            'run_mode': 'train',
            'use_legacy': True
        }
        with pytest.raises(ValueError):
            TrainingGeneralConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_type(self):
        """Test validate_config with invalid parameter type"""
        config = {
            'use_parallel': 'true',  # Should be boolean
        }
        with pytest.raises(ValueError):
            TrainingGeneralConfig.validate_config(config)


class TestInferGeneralConfig:
    """Test InferGeneralConfig class"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_default_values(self):
        """Test default values of InferGeneralConfig"""
        default_config = InferGeneralConfig.default_value()
        assert default_config['run_mode'] is None
        assert default_config['output_dir'] == "./output"
        assert default_config['use_parallel'] is False
        assert default_config['predict_batch_size'] == 1
        assert default_config['load_checkpoint'] == ''
        assert default_config['infer_seed'] == 1234

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_with_valid_config(self):
        """Test apply method with valid configuration"""
        config = {
            'run_mode': 'predict',
            'predict_batch_size': 4,
            'infer_seed': 42
        }
        result = InferGeneralConfig.apply('infer_general_config', config)
        assert result['run_mode'] == 'predict'
        assert result['predict_batch_size'] == 4
        assert result['infer_seed'] == 42

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_valid(self):
        """Test validate_config with valid configuration"""
        config = {
            'run_mode': 'predict',
            'predict_batch_size': 2,
            'load_ckpt_format': 'safetensors',
            'infer_seed': 100
        }
        # Should not raise exception
        InferGeneralConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_run_mode(self):
        """Test validate_config with invalid run_mode"""
        config = {
            'run_mode': 'train',  # Should be 'predict'
        }
        with pytest.raises(ValueError):
            InferGeneralConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_batch_size(self):
        """Test validate_config with invalid predict_batch_size"""
        config = {
            'predict_batch_size': 0,  # Should be >= 1
        }
        with pytest.raises(ValueError):
            InferGeneralConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_format(self):
        """Test validate_config with invalid checkpoint format"""
        config = {
            'load_ckpt_format': 'invalid_format',
        }
        with pytest.raises(ValueError):
            InferGeneralConfig.validate_config(config)


# ============================================================================
# Context Config Tests
# ============================================================================
class TestContextConfig:
    """Test ContextConfig class"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_default_values(self):
        """Test default values of ContextConfig"""
        default_config = ContextConfig.default_value()
        assert default_config['mode'] == 0
        assert default_config['device_target'] == 'Ascend'
        assert default_config['device_id'] == 0
        assert default_config['max_device_memory'] == '58GB'
        assert default_config['mempool_block_size'] == '1GB'
        assert default_config['memory_optimize_level'] == 'O0'
        assert default_config['jit_config']['jit_level'] == 'O0'
        assert default_config['ascend_config']['precision_mode'] == 'must_keep_origin_dtype'
        assert default_config['max_call_depth'] == 10000

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_with_valid_config(self):
        """Test apply method with valid configuration"""
        config = {
            'mode': 0,
            'device_id': 1,
            'max_device_memory': '59GB'
        }
        result = ContextConfig.apply('context', config)
        assert result['mode'] == 0
        assert result['device_id'] == 1
        assert result['max_device_memory'] == '59GB'

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_with_jit_config(self):
        """Test apply method with jit_config"""
        config = {
            'jit_config': {
                'jit_level': 'O1'
            }
        }
        result = ContextConfig.apply('context', config)
        assert result['jit_config']['jit_level'] == 'O1'

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_with_ascend_config(self):
        """Test apply method with ascend_config"""
        config = {
            'ascend_config': {
                'precision_mode': 'must_keep_origin_dtype',
                'parallel_speed_up_json_path': './parallel_speed_up.json'
            }
        }
        result = ContextConfig.apply('context', config)
        assert result['ascend_config']['precision_mode'] == 'must_keep_origin_dtype'
        assert result['ascend_config']['parallel_speed_up_json_path'] == './parallel_speed_up.json'

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_valid(self):
        """Test validate_config with valid configuration"""
        config = {
            'mode': 0,
            'device_target': 'Ascend',
            'max_device_memory': '58GB',
            'memory_optimize_level': 'O0',
            'jit_config': {'jit_level': 'O0'},
            'ascend_config': {'precision_mode': 'must_keep_origin_dtype'}
        }
        # Should not raise exception
        ContextConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_mode(self):
        """Test validate_config with invalid mode"""
        config = {
            'mode': 2,  # Should be 0 or 1
        }
        with pytest.raises(ValueError):
            ContextConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_device_target(self):
        """Test validate_config with invalid device_target"""
        config = {
            'device_target': 'GPU',  # Only 'Ascend' is supported
        }
        with pytest.raises(ValueError):
            ContextConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_memory_optimize_level(self):
        """Test validate_config with invalid memory_optimize_level"""
        config = {
            'memory_optimize_level': 'O2',  # Only 'O0' and 'O1' are supported
        }
        with pytest.raises(ValueError):
            ContextConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_jit_level(self):
        """Test validate_config with invalid jit_level"""
        config = {
            'jit_config': {'jit_level': 'O2'},  # Only 'O0' and 'O1' are supported
        }
        with pytest.raises(ValueError):
            ContextConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_precision_mode(self):
        """Test validate_config with invalid precision_mode"""
        config = {
            'ascend_config': {'precision_mode': 'invalid_mode'},
        }
        with pytest.raises(ValueError):
            ContextConfig.validate_config(config)


class TestInferContextConfig:
    """Test InferContextConfig class"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_default_values(self):
        """Test default values of InferContextConfig"""
        default_config = InferContextConfig.default_value()
        assert default_config['mode'] == 0
        assert default_config['device_target'] == 'Ascend'
        assert default_config['device_id'] == 0
        assert default_config['max_device_memory'] == '59GB'
        assert default_config['jit_config']['jit_level'] == 'O0'
        assert default_config['ascend_config']['precision_mode'] == 'must_keep_origin_dtype'

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_with_valid_config(self):
        """Test apply method with valid configuration"""
        config = {
            'mode': 0,
            'max_device_memory': '60GB'
        }
        result = InferContextConfig.apply('infer_context', config)
        assert result['mode'] == 0
        assert result['max_device_memory'] == '60GB'

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_valid(self):
        """Test validate_config with valid configuration"""
        config = {
            'mode': 0,
            'device_target': 'Ascend',
            'max_device_memory': '59GB',
            'jit_config': {'jit_level': 'O0'},
            'ascend_config': {'precision_mode': 'must_keep_origin_dtype'}
        }
        # Should not raise exception
        InferContextConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_jit_level(self):
        """Test validate_config with invalid jit_level for inference"""
        config = {
            'jit_config': {'jit_level': 'O1'},  # Only 'O0' is supported for inference
        }
        with pytest.raises(ValueError):
            InferContextConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_mode(self):
        """Test validate_config with invalid mode"""
        config = {
            'mode': 2,  # Should be 0 or 1
        }
        with pytest.raises(ValueError):
            InferContextConfig.validate_config(config)


# ============================================================================
# Dataset Config Tests
# ============================================================================
class TestTrainingDatasetConfig:
    """Test TrainingDatasetConfig class"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_default_values(self):
        """Test default values of TrainingDatasetConfig"""
        default_config = TrainingDatasetConfig.default_value()
        assert default_config['num_parallel_workers'] == 8
        assert default_config['python_multiprocessing'] is False
        assert default_config['drop_remainder'] is True
        assert default_config['numa_enable'] is False
        assert default_config['prefetch_size'] == 1
        assert default_config['input_columns'] is None
        assert default_config['construct_args_key'] is None

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_with_valid_config(self):
        """Test apply method with valid configuration"""
        config = {
            'data_loader': {
                'type': 'HFDataLoader',
                'load_func': 'load_dataset',
                'path': 'test_dataset',
                'data_files': 'test.json',
                'handler': []
            },
            'num_parallel_workers': 16,
            'drop_remainder': True
        }
        result = TrainingDatasetConfig.apply('train_dataset', config)
        assert result['num_parallel_workers'] == 16
        assert result['drop_remainder'] is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_missing_required_data_loader(self):
        """Test apply method missing required data_loader"""
        config = {
            'num_parallel_workers': 8
        }
        # Missing 'data_loader' which is required
        with pytest.raises(KeyError):
            TrainingDatasetConfig.apply('train_dataset', config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_valid(self):
        """Test validate_config with valid configuration"""
        config = {
            'num_parallel_workers': 8,
            'python_multiprocessing': False,
            'drop_remainder': True,
            'prefetch_size': 1
        }
        # Should not raise exception
        TrainingDatasetConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_type(self):
        """Test validate_config with invalid parameter type"""
        config = {
            'num_parallel_workers': '8',  # Should be int
        }
        with pytest.raises(ValueError):
            TrainingDatasetConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_with_input_columns(self):
        """Test validate_config with input_columns"""
        config = {
            'input_columns': ['input_ids', 'labels', 'attention_mask']
        }
        # Should not raise exception
        TrainingDatasetConfig.validate_config(config)


class TestMegatronDataLoaderConfig:
    """Test MegatronDataLoaderConfig class"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_default_values(self):
        """Test default values of MegatronDataLoaderConfig"""
        default_config = MegatronDataLoaderConfig.default_value()
        assert default_config['type'] == 'BlendedMegatronDatasetDataLoader'
        assert default_config['datasets_type'] == 'GPTDataset'
        assert 'seed' in default_config['config']
        assert default_config['config']['seed'] == 1234

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_with_valid_config(self):
        """Test apply method with valid configuration"""
        config = {
            'type': 'BlendedMegatronDatasetDataLoader',
            'sizes': [1000, 0, 0],
            'config': {
                'seed': 42,
                'seq_length': 4096,
                'eod_mask_loss': True
            }
        }
        result = MegatronDataLoaderConfig.apply('megatron_dataloader', config)
        assert result['type'] == 'BlendedMegatronDatasetDataLoader'
        assert result['sizes'] == [1000, 0, 0]
        assert result['config']['seed'] == 42

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_missing_required_keys(self):
        """Test apply method missing required keys"""
        config = {
            'type': 'BlendedMegatronDatasetDataLoader',
            'sizes': [1000, 0, 0]
        }
        # Missing 'config' which is required
        with pytest.raises(KeyError):
            MegatronDataLoaderConfig.apply('megatron_dataloader', config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_valid(self):
        """Test validate_config with valid configuration"""
        config = {
            'type': 'BlendedMegatronDatasetDataLoader',
            'datasets_type': 'GPTDataset'
        }
        # Should not raise exception
        MegatronDataLoaderConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_type(self):
        """Test validate_config with invalid type"""
        config = {
            'type': 'InvalidDataLoader',
        }
        with pytest.raises(ValueError):
            MegatronDataLoaderConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_datasets_type(self):
        """Test validate_config with invalid datasets_type"""
        config = {
            'datasets_type': 'InvalidDataset',
        }
        with pytest.raises(ValueError):
            MegatronDataLoaderConfig.validate_config(config)


class TestHFDataLoader:
    """Test HFDataLoader class"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_default_values(self):
        """Test default values of HFDataLoader"""
        default_config = HFDataLoader.default_value()
        assert default_config['type'] == 'HFDataLoader'
        assert default_config['shuffle'] is False
        assert default_config['create_attention_mask'] is True
        assert default_config['create_compressed_eod_mask'] is False
        assert default_config['compressed_eod_mask_length'] == 128
        assert default_config['use_broadcast_data'] is True
        assert default_config['split'] == 'train'

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_with_valid_config(self):
        """Test apply method with valid configuration"""
        config = {
            'type': 'HFDataLoader',
            'load_func': 'load_dataset',
            'path': 'test_dataset',
            'data_files': 'test.json',
            'handler': [],
            'shuffle': True
        }
        result = HFDataLoader.apply('hf_dataloader', config)
        assert result['type'] == 'HFDataLoader'
        assert result['shuffle'] is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_missing_required_keys(self):
        """Test apply method missing required keys"""
        config = {
            'type': 'HFDataLoader',
            'load_func': 'load_dataset'
        }
        # Missing 'path', 'data_files', 'handler' which are required
        with pytest.raises(KeyError):
            HFDataLoader.apply('hf_dataloader', config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_valid(self):
        """Test validate_config with valid configuration"""
        config = {
            'type': 'HFDataLoader',
            'shuffle': True,
            'create_attention_mask': True,
            'split': 'train'
        }
        # Should not raise exception
        HFDataLoader.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_type(self):
        """Test validate_config with invalid type"""
        config = {
            'type': 'InvalidDataLoader',
        }
        with pytest.raises(ValueError):
            HFDataLoader.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_split(self):
        """Test validate_config with invalid split"""
        config = {
            'split': 'test',  # Only 'train' is supported
        }
        with pytest.raises(ValueError):
            HFDataLoader.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_param_type(self):
        """Test validate_config with invalid parameter type"""
        config = {
            'shuffle': 'true',  # Should be boolean
        }
        with pytest.raises(ValueError):
            HFDataLoader.validate_config(config)


# ============================================================================
# Config Template Tests
# ============================================================================
class TestConfigTemplate:
    """Test ConfigTemplate class"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_train_configs_list(self):
        """Test train_configs list contains expected config names"""
        expected_configs = [
            'training_general_config',
            'distribute_parallel_config',
            'recompute_config',
            'swap_config',
            'training_args',
            'parallel',
            'context',
            'train_dataset',
            'trainer',
            'model_config',
            'optimizer',
            'lr_schedule',
            'callbacks',
            'monitor_config',
            'profile',
            'tensorboard',
            'checkpoint_config'
        ]
        assert ConfigTemplate.train_configs == expected_configs

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_predict_configs_list(self):
        """Test predict_configs list contains expected config names"""
        expected_configs = [
            'infer_general_config',
            'distribute_parallel_config',
            'parallel',
            'context',
            'trainer',
            'model_config'
        ]
        assert ConfigTemplate.predict_configs == expected_configs

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_run_modes_list(self):
        """Test _run_modes list contains expected modes"""
        expected_modes = ['train', 'predict', 'finetune']
        assert ConfigTemplate._run_modes == expected_modes  # pylint: disable=protected-access

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_template_train_mode(self):
        """Test apply_template with train mode"""
        config = {
            'run_mode': 'train',
            'output_dir': './output',
            'use_parallel': False
        }
        ConfigTemplate.apply_template(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_template_finetune_mode(self):
        """Test apply_template with finetune mode"""
        config = {
            'run_mode': 'finetune',
            'output_dir': './output',
            'use_parallel': False
        }
        ConfigTemplate.apply_template(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_template_predict_mode(self):
        """Test apply_template with predict mode"""
        config = {
            'run_mode': 'predict',
            'output_dir': './output',
            'use_parallel': False
        }
        ConfigTemplate.apply_template(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_template_invalid_mode(self):
        """Test apply_template with invalid run mode"""
        config = {
            'run_mode': 'invalid_mode',
            'output_dir': './output'
        }
        with pytest.raises(ValueError) as exc_info:
            ConfigTemplate.apply_template(config)
        assert 'invalid' in str(exc_info.value).lower()

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_train_template_method(self):
        """Test _train_template method returns correct list"""
        template = ConfigTemplate._train_template()  # pylint: disable=protected-access
        assert isinstance(template, list)
        assert len(template) == len(ConfigTemplate.train_configs)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_predict_template_method(self):
        """Test _predict_template method returns correct list"""
        template = ConfigTemplate._predict_template()  # pylint: disable=protected-access
        assert isinstance(template, list)
        assert len(template) == len(ConfigTemplate.predict_configs)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_update_distributed_parallel_config_train(self):
        """Test update_distributed_parallel_config for train mode"""
        new_config = {}
        origin_config = {
            'tensor_model_parallel_size': 2,
            'pipeline_model_parallel_size': 2
        }
        # Should not raise exception
        ConfigTemplate.update_distributed_parallel_config(
            'distribute_parallel_config',
            new_config,
            origin_config,
            'train'
        )
        assert 'distribute_parallel_config' in new_config

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_update_distributed_parallel_config_predict(self):
        """Test update_distributed_parallel_config for predict mode"""
        new_config = {}
        origin_config = {
            'tensor_model_parallel_size': 2
        }
        # Should not raise exception
        ConfigTemplate.update_distributed_parallel_config(
            'distribute_parallel_config',
            new_config,
            origin_config,
            'predict'
        )
        assert 'distribute_parallel_config' in new_config

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_update_parallel_context_config_train(self):
        """Test update_parallel_context_config for train mode"""
        new_config = {}
        origin_config = {
            'parallel_mode': 1,
            'enable_alltoall': True
        }
        # Should not raise exception
        ConfigTemplate.update_parallel_context_config(
            'parallel',
            new_config,
            origin_config,
            'train'
        )
        assert 'parallel' in new_config

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_update_context_config_train(self):
        """Test update_context_config for train mode"""
        new_config = {}
        origin_config = {
            'mode': 0,
            'device_target': 'Ascend',
            'max_device_memory': '58GB'
        }
        # Should not raise exception
        ConfigTemplate.update_context_config(
            'context',
            new_config,
            origin_config,
            'train'
        )
        assert 'context' in new_config

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_update_context_config_predict(self):
        """Test update_context_config for predict mode"""
        new_config = {}
        origin_config = {
            'mode': 0,
            'device_target': 'Ascend',
            'max_device_memory': '59GB'
        }
        # Should not raise exception
        ConfigTemplate.update_context_config(
            'context',
            new_config,
            origin_config,
            'predict'
        )
        assert 'context' in new_config


# ============================================================================
# Checkpoint Config Tests
# ============================================================================
class TestCheckpointConfig:
    """Test CheckpointConfig class"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_default_values(self):
        """Test default values of CheckpointConfig"""
        default_config = CheckpointConfig.default_value()
        assert default_config['load_checkpoint'] == ''
        assert default_config['load_ckpt_format'] == 'safetensors'
        assert default_config['balanced_load'] is False
        assert default_config['prefix'] == "llm_model"
        assert default_config['save_checkpoint_seconds'] == 0
        assert default_config['save_checkpoint_steps'] == 1
        assert default_config['keep_checkpoint_max'] == 1
        assert default_config['integrated_save'] is False
        assert default_config['async_save'] is False
        assert default_config['remove_redundancy'] is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_with_valid_config(self):
        """Test apply method with valid configuration"""
        config = {
            'save_checkpoint_steps': 100,
            'keep_checkpoint_max': 5,
            'prefix': 'qwen3'
        }
        result = CheckpointConfig.apply('checkpoint_config', config)
        assert result['save_checkpoint_steps'] == 100
        assert result['keep_checkpoint_max'] == 5
        assert result['prefix'] == 'qwen3'

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_with_required_keys(self):
        """Test apply method with required keys"""
        config = {
            'save_checkpoint_steps': 50
        }
        # Should not raise exception as save_checkpoint_steps is provided
        result = CheckpointConfig.apply('checkpoint_config', config)
        assert result['save_checkpoint_steps'] == 50

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_apply_missing_required_key(self):
        """Test apply method missing required key"""
        config = {
            'prefix': 'test_model'
        }
        # Missing save_checkpoint_steps which is required
        with pytest.raises(KeyError):
            CheckpointConfig.apply('checkpoint_config', config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_valid(self):
        """Test validate_config with valid configuration"""
        config = {
            'load_ckpt_format': 'safetensors',
            'balanced_load': False,
            'save_checkpoint_steps': 100,
            'integrated_save': False,
            'async_save': False
        }
        # Should not raise exception
        CheckpointConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_format(self):
        """Test validate_config with invalid checkpoint format"""
        config = {
            'load_ckpt_format': 'pickle',  # Only 'safetensors' is supported
        }
        with pytest.raises(ValueError):
            CheckpointConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_balanced_load(self):
        """Test validate_config with invalid balanced_load value"""
        config = {
            'balanced_load': True,  # Only False is supported
        }
        with pytest.raises(ValueError):
            CheckpointConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_integrated_save(self):
        """Test validate_config with invalid integrated_save value"""
        config = {
            'integrated_save': True,  # Only False is supported
        }
        with pytest.raises(ValueError):
            CheckpointConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_async_save(self):
        """Test validate_config with invalid async_save value"""
        config = {
            'async_save': True,  # Only False is supported
        }
        with pytest.raises(ValueError):
            CheckpointConfig.validate_config(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validate_config_invalid_type(self):
        """Test validate_config with invalid parameter type"""
        config = {
            'save_checkpoint_steps': '100',  # Should be int
        }
        with pytest.raises(ValueError):
            CheckpointConfig.validate_config(config)
