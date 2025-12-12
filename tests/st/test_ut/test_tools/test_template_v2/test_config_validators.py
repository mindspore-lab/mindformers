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
"""Test Suite for Config Validators Module"""
import pytest

from mindformers.tools.register.config_validators import (
    ConfigValidationError,
    validate_pipeline_parallel_config,
    validate_optimizer_parallel_config,
    validate_gradient_aggregation_group,
    validate_sink_size,
    validate_jit_config_training,
    validate_jit_config_inference,
    validate_ascend_config_training,
    validate_ascend_config_inference,
    validate_default_lr_schedule,
    validate_grouped_lr_schedules,
    create_default_lr_schedule_validator,
    create_grouped_lr_schedules_validator,
)

from mindformers.tools.register.llm_template_v2 import (
    LR_SUPPORT_LIST,
    TrainingParallelConfig,
    TrainingConfig,
    ContextConfig,
    InferContextConfig,
)


# ============================================================================
# ConfigValidationError Tests
# ============================================================================
class TestConfigValidationError:
    """Test ConfigValidationError class"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_error_message_format(self):
        """Test error message format"""
        error = ConfigValidationError("test_param", ["error1", "error2"])
        assert "test_param" in str(error)
        assert "error1" in str(error)
        assert "error2" in str(error)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_error_attributes(self):
        """Test error attributes"""
        errors = ["error1", "error2"]
        error = ConfigValidationError("test_param", errors)
        assert error.param_name == "test_param"
        assert error.errors == errors

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_single_error(self):
        """Test with single error"""
        error = ConfigValidationError("param", ["single error"])
        assert "single error" in str(error)


# ============================================================================
# Pipeline Parallel Config Validator Tests
# ============================================================================
class TestValidatePipelineParallelConfig:
    """Test validate_pipeline_parallel_config function"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_valid_config(self):
        """Test with valid configuration"""
        config = {
            "pipeline_interleave": True,
            "pipeline_scheduler": "1f1b",
            "virtual_pipeline_model_parallel_size": 2,
            "pipeline_stage_offset": 0
        }
        assert validate_pipeline_parallel_config(config) is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_valid_config_gpipe_scheduler(self):
        """Test with gpipe scheduler"""
        config = {
            "pipeline_interleave": False,
            "pipeline_scheduler": "gpipe",
            "virtual_pipeline_model_parallel_size": 1,
            "pipeline_stage_offset": [0, 1]
        }
        assert validate_pipeline_parallel_config(config) is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_valid_config_zero_bubble_scheduler(self):
        """Test with zero_bubble_v scheduler"""
        config = {
            "pipeline_interleave": True,
            "pipeline_scheduler": "zero_bubble_v",
            "virtual_pipeline_model_parallel_size": 4,
            "pipeline_stage_offset": "auto"
        }
        assert validate_pipeline_parallel_config(config) is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_invalid_pipeline_interleave_type(self):
        """Test with invalid pipeline_interleave type"""
        config = {
            "pipeline_interleave": "true",  # Should be bool
            "pipeline_scheduler": "1f1b",
            "virtual_pipeline_model_parallel_size": 1,
            "pipeline_stage_offset": 0
        }
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_pipeline_parallel_config(config)
        assert "pipeline_interleave" in str(exc_info.value)
        assert "bool" in str(exc_info.value)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_invalid_pipeline_scheduler(self):
        """Test with invalid pipeline_scheduler"""
        config = {
            "pipeline_interleave": True,
            "pipeline_scheduler": "invalid_scheduler",
            "virtual_pipeline_model_parallel_size": 1,
            "pipeline_stage_offset": 0
        }
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_pipeline_parallel_config(config)
        assert "pipeline_scheduler" in str(exc_info.value)
        assert "1f1b" in str(exc_info.value)
        assert "gpipe" in str(exc_info.value)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_invalid_vpp_size_type(self):
        """Test with invalid virtual_pipeline_model_parallel_size type"""
        config = {
            "pipeline_interleave": True,
            "pipeline_scheduler": "1f1b",
            "virtual_pipeline_model_parallel_size": "2",  # Should be int
            "pipeline_stage_offset": 0
        }
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_pipeline_parallel_config(config)
        assert "virtual_pipeline_model_parallel_size" in str(exc_info.value)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_invalid_vpp_size_value(self):
        """Test with invalid virtual_pipeline_model_parallel_size value"""
        config = {
            "pipeline_interleave": True,
            "pipeline_scheduler": "1f1b",
            "virtual_pipeline_model_parallel_size": 0,  # Should be >= 1
            "pipeline_stage_offset": 0
        }
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_pipeline_parallel_config(config)
        assert "virtual_pipeline_model_parallel_size" in str(exc_info.value)
        assert ">= 1" in str(exc_info.value)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_invalid_pipeline_stage_offset_type(self):
        """Test with invalid pipeline_stage_offset type"""
        config = {
            "pipeline_interleave": True,
            "pipeline_scheduler": "1f1b",
            "virtual_pipeline_model_parallel_size": 1,
            "pipeline_stage_offset": 1.5  # Should be int, list, or str
        }
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_pipeline_parallel_config(config)
        assert "pipeline_stage_offset" in str(exc_info.value)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_multiple_errors(self):
        """Test that multiple errors are collected"""
        config = {
            "pipeline_interleave": "invalid",
            "pipeline_scheduler": "invalid",
            "virtual_pipeline_model_parallel_size": 0,
            "pipeline_stage_offset": None
        }
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_pipeline_parallel_config(config)
        error_msg = str(exc_info.value)
        # Should contain multiple error messages
        assert "pipeline_interleave" in error_msg
        assert "pipeline_scheduler" in error_msg


# ============================================================================
# Optimizer Parallel Config Validator Tests
# ============================================================================
class TestValidateOptimizerParallelConfig:
    """Test validate_optimizer_parallel_config function"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_valid_config(self):
        """Test with valid configuration"""
        config = {
            "enable_parallel_optimizer": True,
            "optimizer_level": "level1",
            "optimizer_weight_shard_size": 2,
            "gradient_accumulation_shard": False
        }
        assert validate_optimizer_parallel_config(config) is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_valid_config_disabled(self):
        """Test with optimizer disabled"""
        config = {
            "enable_parallel_optimizer": False,
            "optimizer_level": "level1",
            "optimizer_weight_shard_size": -1,
            "gradient_accumulation_shard": False
        }
        assert validate_optimizer_parallel_config(config) is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_invalid_enable_parallel_optimizer_type(self):
        """Test with invalid enable_parallel_optimizer type"""
        config = {
            "enable_parallel_optimizer": "true",  # Should be bool
            "optimizer_level": "level1",
            "optimizer_weight_shard_size": 2,
            "gradient_accumulation_shard": False
        }
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_optimizer_parallel_config(config)
        assert "enable_parallel_optimizer" in str(exc_info.value)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_invalid_optimizer_level(self):
        """Test with invalid optimizer_level"""
        config = {
            "enable_parallel_optimizer": True,
            "optimizer_level": "level2",  # Only level1 is supported
            "optimizer_weight_shard_size": 2,
            "gradient_accumulation_shard": False
        }
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_optimizer_parallel_config(config)
        assert "optimizer_level" in str(exc_info.value)
        assert "level1" in str(exc_info.value)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_invalid_weight_shard_size_type(self):
        """Test with invalid optimizer_weight_shard_size type"""
        config = {
            "enable_parallel_optimizer": True,
            "optimizer_level": "level1",
            "optimizer_weight_shard_size": "2",  # Should be int
            "gradient_accumulation_shard": False
        }
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_optimizer_parallel_config(config)
        assert "optimizer_weight_shard_size" in str(exc_info.value)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_invalid_gradient_accumulation_shard(self):
        """Test with invalid gradient_accumulation_shard value"""
        config = {
            "enable_parallel_optimizer": True,
            "optimizer_level": "level1",
            "optimizer_weight_shard_size": 2,
            "gradient_accumulation_shard": True  # Must be False
        }
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_optimizer_parallel_config(config)
        assert "gradient_accumulation_shard" in str(exc_info.value)
        assert "False" in str(exc_info.value)


# ============================================================================
# Simple Value Validator Tests
# ============================================================================
class TestValidateGradientAggregationGroup:
    """Test validate_gradient_aggregation_group function"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_valid_value_one(self):
        """Test with valid value 1"""
        assert validate_gradient_aggregation_group(1) is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_valid_value_greater_than_one(self):
        """Test with valid value greater than 1"""
        assert validate_gradient_aggregation_group(4) is True
        assert validate_gradient_aggregation_group(8) is True
        assert validate_gradient_aggregation_group(100) is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_invalid_value_zero(self):
        """Test with invalid value 0"""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_gradient_aggregation_group(0)
        assert "gradient_aggregation_group" in str(exc_info.value)
        assert ">= 1" in str(exc_info.value)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_invalid_value_negative(self):
        """Test with invalid negative value"""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_gradient_aggregation_group(-1)
        assert "gradient_aggregation_group" in str(exc_info.value)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_invalid_type(self):
        """Test with invalid type"""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_gradient_aggregation_group("1")
        assert "gradient_aggregation_group" in str(exc_info.value)


class TestValidateSinkSize:
    """Test validate_sink_size function"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_valid_value(self):
        """Test with valid value 1"""
        assert validate_sink_size(1) is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_invalid_value_zero(self):
        """Test with invalid value 0"""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_sink_size(0)
        assert "sink_size" in str(exc_info.value)
        assert "1" in str(exc_info.value)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_invalid_value_two(self):
        """Test with invalid value 2"""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_sink_size(2)
        assert "sink_size" in str(exc_info.value)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_invalid_value_negative(self):
        """Test with invalid negative value"""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_sink_size(-1)
        assert "sink_size" in str(exc_info.value)


# ============================================================================
# JIT Config Validator Tests
# ============================================================================
class TestValidateJitConfigTraining:
    """Test validate_jit_config_training function"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_valid_jit_level_o0(self):
        """Test with valid jit_level O0"""
        config = {"jit_level": "O0"}
        assert validate_jit_config_training(config) is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_valid_jit_level_o1(self):
        """Test with valid jit_level O1"""
        config = {"jit_level": "O1"}
        assert validate_jit_config_training(config) is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_invalid_jit_level(self):
        """Test with invalid jit_level"""
        config = {"jit_level": "O2"}
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_jit_config_training(config)
        assert "jit_level" in str(exc_info.value)
        assert "O0" in str(exc_info.value)
        assert "O1" in str(exc_info.value)


class TestValidateJitConfigInference:
    """Test validate_jit_config_inference function"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_valid_jit_level_o0(self):
        """Test with valid jit_level O0"""
        config = {"jit_level": "O0"}
        assert validate_jit_config_inference(config) is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_invalid_jit_level_o1(self):
        """Test with invalid jit_level O1 for inference"""
        config = {"jit_level": "O1"}
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_jit_config_inference(config)
        assert "jit_level" in str(exc_info.value)
        assert "O0" in str(exc_info.value)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_invalid_jit_level_o2(self):
        """Test with invalid jit_level O2"""
        config = {"jit_level": "O2"}
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_jit_config_inference(config)
        assert "jit_level" in str(exc_info.value)


# ============================================================================
# Ascend Config Validator Tests
# ============================================================================
class TestValidateAscendConfigTraining:
    """Test validate_ascend_config_training function"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_valid_config(self):
        """Test with valid configuration"""
        config = {
            "precision_mode": "must_keep_origin_dtype"
        }
        assert validate_ascend_config_training(config) is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_valid_config_with_parallel_speed_up_path(self):
        """Test with valid parallel_speed_up_json_path"""
        config = {
            "precision_mode": "must_keep_origin_dtype",
            "parallel_speed_up_json_path": "./parallel_speed_up.json"
        }
        assert validate_ascend_config_training(config) is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_valid_config_with_none_parallel_speed_up_path(self):
        """Test with None parallel_speed_up_json_path"""
        config = {
            "precision_mode": "must_keep_origin_dtype",
            "parallel_speed_up_json_path": None
        }
        assert validate_ascend_config_training(config) is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_invalid_precision_mode(self):
        """Test with invalid precision_mode"""
        config = {
            "precision_mode": "fp16"
        }
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_ascend_config_training(config)
        assert "precision_mode" in str(exc_info.value)
        assert "must_keep_origin_dtype" in str(exc_info.value)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_invalid_parallel_speed_up_path_type(self):
        """Test with invalid parallel_speed_up_json_path type"""
        config = {
            "precision_mode": "must_keep_origin_dtype",
            "parallel_speed_up_json_path": 123  # Should be str or None
        }
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_ascend_config_training(config)
        assert "parallel_speed_up_json_path" in str(exc_info.value)


class TestValidateAscendConfigInference:
    """Test validate_ascend_config_inference function"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_valid_config(self):
        """Test with valid configuration"""
        config = {
            "precision_mode": "must_keep_origin_dtype"
        }
        assert validate_ascend_config_inference(config) is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_invalid_precision_mode(self):
        """Test with invalid precision_mode"""
        config = {
            "precision_mode": "fp16"
        }
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_ascend_config_inference(config)
        assert "precision_mode" in str(exc_info.value)


# ============================================================================
# Learning Rate Schedule Validator Tests
# ============================================================================
class TestValidateDefaultLrSchedule:
    """Test validate_default_lr_schedule function"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_valid_config_with_warmup_ratio(self):
        """Test with valid config using warmup_ratio"""
        config = {
            "type": "CosineWithWarmUpLR",
            "warmup_ratio": 0.1
        }
        assert validate_default_lr_schedule(config, LR_SUPPORT_LIST) is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_valid_config_with_warmup_steps(self):
        """Test with valid config using warmup_steps"""
        config = {
            "type": "LinearWithWarmUpLR",
            "warmup_steps": 1000
        }
        assert validate_default_lr_schedule(config, LR_SUPPORT_LIST) is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_valid_all_lr_types(self):
        """Test with all supported LR types"""
        for lr_type in LR_SUPPORT_LIST:
            config = {
                "type": lr_type,
                "warmup_ratio": 0.1
            }
            assert validate_default_lr_schedule(config, LR_SUPPORT_LIST) is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_invalid_type(self):
        """Test with invalid type"""
        config = {
            "type": "InvalidLR",
            "warmup_ratio": 0.1
        }
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_default_lr_schedule(config, LR_SUPPORT_LIST)
        assert "type" in str(exc_info.value)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_missing_warmup(self):
        """Test with missing warmup_ratio and warmup_steps"""
        config = {
            "type": "CosineWithWarmUpLR"
        }
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_default_lr_schedule(config, LR_SUPPORT_LIST)
        assert "warmup_ratio" in str(exc_info.value)
        assert "warmup_steps" in str(exc_info.value)


class TestValidateGroupedLrSchedules:
    """Test validate_grouped_lr_schedules function"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_valid_config(self):
        """Test with valid configuration"""
        config_list = [
            {
                "type": "CosineWithWarmUpLR",
                "params": ["embedding.*"],
                "warmup_ratio": 0.1
            },
            {
                "type": "LinearWithWarmUpLR",
                "params": ["layer.*"],
                "warmup_steps": 1000
            }
        ]
        assert validate_grouped_lr_schedules(config_list, LR_SUPPORT_LIST) is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_valid_empty_list(self):
        """Test with empty list"""
        assert validate_grouped_lr_schedules([], LR_SUPPORT_LIST) is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_invalid_not_dict(self):
        """Test with non-dict item in list"""
        config_list = [
            "invalid"
        ]
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_grouped_lr_schedules(config_list, LR_SUPPORT_LIST)
        assert "grouped[0]" in str(exc_info.value)
        assert "dict" in str(exc_info.value)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_invalid_type(self):
        """Test with invalid type"""
        config_list = [
            {
                "type": "InvalidLR",
                "params": ["embedding.*"],
                "warmup_ratio": 0.1
            }
        ]
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_grouped_lr_schedules(config_list, LR_SUPPORT_LIST)
        assert "grouped[0]" in str(exc_info.value)
        assert "type" in str(exc_info.value)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_invalid_params_type(self):
        """Test with invalid params type"""
        config_list = [
            {
                "type": "CosineWithWarmUpLR",
                "params": "embedding.*",  # Should be list
                "warmup_ratio": 0.1
            }
        ]
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_grouped_lr_schedules(config_list, LR_SUPPORT_LIST)
        assert "grouped[0]" in str(exc_info.value)
        assert "params" in str(exc_info.value)
        assert "list" in str(exc_info.value)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_missing_warmup(self):
        """Test with missing warmup"""
        config_list = [
            {
                "type": "CosineWithWarmUpLR",
                "params": ["embedding.*"]
            }
        ]
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_grouped_lr_schedules(config_list, LR_SUPPORT_LIST)
        assert "grouped[0]" in str(exc_info.value)
        assert "warmup" in str(exc_info.value)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_multiple_errors_in_single_item(self):
        """Test multiple errors in a single item"""
        config_list = [
            {
                "type": "InvalidLR",
                "params": "not_a_list"
                # missing warmup
            }
        ]
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_grouped_lr_schedules(config_list, LR_SUPPORT_LIST)
        error_msg = str(exc_info.value)
        # Should contain multiple error messages
        assert "type" in error_msg
        assert "params" in error_msg
        assert "warmup" in error_msg

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_multiple_items_with_errors(self):
        """Test errors across multiple items"""
        config_list = [
            {
                "type": "InvalidLR1",
                "params": ["layer1"],
                "warmup_ratio": 0.1
            },
            {
                "type": "InvalidLR2",
                "params": ["layer2"],
                "warmup_steps": 100
            }
        ]
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_grouped_lr_schedules(config_list, LR_SUPPORT_LIST)
        error_msg = str(exc_info.value)
        assert "grouped[0]" in error_msg
        assert "grouped[1]" in error_msg


# ============================================================================
# Factory Function Tests
# ============================================================================
class TestCreateDefaultLrScheduleValidator:
    """Test create_default_lr_schedule_validator factory function"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_creates_working_validator(self):
        """Test that factory creates a working validator"""
        validator = create_default_lr_schedule_validator(LR_SUPPORT_LIST)
        config = {
            "type": "CosineWithWarmUpLR",
            "warmup_ratio": 0.1
        }
        assert validator(config) is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validator_raises_on_invalid(self):
        """Test that created validator raises on invalid config"""
        validator = create_default_lr_schedule_validator(LR_SUPPORT_LIST)
        config = {
            "type": "InvalidLR",
            "warmup_ratio": 0.1
        }
        with pytest.raises(ConfigValidationError):
            validator(config)


class TestCreateGroupedLrSchedulesValidator:
    """Test create_grouped_lr_schedules_validator factory function"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_creates_working_validator(self):
        """Test that factory creates a working validator"""
        validator = create_grouped_lr_schedules_validator(LR_SUPPORT_LIST)
        config_list = [
            {
                "type": "CosineWithWarmUpLR",
                "params": ["embedding.*"],
                "warmup_ratio": 0.1
            }
        ]
        assert validator(config_list) is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_validator_raises_on_invalid(self):
        """Test that created validator raises on invalid config"""
        validator = create_grouped_lr_schedules_validator(LR_SUPPORT_LIST)
        config_list = [
            {
                "type": "InvalidLR",
                "params": ["embedding.*"],
                "warmup_ratio": 0.1
            }
        ]
        with pytest.raises(ConfigValidationError):
            validator(config_list)


# ============================================================================
# Integration Tests with llm_template_v2
# ============================================================================
class TestIntegrationWithLlmTemplateV2:
    """Integration tests to verify validators work correctly with llm_template_v2"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_training_parallel_config_validation(self):
        """Test that TrainingParallelConfig uses new validators correctly"""
        # Valid pipeline_parallel_config
        config = {
            'pipeline_parallel_config': {
                'pipeline_interleave': True,
                'pipeline_scheduler': '1f1b',
                'virtual_pipeline_model_parallel_size': 2,
                'pipeline_stage_offset': 0
            }
        }
        result = TrainingParallelConfig.apply('training_parallel_config', config)
        TrainingParallelConfig.validate_config(result)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_training_parallel_config_invalid_pipeline(self):
        """Test validation fails with invalid pipeline config"""
        config = {
            'pipeline_parallel_config': {
                'pipeline_interleave': "invalid",  # Should be bool
                'pipeline_scheduler': '1f1b',
                'virtual_pipeline_model_parallel_size': 1,
                'pipeline_stage_offset': 0
            }
        }
        result = TrainingParallelConfig.apply('training_parallel_config', config)
        with pytest.raises(ValueError):
            TrainingParallelConfig.validate_config(result)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_training_config_sink_size_validation(self):
        """Test that TrainingConfig uses sink_size validator correctly"""
        # Valid sink_size
        config = {
            'sink_size': 1
        }
        result = TrainingConfig.apply('training_args', config)
        TrainingConfig.validate_config(result)

        # Invalid sink_size
        config = {
            'sink_size': 2
        }
        result = TrainingConfig.apply('training_args', config)
        with pytest.raises(ValueError):
            TrainingConfig.validate_config(result)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_context_config_jit_validation(self):
        """Test that ContextConfig uses jit validators correctly"""
        # Valid jit_config
        config = {
            'jit_config': {'jit_level': 'O0'}
        }
        result = ContextConfig.apply('context', config)
        ContextConfig.validate_config(result)

        # Invalid jit_config
        config = {
            'jit_config': {'jit_level': 'O2'}
        }
        result = ContextConfig.apply('context', config)
        with pytest.raises(ValueError):
            ContextConfig.validate_config(result)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_context_config_ascend_validation(self):
        """Test that ContextConfig uses ascend validators correctly"""
        # Valid ascend_config (also specify jit_config to avoid state pollution)
        config = {
            'jit_config': {'jit_level': 'O0'},
            'ascend_config': {'precision_mode': 'must_keep_origin_dtype'}
        }
        result = ContextConfig.apply('context', config)
        ContextConfig.validate_config(result)

        # Invalid ascend_config
        config = {
            'jit_config': {'jit_level': 'O0'},
            'ascend_config': {'precision_mode': 'fp16'}
        }
        result = ContextConfig.apply('context', config)
        with pytest.raises(ValueError):
            ContextConfig.validate_config(result)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_infer_context_config_jit_validation(self):
        """Test that InferContextConfig uses inference jit validators correctly"""
        # Valid jit_config for inference (only O0)
        config = {
            'jit_config': {'jit_level': 'O0'}
        }
        result = InferContextConfig.apply('infer_context', config)
        InferContextConfig.validate_config(result)

        # Invalid jit_config for inference (O1 not allowed)
        config = {
            'jit_config': {'jit_level': 'O1'}
        }
        result = InferContextConfig.apply('infer_context', config)
        with pytest.raises(ValueError):
            InferContextConfig.validate_config(result)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_error_message_clarity(self):
        """Test that error messages are clear and helpful"""
        config = {
            'pipeline_parallel_config': {
                'pipeline_interleave': "not_a_bool",
                'pipeline_scheduler': 'invalid_scheduler',
                'virtual_pipeline_model_parallel_size': 0,
                'pipeline_stage_offset': 1.5
            }
        }
        result = TrainingParallelConfig.apply('training_parallel_config', config)

        try:
            TrainingParallelConfig.validate_config(result)
            assert False, "Should have raised ValueError"
        except ValueError as err:
            error_msg = str(err)
            # Check that error message is informative
            assert "pipeline_parallel_config" in error_msg
            # The error should contain specific field information
            assert "pipeline_interleave" in error_msg or "pipeline_scheduler" in error_msg
