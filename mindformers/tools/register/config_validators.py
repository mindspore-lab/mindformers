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
"""Configuration validators with detailed error messages for llm_template_v2."""

from typing import Any, Dict, List


class ConfigValidationError(ValueError):
    """Custom exception for configuration validation errors with detailed messages."""

    def __init__(self, param_name: str, errors: List[str]):
        self.param_name = param_name
        self.errors = errors
        message = f"Validation failed for '{param_name}':\n" + "\n".join(f"  - {e}" for e in errors)
        super().__init__(message)


def _validate_field_type(config: Dict[str, Any], field: str, expected_type: type,
                         errors: List[str]) -> bool:
    """
    Validate that a field in the config is of the expected type.

    Args:
        config: Configuration dictionary
        field: Field name to validate
        expected_type: Expected type for the field
        errors: List to append error messages to

    Returns:
        True if validation passed, False otherwise
    """
    value = config.get(field)
    if not isinstance(value, expected_type):
        actual_type = type(value).__name__ if value is not None else "None"
        errors.append(f"'{field}' must be of type {expected_type.__name__}, got {actual_type}")
        return False
    return True


def _validate_field_in_list(config: Dict[str, Any], field: str, allowed_values: List[Any],
                            errors: List[str]) -> bool:
    """
    Validate that a field value is in the list of allowed values.

    Args:
        config: Configuration dictionary
        field: Field name to validate
        allowed_values: List of allowed values
        errors: List to append error messages to

    Returns:
        True if validation passed, False otherwise
    """
    value = config.get(field)
    if value not in allowed_values:
        errors.append(f"'{field}' must be one of {allowed_values}, got '{value}'")
        return False
    return True


def _validate_field_gte(config: Dict[str, Any], field: str, min_value: int,
                        errors: List[str]) -> bool:
    """
    Validate that a field value is greater than or equal to a minimum value.

    Args:
        config: Configuration dictionary
        field: Field name to validate
        min_value: Minimum allowed value
        errors: List to append error messages to

    Returns:
        True if validation passed, False otherwise
    """
    value = config.get(field)
    if not isinstance(value, int) or value < min_value:
        errors.append(f"'{field}' must be an integer >= {min_value}, got '{value}'")
        return False
    return True


def _validate_field_eq(config: Dict[str, Any], field: str, expected_value: Any,
                       errors: List[str]) -> bool:
    """
    Validate that a field value equals the expected value.

    Args:
        config: Configuration dictionary
        field: Field name to validate
        expected_value: Expected value
        errors: List to append error messages to

    Returns:
        True if validation passed, False otherwise
    """
    value = config.get(field)
    if value != expected_value:
        errors.append(f"'{field}' must be {expected_value}, got '{value}'")
        return False
    return True


def _validate_field_is_false(config: Dict[str, Any], field: str,
                             errors: List[str]) -> bool:
    """
    Validate that a field value is False.

    Args:
        config: Configuration dictionary
        field: Field name to validate
        errors: List to append error messages to

    Returns:
        True if validation passed, False otherwise
    """
    value = config.get(field)
    if value:
        errors.append(f"'{field}' must be False, got '{value}'")
        return False
    return True


def _validate_field_optional_type(config: Dict[str, Any], field: str,
                                  expected_type: type, errors: List[str]) -> bool:
    """
    Validate that a field is either None or of the expected type.

    Args:
        config: Configuration dictionary
        field: Field name to validate
        expected_type: Expected type if not None
        errors: List to append error messages to

    Returns:
        True if validation passed, False otherwise
    """
    value = config.get(field)
    if value is not None and not isinstance(value, expected_type):
        actual_type = type(value).__name__
        errors.append(f"'{field}' must be None or {expected_type.__name__}, got {actual_type}")
        return False
    return True


# ============================================================================
# Pipeline Parallel Config Validators
# ============================================================================

def validate_pipeline_parallel_config(config: Dict[str, Any]) -> bool:
    """
    Validate pipeline_parallel_config dictionary.

    Required validations:
    - pipeline_interleave: must be bool
    - pipeline_scheduler: must be one of ["1f1b", "gpipe", "zero_bubble_v"]
    - virtual_pipeline_model_parallel_size: must be int >= 1
    - pipeline_stage_offset: must be int, list, or str

    Args:
        config: Pipeline parallel configuration dictionary

    Returns:
        True if validation passed

    Raises:
        ConfigValidationError: If validation fails with detailed error messages
    """
    errors = []

    _validate_field_type(config, "pipeline_interleave", bool, errors)
    _validate_field_in_list(config, "pipeline_scheduler", ["1f1b", "gpipe", "zero_bubble_v"], errors)

    # Validate virtual_pipeline_model_parallel_size type and value
    vpp_size = config.get("virtual_pipeline_model_parallel_size")
    if not isinstance(vpp_size, int):
        errors.append(f"'virtual_pipeline_model_parallel_size' must be int, got {type(vpp_size).__name__}")
    elif vpp_size < 1:
        errors.append(f"'virtual_pipeline_model_parallel_size' must be >= 1, got {vpp_size}")

    # Validate pipeline_stage_offset type
    offset = config.get("pipeline_stage_offset")
    if not isinstance(offset, (int, list, str)):
        actual_type = type(offset).__name__ if offset is not None else "None"
        errors.append(f"'pipeline_stage_offset' must be int, list, or str, got {actual_type}")

    if errors:
        raise ConfigValidationError("pipeline_parallel_config", errors)
    return True


# ============================================================================
# Optimizer Parallel Config Validators
# ============================================================================

def validate_optimizer_parallel_config(config: Dict[str, Any]) -> bool:
    """
    Validate optimizer_parallel_config dictionary.

    Required validations:
    - enable_parallel_optimizer: must be bool
    - optimizer_level: must be "level1"
    - optimizer_weight_shard_size: must be int
    - gradient_accumulation_shard: must be False

    Args:
        config: Optimizer parallel configuration dictionary

    Returns:
        True if validation passed

    Raises:
        ConfigValidationError: If validation fails with detailed error messages
    """
    errors = []

    _validate_field_type(config, "enable_parallel_optimizer", bool, errors)
    _validate_field_in_list(config, "optimizer_level", ["level1"], errors)
    _validate_field_type(config, "optimizer_weight_shard_size", int, errors)
    _validate_field_is_false(config, "gradient_accumulation_shard", errors)

    if errors:
        raise ConfigValidationError("optimizer_parallel_config", errors)
    return True


# ============================================================================
# Simple Value Validators
# ============================================================================

def validate_gradient_aggregation_group(value: int) -> bool:
    """
    Validate gradient_aggregation_group value.

    Args:
        value: Must be >= 1

    Returns:
        True if validation passed

    Raises:
        ConfigValidationError: If value < 1
    """
    if not isinstance(value, int) or value < 1:
        raise ConfigValidationError("gradient_aggregation_group",
                                    [f"Must be an integer >= 1, got '{value}'"])
    return True


def validate_sink_size(value: int) -> bool:
    """
    Validate sink_size value.

    Args:
        value: Must be exactly 1

    Returns:
        True if validation passed

    Raises:
        ConfigValidationError: If value != 1
    """
    if value != 1:
        raise ConfigValidationError("sink_size",
                                    [f"Must be exactly 1, got '{value}'"])
    return True


# ============================================================================
# JIT Config Validators
# ============================================================================

def validate_jit_config_training(config: Dict[str, Any]) -> bool:
    """
    Validate jit_config for training context.

    Args:
        config: JIT configuration dictionary with jit_level

    Returns:
        True if validation passed

    Raises:
        ConfigValidationError: If jit_level is not in ["O0", "O1"]
    """
    errors = []
    _validate_field_in_list(config, "jit_level", ["O0", "O1"], errors)

    if errors:
        raise ConfigValidationError("jit_config", errors)
    return True


def validate_jit_config_inference(config: Dict[str, Any]) -> bool:
    """
    Validate jit_config for inference context.

    Args:
        config: JIT configuration dictionary with jit_level

    Returns:
        True if validation passed

    Raises:
        ConfigValidationError: If jit_level is not "O0"
    """
    errors = []
    _validate_field_in_list(config, "jit_level", ["O0"], errors)

    if errors:
        raise ConfigValidationError("jit_config", errors)
    return True


# ============================================================================
# Ascend Config Validators
# ============================================================================

def validate_ascend_config_training(config: Dict[str, Any]) -> bool:
    """
    Validate ascend_config for training context.

    Required validations:
    - precision_mode: must be "must_keep_origin_dtype"
    - parallel_speed_up_json_path: must be None or str

    Args:
        config: Ascend configuration dictionary

    Returns:
        True if validation passed

    Raises:
        ConfigValidationError: If validation fails with detailed error messages
    """
    errors = []

    _validate_field_in_list(config, "precision_mode", ["must_keep_origin_dtype"], errors)
    _validate_field_optional_type(config, "parallel_speed_up_json_path", str, errors)

    if errors:
        raise ConfigValidationError("ascend_config", errors)
    return True


def validate_ascend_config_inference(config: Dict[str, Any]) -> bool:
    """
    Validate ascend_config for inference context.

    Args:
        config: Ascend configuration dictionary

    Returns:
        True if validation passed

    Raises:
        ConfigValidationError: If precision_mode is not "must_keep_origin_dtype"
    """
    errors = []
    _validate_field_in_list(config, "precision_mode", ["must_keep_origin_dtype"], errors)

    if errors:
        raise ConfigValidationError("ascend_config", errors)
    return True


# ============================================================================
# Learning Rate Schedule Validators
# ============================================================================

def validate_default_lr_schedule(config: Dict[str, Any], lr_support_list: List[str]) -> bool:
    """
    Validate default learning rate schedule configuration.

    Required validations:
    - type: must be in lr_support_list
    - Either warmup_ratio or warmup_steps must be provided

    Args:
        config: Default learning rate schedule configuration
        lr_support_list: List of supported learning rate schedule types

    Returns:
        True if validation passed

    Raises:
        ConfigValidationError: If validation fails with detailed error messages
    """
    errors = []

    lr_type = config.get("type")
    if lr_type not in lr_support_list:
        errors.append(f"'type' must be one of {lr_support_list}, got '{lr_type}'")

    warmup_ratio = config.get("warmup_ratio")
    warmup_steps = config.get("warmup_steps")
    if warmup_ratio is None and warmup_steps is None:
        errors.append("Either 'warmup_ratio' or 'warmup_steps' must be provided")

    if errors:
        raise ConfigValidationError("default (lr_schedule)", errors)
    return True


def validate_grouped_lr_schedules(config_list: List[Dict[str, Any]], lr_support_list: List[str]) -> bool:
    """
    Validate grouped learning rate schedule configurations.

    Required validations for each grouped_lr:
    - Must be a dict
    - type: must be in lr_support_list
    - params: must be a list
    - Either warmup_ratio or warmup_steps must be provided

    Args:
        config_list: List of grouped learning rate schedule configurations
        lr_support_list: List of supported learning rate schedule types

    Returns:
        True if validation passed

    Raises:
        ConfigValidationError: If validation fails with detailed error messages
    """
    errors = []

    for idx, grouped_lr in enumerate(config_list):
        prefix = f"grouped[{idx}]"

        if not isinstance(grouped_lr, dict):
            errors.append(f"{prefix}: must be a dict, got {type(grouped_lr).__name__}")
            continue

        lr_type = grouped_lr.get("type")
        if lr_type not in lr_support_list:
            errors.append(f"{prefix}: 'type' must be one of {lr_support_list}, got '{lr_type}'")

        params = grouped_lr.get("params")
        if not isinstance(params, list):
            actual_type = type(params).__name__ if params is not None else "None"
            errors.append(f"{prefix}: 'params' must be a list, got {actual_type}")

        warmup_ratio = grouped_lr.get("warmup_ratio")
        warmup_steps = grouped_lr.get("warmup_steps")
        if warmup_ratio is None and warmup_steps is None:
            errors.append(f"{prefix}: Either 'warmup_ratio' or 'warmup_steps' must be provided")

    if errors:
        raise ConfigValidationError("grouped (lr_schedule)", errors)
    return True


# ============================================================================
# Factory functions for creating validators with bound parameters
# ============================================================================

def create_default_lr_schedule_validator(lr_support_list: List[str]):
    """Create a validator function for default learning rate schedule with bound lr_support_list."""
    def validator(config: Dict[str, Any]) -> bool:
        return validate_default_lr_schedule(config, lr_support_list)
    return validator


def create_grouped_lr_schedules_validator(lr_support_list: List[str]):
    """Create a validator function for grouped learning rate schedules with bound lr_support_list."""
    def validator(config_list: List[Dict[str, Any]]) -> bool:
        return validate_grouped_lr_schedules(config_list, lr_support_list)
    return validator
