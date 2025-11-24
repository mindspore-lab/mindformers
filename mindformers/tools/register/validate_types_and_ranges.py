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
"""Utility functions for validating configuration parameter types and value ranges."""

from typing import Any, Union, Tuple, List, Optional, Callable


def _normalize_param_type(
        param_type: Union[type, Tuple[type, ...]],
        param_value: Any
) -> Tuple[Union[type, Tuple[type, ...]], bool]:
    """
    Normalize parameter type by handling None values in type tuple.

    Args:
        param_type: Expected type or tuple of acceptable types
        param_value: The parameter value to validate

    Returns:
        Tuple of (normalized_type, allows_none)

    Raises:
        TypeError: If param_value is None but None is not allowed
    """
    if isinstance(param_type, tuple) and None in param_type:
        normalized_type = tuple(item for item in param_type if item is not None)
        return normalized_type, True

    if param_value is None and isinstance(param_type, tuple):
        raise TypeError(f"Parameter must be of type {param_type} but got None.")

    return param_type, False


def _validate_type(
        param_value: Any,
        param_type: Union[type, Tuple[type, ...]],
        allows_none: bool,
        param_name: str
) -> None:
    """
    Validate that param_value matches the expected type.

    Args:
        param_value: The parameter value to validate
        param_type: Expected type or tuple of acceptable types
        allows_none: Whether None is allowed as a valid value
        param_name: Name of the parameter for error messages

    Raises:
        TypeError: If the parameter value is not of the expected type
    """
    if allows_none and param_value is None:
        return

    if isinstance(param_value, param_type):
        return

    if isinstance(param_type, tuple):
        type_names = ", ".join(t.__name__ for t in param_type)
        raise TypeError(
            f"{param_name} must be one of types: {type_names}. "
            f"Got {type(param_value).__name__} instead."
        )

    raise TypeError(
        f"{param_name} must be of type {param_type.__name__}. "
        f"Got {type(param_value).__name__} instead."
    )


def _validate_min_max_range(param_value: Any, value_range: Tuple, param_name: str) -> None:
    """Validate value is within min-max range."""
    min_val, max_val = value_range
    if min_val > param_value or param_value > max_val:
        raise ValueError(
            f"{param_name} must be between {min_val} and {max_val}. "
            f"Got {param_value} instead."
        )


def _validate_allowed_values(param_value: Any, value_range: Union[List, Tuple], param_name: str) -> None:
    """Validate value is in the list of allowed values."""
    if param_value not in value_range:
        raise ValueError(
            f"{param_name} must be one of {value_range}. "
            f"Got {param_value} instead."
        )


def _validate_range_object(param_value: Any, value_range: range, param_name: str) -> None:
    """Validate value is in the range object."""
    if param_value not in value_range:
        raise ValueError(
            f"{param_name} must be in range {value_range}. "
            f"Got {param_value} instead."
        )


def _validate_custom_function(param_value: Any, value_range: Callable, param_name: str) -> None:
    """Validate value using custom validation function."""
    if not value_range(param_value):
        raise ValueError(
            f"{param_name} failed custom validation. "
            f"Value: {param_value}"
        )


def _validate_value_range(
        param_value: Any,
        value_range: Optional[Union[Tuple, List, range, Callable]],
        param_name: str
) -> None:
    """
    Validate the value range of a parameter.

    Args:
        param_value: The parameter value to validate
        value_range: Range or constraints for the value
        param_name: Name of the parameter for error messages

    Raises:
        ValueError: If the parameter value is outside the allowed range
    """
    if value_range is None:
        return

    # Case 1: Range specified as (min, max) tuple
    if isinstance(value_range, tuple) and len(value_range) == 2:
        _validate_min_max_range(param_value, value_range, param_name)
        return

    # Case 2: List of allowed values
    if isinstance(value_range, (list, tuple)) and len(value_range) > 0:
        _validate_allowed_values(param_value, value_range, param_name)
        return

    # Case 3: Range object
    if isinstance(value_range, range):
        _validate_range_object(param_value, value_range, param_name)
        return

    # Case 4: Custom validation function
    if callable(value_range):
        _validate_custom_function(param_value, value_range, param_name)
        return

    # Case 5: Unsupported range specification
    raise ValueError(f"Unsupported range specification: {value_range}")


def validate_config_types_and_ranges(
        param_value: Any,
        param_type: Union[type, Tuple[type, ...]],
        value_range: Optional[Union[Tuple, List, range, Callable]] = None,
        param_name: str = "parameter"
) -> None:
    """
    Validate the type and value range of a configuration parameter.

    Args:
        param_value: The parameter value to validate
        param_type: Expected type or tuple of acceptable types
        value_range: Optional range or constraints for the value.
                   Can be a tuple (min, max), list of allowed values,
                   range object, or a validation function.
        param_name: Name of the parameter for error messages

    Raises:
        TypeError: If the parameter value is not of the expected type
        ValueError: If the parameter value is outside the allowed range
    """
    # 1. Validate type
    normalized_type, allows_none = _normalize_param_type(param_type, param_value)
    _validate_type(param_value, normalized_type, allows_none, param_name)

    # 2. Validate range/constraints if provided
    _validate_value_range(param_value, value_range, param_name)
