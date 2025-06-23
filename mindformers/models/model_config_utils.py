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
"""Check HuggingFace ModelConfig And Register mf parameter by decorator."""

import inspect
from dataclasses import dataclass, asdict
from typing import List, Tuple
from functools import wraps

from mindformers.parallel_core.mf_model_config import MFModelConfig
from mindformers.tools.logger import logger


def validate_ignore_parameter_format(ignore_list: List[Tuple[str, str]] = None):
    """
    Args:
        ignore_list: The ignore parameter list to be validated.

    Return:
        A tuple list of ignored information that passed the validation.
        The tuple structure is: (Parameter Key, String explaining why the parameter is ignored)

    Raises:
        TypeError: Raised when the data type is invalid.
        ValueError: Raised when the data structure is invalid.
    """
    # 1. Check if it is a list.
    if not isinstance(ignore_list, list):
        raise TypeError(f"IGNORE_COMMON_HF_PARAMETER must be a list, got {type(ignore_list).__name__}")

    # 2. Check if each element is a tuple.
    for i, item in enumerate(ignore_list):
        if not isinstance(item, tuple):
            raise TypeError(
                f"Item at index {i} must be a tuple, got {type(item).__name__}. "
                f"Expected format: [('param_name', 'reason'), ...]"
            )

    # 3. Check if the tuple length is `2`.
    for i, item in enumerate(ignore_list):
        if len(item) != 2:
            raise ValueError(
                f"Tuple at index {i} must have exactly 2 elements, got {len(item)}. "
                f"Expected format: ('param_name', 'ignore_reason')"
            )

    # 4. Checks if the first element is a string.
    for i, (param_name, _) in enumerate(ignore_list):
        if not isinstance(param_name, str):
            raise TypeError(
                f"First element in tuple at index {i} must be a string (parameter name), "
                f"got {type(param_name).__name__}"
            )

    # 5. Checks if the second element is of a valid type.
    for i, (_, reason) in enumerate(ignore_list):
        if not isinstance(reason, str):
            raise TypeError(
                f"Second element in tuple at index {i} must be a string or have a string representation, "
                f"got {type(reason).__name__}"
            )

    return ignore_list


@dataclass
class NotSupportedInfo:
    """Not supported info of the HuggingFace Config Keys for `ignore_and_delete_parameter` to print."""
    useless = "Useless"  # useless for MF
    not_implemented = "NotImplemented"  # not supported for now


IGNORE_COMMON_HF_PARAMETER = [
    ('auto_map', NotSupportedInfo.useless),
    ('torch_dtype', 'Useless, replace by compute_dtype'),
    ('use_cache', 'Useless, enable kv_cache by default in MF'),
    ('transformers_version', NotSupportedInfo.useless),
]

PRINTED_CLASSES = set()


def ignore_and_delete_parameter(extra_ignore_param: List[Tuple[str, str]] = None):
    """
    A function-based decorator to intercept and print unsupported parameters in __init__,
    and delete corresponding attributes after initialization.

    Args:
        ignore_type (str): Type of model/config to determine ignored parameters.
        extra_ignore_param (List[Tuple[str, str]]): Additional parameters to ignore.

    Returns:
        The decorator function.
    """
    # Default list of common parameters to ignore
    ignore_info = IGNORE_COMMON_HF_PARAMETER.copy()

    if extra_ignore_param:
        model_ignore = validate_ignore_parameter_format(extra_ignore_param)
        ignore_info.extend(model_ignore)

    # Extract all parameter names to ignore
    ignore_param_names = [item[0] for item in ignore_info]

    def decorator(init_func):
        @wraps(init_func)
        def wrapper(self_instance, *args, **kwargs):
            # Get original signature parameters
            sig = inspect.signature(init_func)
            all_parameters_kwargs = dict()
            for name, param in sig.parameters.items():
                if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue
                if name in ('self', 'kwargs'):
                    continue
                all_parameters_kwargs.setdefault(name, param.default)

            # Merge __init__ default signature parameters with custom input parameters
            # Attention:
            # kwargs`(hf_config.json with yaml file) will override the values in `all_parameters_kwargs`(config init).
            merge_kwargs = {**all_parameters_kwargs, **kwargs}

            # Remove ignored parameters from input
            for param_name in ignore_param_names:
                merge_kwargs.pop(param_name, None)

            # Get current class name
            class_name = self_instance.__class__.__name__

            # Check if already printed for this class
            global PRINTED_CLASSES
            need_print = class_name not in PRINTED_CLASSES
            if need_print:
                PRINTED_CLASSES.add(class_name)

                # Print table of unsupported parameters
                logger.warning(f"Found unsupported HuggingFace arguments in {class_name}:")

                # Calculate column widths
                max_key_len = max(len(str(item[0])) for item in ignore_info) + 2
                max_val_len = max(len(str(item[1])) for item in ignore_info) + 2

                # Create table border
                border = f"+{'-' * (max_key_len + 2)}+{'-' * (max_val_len + 2)}+"

                # Print table header
                logger.warning(border)
                logger.warning(f"| {'Argument'.ljust(max_key_len)} | {'Status-Info'.ljust(max_val_len)} |")
                logger.warning(f"|:{'-' * (max_key_len + 1)}|:{'-' * (max_val_len + 1)}|")

                # Print parameter rows
                for arg, value in ignore_info:
                    logger.warning(f"| {str(arg).ljust(max_key_len)} | {str(value).ljust(max_val_len)} |")

                logger.warning(border)

            # Call original initialization method
            result = init_func(self_instance, *args, **merge_kwargs)

            # Remove attributes for ignored parameters
            for param_name in ignore_param_names:
                if hasattr(self_instance, param_name):
                    delattr(self_instance, param_name)

            return result

        return wrapper

    return decorator


def register_mf_model_parameter(mf_model_kwargs=None):
    """
    Decorator factory function for customizing kwargs in __init__ method.

    Args:
        mf_model_kwargs: Default keyword arguments that will be added to or override the original kwargs.

    Returns:
        The decorator function.
    """
    mf_model_kwargs = asdict(mf_model_kwargs) \
        if mf_model_kwargs is not None else asdict(MFModelConfig())

    def decorator(init_func):
        @wraps(init_func)
        def wrapper(self, *args, **kwargs):

            sig = inspect.signature(init_func)
            all_parameters_kwargs = dict()
            for name, param in sig.parameters.items():
                if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue
                if name in ('self', 'kwargs'):
                    continue
                all_parameters_kwargs.setdefault(name, param.default)
            # Merge default parameters with input parameters, giving priority to input parameters
            merged_kwargs = {**mf_model_kwargs, **all_parameters_kwargs, **kwargs}
            return init_func(self, *args, **merged_kwargs)

        return wrapper

    return decorator
