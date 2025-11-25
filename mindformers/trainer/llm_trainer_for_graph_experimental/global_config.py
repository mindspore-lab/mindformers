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
"""Global configuration management module for MindFormer.

This module provides a global configuration management class that allows
recording and retrieving MindFormer configuration items from anywhere in
the codebase without needing to pass configuration parameters through
multiple function calls.
"""
from typing import Optional, Dict, Any

from mindformers.tools import MindFormerConfig


class MFGlobalConfig:
    """Global configuration management class for recording and retrieving MindFormer configuration items.

    This class provides globally accessible configuration management functionality, supporting
    adding records and retrieving configurations at any location to avoid passing various
    parameters to APIs.
    """

    global_config: Dict[str, Any] = {}

    def __init__(self) -> None:
        """Initialize MFGlobalConfig instance."""

    @classmethod
    def record_config_dict(cls, config: Optional[MindFormerConfig] = None, **kwargs) -> None:
        """
        Record configuration dictionary to global configuration.

        Args:
            config (MindFormerConfig, optional): MindFormer configuration object. Defaults to None.
            **kwargs: Other configuration items that will be updated to global configuration.
        """
        if config is not None:
            cls.global_config.update(**config)
        cls.global_config.update(**kwargs)

    @classmethod
    def get_config_value(cls, target_key: str, target_config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        Get the value of the specified key from global configuration.

        Args:
            target_key (str): Target key name to search for.
            target_config (dict, optional): Specific configuration dictionary to search.
                If None, searches the global configuration. Defaults to None.

        Returns:
            Any: Found configuration value, or None if not found.

        Notes:
            Supports recursive search in nested dictionaries and dictionary elements in lists.
        """
        search_config = cls.global_config if target_config is None else target_config

        if target_key in search_config:
            return search_config[target_key]

        # Traverse all values to find nested dictionaries
        for value in search_config.values():
            # If value is a dictionary type, recursively search
            if isinstance(value, dict):
                result = cls.get_config_value(
                    target_key, target_config=value)
                if result is not None:
                    return result
            # If value is a list type, check if each element is a dictionary
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        result = cls.get_config_value(
                            target_key, target_config=item)
                        if result is not None:
                            return result
        # Target key not found
        return None
