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

"""Check rules"""

import yaml
from yaml.nodes import MappingNode

YAML_MAX_NESTING_DEPTH = 10


def _get_yaml_ast_depth(node, depth=0):
    """Recursively calculate the maximum nesting depth of yaml ast structures."""
    if isinstance(node, MappingNode):  # process dict
        return max(
            (_get_yaml_ast_depth(v, depth + 1) for _, v in node.value), default=depth
        )
    return depth


def check_yaml_depth_before_loading(yaml_str, max_depth=YAML_MAX_NESTING_DEPTH):
    """Check yaml depth before loading"""
    try:
        node = yaml.compose(yaml_str)  # parse yaml to ast
        if node is None:
            return  # null file has no question
        depth = _get_yaml_ast_depth(node)
        if depth > max_depth:
            raise ValueError(
                f"YAML nesting depth {depth} exceeds the maximum allowed value of {max_depth}"
            )
    except yaml.YAMLError as e:
        raise ValueError(f"YAML parse error: {e}") from e
