# Copyright 2023 Huawei Technologies Co., Ltd
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
"""
Note: functions.
"""
import re
from typing import Optional, List


def re_match_list(mstr, patterns: Optional[List[str]] = None):
    """string regular match"""
    if not mstr:
        return False
    if patterns is None:
        return False
    for pattern in patterns:
        if not isinstance(pattern, str):
            raise TypeError(f"List item '{pattern}' is not a string.")
        if re.match(pattern, mstr):
            return True
    return False
