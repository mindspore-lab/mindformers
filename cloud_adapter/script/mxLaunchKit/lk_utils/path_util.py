# Copyright 2022 Huawei Technologies Co., Ltd
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
功能: common path utils of launcher
"""

import re

FOLDER_PATTERN = r"^/(\.?[-\w]+/)*([-\w]+)/$"
# file must have a suffix
FILE_PATTERN = r"^/([-\w]+/)*([-\w]+\.(\w+))$"


class PathPair(object):
    """
        Path mapping relationship
    """

    def __init__(self, src_path, dst_path):
        self.src_path = src_path
        self.dst_path = dst_path

    def is_same(self):
        return self.src_path == self.dst_path


def is_valid_path(path, is_folder):
    if is_folder:
        if not re.match(FOLDER_PATTERN, path):
            return False
    else:
        if not re.match(FILE_PATTERN, path):
            return False
    return True
