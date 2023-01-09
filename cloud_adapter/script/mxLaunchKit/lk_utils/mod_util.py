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
功能: common mod utils of launcher
"""

import os

# stat.S_IXUSR | stat.S_IWUSR | stat.S_IRUSR | stat.S_IXGRP | stat.S_IRGRP = 488
DEFAULT_PATH_MODE = 488


def change_folder_mod(path, mode=DEFAULT_PATH_MODE):
    os.chmod(path, mode)
    for root, folders, files in os.walk(path):
        for folder in folders:
            os.chmod(os.path.join(root, folder), mode)
        for f in files:
            os.chmod(os.path.join(root, f), mode)


def change_file_mod(path, mode=DEFAULT_PATH_MODE):
    os.chmod(path, mode)
