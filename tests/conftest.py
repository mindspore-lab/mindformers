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
"""The common pytest fixtures."""

import os

from pytest import fixture

from mindformers import logger

@fixture(scope="session", autouse=True)
def check_ascend_home_path():
    """Check if environment variable ASCEND_HOME_PATH in the CI machine."""

    ascend_home_path = os.getenv("ASCEND_HOME_PATH")
    logger.info("\n=============== Check ASCEND_HOME_PATH ENV ===============")
    if ascend_home_path is not None:
        logger.info("ASCEND_HOME_PATH: %s\n", ascend_home_path)
    else:
        logger.error(
            "ASCEND_HOME_PATH not found, please contact CI administrator!!!\n")
