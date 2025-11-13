
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
""" Process utils."""
from mindspore.communication import comm_func

from mindformers.tools.logger import logger
from mindformers.tools.utils import get_real_group_size


def barrier_world(action: str = None):
    """barrier all rank until action is done"""
    if get_real_group_size() > 1:
        if action is not None:
            logger.info("Wait " + str(action))
        else:
            logger.info("Now barriered...")

        comm_func.barrier()
