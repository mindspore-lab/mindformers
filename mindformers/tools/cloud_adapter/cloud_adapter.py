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
"""Cloud Adapter."""
import os
import time

from ..logger import logger
from ..utils import check_in_modelarts, \
    Validator

if check_in_modelarts():
    import moxing as mox


__all__ = ['mox_adapter']


def mox_adapter(src_dir, target_dir, retry=3, retry_time=5, log=logger):
    """File interaction with Moxing."""
    success = False
    for i in range(retry + 1):
        start = time.time()
        try:
            mox.file.copy_parallel(src_url=src_dir, dst_url=target_dir, threads=0, is_processing=False)
        except (FileNotFoundError, RuntimeError) as e:
            log.info("%s, from %s download to %s failed, will retry(%d) again.",
                     e, src_dir, target_dir, i)
            # sleep due to restriction of obs
            log.info("sleep time %d for waiting download file from obs.", retry_time)
            continue
        end = time.time()
        if Validator.is_obs_url(target_dir):
            if mox.file.exists(target_dir):
                success = True
                log.info("Pull/Push file %s success, cost time: %f", target_dir, end - start)
                break
        else:
            if os.path.exists(target_dir):
                success = True
                log.info("Pull/Push file %s success, cost time: %f", target_dir, end - start)
                break
    return success
