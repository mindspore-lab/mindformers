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
import os
import time
import logging as logger

from fm.aicc_tools.utils.validator import check_in_modelarts, Validator

if check_in_modelarts():
    import moxing as mox


class Obs2Local:
    """File saved from OBS server to local system of AI computing center platform.

    Args:
        rank_id (int): The obs's contents will be upload according to the actual rank_id value.
            Default: 0, means stored only one OBS file each node.
        retry (int): The number of attempts to save again if the first attempt fails.
            Default: 3, will be try three times.
        retry_time: The time of resaving the previously dormant program, after each attempt fails.
            Default: 5, will sleep five seconds.
        log (logger): Use the log system to print information.
            Default: logging class for Python.
    """

    def __init__(self, rank_id=0, retry=3, retry_time=5, log=logger):
        self.rank_id = int(rank_id)
        self.retry_time = retry_time
        self.retry = retry
        self.log = log

    def obs2local(self, obs_url, local_url):
        obs_name = obs_url.split("/")[-1]
        mox_lock = os.path.join(local_url, "mox_copy_{}.lock".format(obs_name))
        local_url = os.path.join(local_url, obs_name)
        if self.rank_id % 8 == 0:
            mox_adapter(obs_url, local_url, self.retry, self.retry_time, self.log)
            try:
                os.mknod(mox_lock)
            except IOError:
                pass
        else:
            self.log.info("programming sleep for waiting download file from obs to local.")
            while True:
                if os.path.exists(mox_lock):
                    break
                time.sleep(1)
        return local_url


def mox_adapter(src_dir, target_dir, retry=3, retry_time=5, log=logger):
    """File interaction with Moxing."""
    success = False
    for i in range(retry + 1):
        start = time.time()
        try:
            mox.file.copy_parallel(src_url=src_dir, dst_url=target_dir)
        except RuntimeError as e:
            log.error(f"{e}, from {src_dir} download to {target_dir} failed, will retry({i}) again.")
            # sleep due to restriction of obs
            log.error(f"sleep time {retry_time} for waiting download file from obs.")
            time.sleep(retry_time)
            continue

        end = time.time()
        if Validator.is_obs_url(target_dir):
            if mox.file.exists(target_dir):
                success = True
                log.info(f"Pull/Push file {target_dir} success, cost time: {end - start}")
                break
        else:
            if os.path.exists(target_dir):
                success = True
                log.info(f"Pull/Push file {target_dir} success, cost time: {end - start}")
                break
    return success


def obs_register(**kwargs):
    """OBS register with Moxing."""
    if check_in_modelarts():
        os.environ.pop('CREDENTIAL_PROFILES_FILE', None)
        os.environ.pop('AWS_SHARED_CREDENTIALS_FILE', None)
        # mox.file.set_auth(**kwargs)
