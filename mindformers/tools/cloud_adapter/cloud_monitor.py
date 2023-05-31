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
"""Aicc monitor."""

import os
import traceback

from mindformers.tools.logger import get_logger
from .cloud_adapter import mox_adapter
from ..utils import DEBUG_INFO_PATH, PROFILE_INFO_PATH, PLOG_PATH,\
    get_output_root_path, get_remote_save_url


def cloud_monitor(log=get_logger()):
    """Aicc monitor for main function."""
    def decorator(run_func):

        def wrapper(*args, **kwargs):
            local_id = int(os.getenv('RANK_ID', '0'))
            try:
                result = run_func(*args, **kwargs)
            except BaseException as exc:  # pylint: disable=W0703
                error = traceback.format_exc()
                log.error(error)
                raise exc
            finally:
                _last_transform(local_id, log)
            return result
        return wrapper

    return decorator


def _last_transform(local_id, log=get_logger()):
    """Transform file when progress ending or except."""
    if os.environ.get("SPECIAL_ID") and get_remote_save_url():
        target_dir = get_remote_save_url()
        mox_adapter(src_dir=get_output_root_path(),
                    target_dir=get_remote_save_url(), log=log)
    else:
        if local_id % 8 == 0 and get_remote_save_url():
            target_dir = get_remote_save_url()
            mox_adapter(src_dir=get_output_root_path(), target_dir=target_dir, log=log)
    if local_id % 8 == 0 and get_remote_save_url():
        target_dir = get_remote_save_url()
        task_dir = os.path.join(target_dir, 'ascend-log')
        node = 'node_{}'.format(local_id)
        if os.path.exists(PLOG_PATH):
            mox_adapter(src_dir=PLOG_PATH, target_dir=os.path.join(task_dir, 'plog', node), log=log)
        if os.path.exists(DEBUG_INFO_PATH):
            mox_adapter(src_dir=DEBUG_INFO_PATH, target_dir=os.path.join(target_dir, 'debug_info', node), log=log)
        if os.path.exists(PROFILE_INFO_PATH):
            mox_adapter(src_dir=PROFILE_INFO_PATH, target_dir=os.path.join(target_dir, 'profile'), log=log)
        os.environ.setdefault('TRANSFORM_END_FLAG', '1')
    elif get_remote_save_url():
        log.info("Wait for the first card to complete the file and send it back to OBS: %s.",
                 get_remote_save_url())
        while True:
            end_flag = int(os.getenv("TRANSFORM_END_FLAG", "0"))
            if end_flag == 1:
                log.info("All files have been sent back to the OBS: %s,"
                         "and the process exits normally.", get_remote_save_url())
                break


def upload_log():
    """Upload log according to runtime for FM."""
    local_id = int(os.getenv('RANK_ID', '0'))
    _last_transform(local_id)
