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
import pwd
import psutil

from fm.src.aicc_tools.ailog.log import service_logger
from fm.src.utils import constants


def concurrency_check(process_name):
    """
    并发控制
    """
    # 获取当前登录用户名
    current_login_user_name = pwd.getpwuid(os.geteuid()).pw_name

    pids_pool = psutil.pids()
    matched_process_count = 0

    for pid in pids_pool:
        if psutil.pid_exists(pid):
            _p = psutil.Process(pid)
            # 统计进程池快照中进程名为fm, 且属主是当前用户名的进程数量
            if _p.name() == process_name and _p.username() == current_login_user_name:
                matched_process_count += 1

    if matched_process_count > constants.CONCURRENCY_LIMIT_THRESHOLD:
        service_logger.error('detect concurrency attack risk, current request is rejected!')
        return False

    return True
