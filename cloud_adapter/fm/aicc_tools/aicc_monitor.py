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

from fm.aicc_tools.utils import mox_adapter
from fm.aicc_tools.utils.utils import SERVICE_LOG_PATH
from fm.aicc_tools.ailog.log import service_logger, service_logger_without_std, operation_logger


def aicc_monitor(run_func):
    """AICC monitor for main function."""

    def wrapper(*args, **kwargs):
        local_id = int(os.getenv('RANK_ID', '0'))
        result = None
        if len(args) == 0:
            arg = 'config'
        else:
            arg = args[0]
        operation_logger.info(f'{arg} task starts')
        try:
            result = run_func(*args, **kwargs)
            if arg in ['publish', 'deploy']:
                operation_logger.info(f'{arg} task success, the result is : {result}')
            elif arg in ['finetune', 'infer', 'evaluate']:
                operation_logger.info(f'{arg} task has been launched, the result is : {result}')
            elif arg != 'show':
                operation_logger.info(f'{arg} task done, the result is : {result}')
        except Exception as e:
            if str(e):
                service_logger.error(e)
            operation_logger.error(f'{arg} failed. An exception occurred.')
            if local_id % 8 == 0:
                raise e
        finally:
            _last_transform(local_id, service_logger_without_std)
        return result

    return wrapper


def _last_transform(local_id, log=service_logger_without_std):
    """Transform file when progress ending or except."""
    if local_id == 0 and os.environ.get('OBS_PATH'):
        target_dir = os.environ.get('OBS_PATH')
        if os.path.exists(SERVICE_LOG_PATH):
            mox_adapter(src_dir=SERVICE_LOG_PATH, target_dir=target_dir, log=log)


def upload_log():
    """Upload log according to runtime for fm."""
    local_id = int(os.getenv('RANK_ID', '0'))
    _last_transform(local_id)
