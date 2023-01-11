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
""" Common executors of launcher. """

import os
import multiprocessing
from subprocess import Popen, PIPE, STDOUT

from lk_utils import get_logger


def execute_by_subprocess(execute_cmd, cwd=None):
    execute_env = os.environ.copy()
    execute_env["PYTHONUNBUFFERED"] = "1"
    process = Popen(execute_cmd, shell=False, cwd=cwd, stdout=PIPE, stderr=STDOUT, env=execute_env)
    subp_logger = get_logger(name="subp", msg_pattern="%(message)s", time_pattern=None, with_file_handler=False)
    # process block
    while process.poll() is None:
        subp_logger.info(str(process.stdout.readline().decode("utf-8")).rstrip())
    return_code = process.poll()
    rest_stdout = process.communicate()[0]
    subp_logger.info(str(rest_stdout.decode("utf-8")).rstrip())
    if return_code != 0:
        raise RuntimeError(f"Unable to execute command {execute_cmd}")


def execute_by_multiprocessing(func, args):
    process = multiprocessing.Process(target=func, args=args)
    process.start()
    return process
