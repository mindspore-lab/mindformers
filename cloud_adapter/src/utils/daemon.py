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
import atexit
import os
import psutil
import setproctitle
import signal
import sys
import time

from fm.src.aicc_tools.aicc_monitor import upload_log
from fm.src.aicc_tools.utils import check_in_modelarts
from fm.src.utils.io_utils import DEFAULT_FLAGS, DEFAULT_MODES, \
    wrap_local_working_directory, is_link, get_config_dir_setting

PID_FILE = wrap_local_working_directory(file_name='netkiller.pid', specific_path_config=get_config_dir_setting())
LOOP = True
JOB = True
DAEMON_NAME = 'fm-daemon'
SLEEP_TIME = 300


def save_pid(pid):
    """save pid to pidfile"""
    if not is_link(PID_FILE):
        with os.fdopen(os.open(PID_FILE, DEFAULT_FLAGS, DEFAULT_MODES), 'w') as f:
            f.write(pid)
    atexit.register(stop)


def get_pid():
    """load pid from pidfile"""
    pid = 0
    if is_link(PID_FILE):
        return pid
    try:
        with os.fdopen(os.open(PID_FILE, DEFAULT_FLAGS, DEFAULT_MODES), 'r') as f:
            pid = int(f.readline())
    except FileNotFoundError:
        pass
    return pid


def signal_handle(signum):
    """process signal"""
    global LOOP, JOB
    if signum == signal.SIGHUP:
        """reload"""
        JOB = False
    elif signum == signal.SIGINT:
        """quit"""
        LOOP = False
        JOB = False


def daemonize():
    """run the daemon process"""
    global JOB
    signal.signal(signal.SIGHUP, signal_handle)
    signal.signal(signal.SIGINT, signal_handle)
    signal.alarm(5)
    pid = os.fork()
    sys.stdout.flush()
    sys.stderr.flush()
    if pid:
        return
    # change name of daemon process, avoid blocking
    setproctitle.setproctitle(DAEMON_NAME)
    save_pid(str(os.getpid()))
    while LOOP:
        while JOB:
            main()
        JOB = True


def upload():
    """upload log"""
    if not check_in_modelarts():
        return
    if os.path.isfile(PID_FILE):
        pid = get_pid()
        if psutil.pid_exists(pid):
            return
        else:
            # if pid_file exists and bug process is not running, remove the file and restart son process
            os.remove(PID_FILE)
    daemonize()


def stop():
    """kill the son process """
    pid = get_pid()
    if pid != 0:
        try:
            os.kill(pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        os.remove(PID_FILE)


def reloads():
    """reload the son process """
    pid = get_pid()
    if pid != 0:
        try:
            os.kill(pid, signal.SIGHUP)
        except ProcessLookupError:
            pass


def main():
    """ main function,do what you want """
    time.sleep(SLEEP_TIME)
    upload_log()
