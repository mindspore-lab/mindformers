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
功能: entrypoint of model task in modelarts container
"""

import os
import argparse
import time

from execution import execute_by_multiprocessing
from lk_utils import PathPair, change_file_mod, prepare_common_path_parameters
from common_consts import ABNORMAL_EXIT_CODE, NORMAL_EXIT_CODE

from ma import FileTransporter
from ma.constants import MA_LOGGER, MX_LOG_PATH, LOG_SYNC_PERIOD, MODEL_TASK_NAME, CURRENT_NODE_INDEX, CURRENT_NODE_NAME
from ma.tasks import call_task_func

parser = argparse.ArgumentParser(description="ModelArts Launcher")
parser.add_argument("--task_type", type=str, help="type of model task")
parser.add_argument("--model_config_path", type=str, help="obs model config path")
parser.add_argument("--code_path", type=str, help="local code ptah of model task")
parser.add_argument("--boot_file_path", type=str, help="local boot file path of model task")
parser.add_argument("--data_path", type=str, help="local data(input) path model task")
parser.add_argument("--output_path", type=str, help="local output path model task")
parser.add_argument("--use_sfs", type=str, help="data from sfs or obs")
parser.add_argument("--ckpt_path", type=str, help="local model path of evaluate and infer")
parser.add_argument("--pretrained_model_path", type=str, help="local model path of finetune")
parser.add_argument("--log_path", type=str, help="obs log path of model_task")
parser.add_argument("--model_name", type=str, help="model name")
parser.add_argument("--device_type", type=str, help="device type npu")
parser.add_argument("--device_num", type=int, help="device num")


# There is a timezone issue of MoXing, sync in background is not available at present
def sync_fmtk_log(path_items, is_recursion=True):
    # path_item: (label, remote_path)
    for path_item in path_items:
        FileTransporter(path_pair=PathPair(src_path=MX_LOG_PATH.get(path_item[0]),
                                           dst_path=path_item[1]),
                        is_folder=True,
                        existence_detection=False,
                        info=f"{path_item[0]}_log").synchronize()
    while is_recursion:
        for path_item in path_items:
            FileTransporter(path_pair=PathPair(src_path=MX_LOG_PATH.get(path_item[0]),
                                               dst_path=path_item[1]),
                            is_folder=True,
                            existence_detection=False,
                            info=f"{path_item[0]}_log").synchronize()
        time.sleep(LOG_SYNC_PERIOD)


def perform_model_task(args):
    # handle common parameters
    try:
        path_params = prepare_common_path_parameters(args, MODEL_TASK_NAME, CURRENT_NODE_NAME)
    except ValueError as ve:
        MA_LOGGER.error(ve)
        MA_LOGGER.error("Failed to launch model task due to parameter missing.")
        return ABNORMAL_EXIT_CODE
    if args.boot_file_path:
        change_file_mod(args.boot_file_path)
    # node index for tk
    os.environ["SERVER_ID"] = CURRENT_NODE_INDEX
    # cwd for run_ascend910
    os.environ["MODEL_TASK_CWD"] = args.code_path

    is_success = call_task_func(args, path_params)
    if not is_success:
        return ABNORMAL_EXIT_CODE
    return NORMAL_EXIT_CODE


if __name__ == '__main__':
    arguments = parser.parse_known_args()[0]

    # synchronize mxLaunchKit and mxTuningKit log folder at background
    obs_tk_log_path, obs_lk_log_path, log_sync_process = None, None, None
    if arguments.log_path:
        # run_ascend will change the ma to modelarts when creating log dir
        obs_tk_log_path = os.path.join(arguments.log_path, MODEL_TASK_NAME, "mxTuningKit" + os.sep)
        obs_lk_log_path = os.path.join(arguments.log_path, MODEL_TASK_NAME, "mxLaunchKit", CURRENT_NODE_NAME + os.sep)
        log_sync_process = execute_by_multiprocessing(func=sync_fmtk_log,
                                                      args=([("tk", obs_tk_log_path), ("lk", obs_lk_log_path)], True))

    try:
        ret_code = perform_model_task(arguments)
    except Exception as e:
        MA_LOGGER.error(e)
        ret_code = ABNORMAL_EXIT_CODE

    # stop tk log synchronization
    if log_sync_process:
        if not log_sync_process.is_alive():
            MA_LOGGER.warning("log synchronization process is abnormal.")
        log_sync_process.terminate()
        sync_fmtk_log([("tk", obs_tk_log_path), ("lk", obs_lk_log_path)], False)

    raise SystemExit(ret_code)
