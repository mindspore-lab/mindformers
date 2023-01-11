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
功能: common modelarts constants
"""

import os

from lk_utils import get_logger


MA_HOME = os.environ.get("MA_HOME", "/home/ma-user")
MA_MOUNT_PATH = os.environ.get("MA_MOUNT_PATH", "/home/ma-user/modelarts")
MA_JOB_DIR = os.environ.get("MA_JOB_DIR", "/home/ma-user/modelarts/user-job-dir")
MA_LOG_DIR = os.environ.get("MA_LOG_DIR", "/home/ma-user/modelarts/log")

# davinci tool entrypoint file
MA_DAVINCI_TOOL = os.path.join(MA_HOME, "ascend910", "run_ascend910.py")
# ascend log file folder
ASCEND_LOG_PATH = os.path.join(MA_LOG_DIR, "ascend/")
LOG_SYNC_PERIOD = 30


DEFAULT_CACHE_PATH = "/cache"
CACHE_MODEL_CONFIG_PATH = os.path.join(DEFAULT_CACHE_PATH, "model_config.yaml")
# for evaluation, inference
CACHE_CKPT_PATH = os.path.join(DEFAULT_CACHE_PATH, "fmtk_ckpt/")
# for fine-tuning
CACHE_PRETRAINED_MODEL_PATH = os.path.join(DEFAULT_CACHE_PATH, "fmtk_pretrained_model/")

MA_LOGGER = get_logger(name="ma_launcher")

MX_LOG_PATH = {
    "lk": os.path.join(MA_HOME, ".cache/Huawei/mxLaunchKit/log/"),
    "tk": os.path.join(MA_HOME, ".cache/Huawei/mxTuningKit/log/")
}

MODEL_TASK_NAME = os.getenv("MA_VJ_NAME", "default_model_task").replace('ma-job', 'modelarts-job', 1)
CURRENT_NODE_INDEX = os.getenv('VC_TASK_INDEX', os.getenv('VK_TASK_INDEX', "0"))
CURRENT_NODE_NAME = "node_" + CURRENT_NODE_INDEX

NODE_NUM = str(os.getenv("MA_NUM_HOSTS"))
