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
功能: 适配模块ModelArts场景常量
"""


# 共享资源池节点规格类型
INSTANCE_TYPE_SHARE = {
    1: 'modelarts.kat1.xlarge',
    2: 'modelarts.kat1.2xlarge',
    4: 'modelarts.kat1.4xlarge',
    8: 'modelarts.kat1.8xlarge'
}

# 专属资源池节点规格类型
TRAIN_INSTANCE_TYPE_SPEC = {
    1: 'modelarts.pool.visual.xlarge',
    2: 'modelarts.pool.visual.2xlarge',
    4: 'modelarts.pool.visual.4xlarge',
    8: 'modelarts.pool.visual.8xlarge'
}

DEFAULT_NODE_NUM = 1
DEFAULT_DEVICE_NUM = 8

MA_JOB_DIR = "${MA_JOB_DIR}"

DEFAULT_CACHE_DATA_PATH = "/cache/fmtk_data/"
DEFAULT_CACHE_OUTPUT_PATH = "/cache/fmtk_output/"
DEFAULT_CACHE_PRETRAINED_MODEL_PATH = "/cache/fmtk_pretrained_model/"
DEFAULT_CACHE_CKPT_PATH = "/cache/fmtk_ckpt/"
MA_LAUNCHER = "/home/ma-user/mxLaunchKit/ma/launcher.py"

JOB_STATUS = ['Completed', 'Terminated', 'Failed', 'Abnormal']
