#!/bin/bash
# Copyright 2024 Huawei Technologies Co., Ltd
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
set -e
BASE_PATH=$(cd "$(dirname $0)"; pwd)
USE_DEVICE_NUM=$1
TEST_MODE=$2

export MS_ENABLE_FORMAT_MODE=1
export MS_GE_TRAIN=1
export MS_ENABLE_REF_MODE=1
export MS_ENABLE_GE=1
export MS_DEV_CELL_REUSE=1

export MS_MEMORY_POOL_RECYCLE=1

if [ "$TEST_MODE" == "test_train" ]
then
  msrun --worker_num=${USE_DEVICE_NUM} \
   --local_worker_num=${USE_DEVICE_NUM} \
   --master_port=8118 \
   --log_dir=msrun_log \
   --join=True \
   --cluster_time_out=300 \
   ${BASE_PATH}/parallel_mixtral.py --test_train True > parallel_mixtral_train.log 2>&1
elif [ "$TEST_MODE" == "test_predict" ]
then
  msrun --worker_num=${USE_DEVICE_NUM} \
   --local_worker_num=${USE_DEVICE_NUM} \
   --master_port=8118 \
   --log_dir=msrun_log \
   --join=True \
   --cluster_time_out=300 \
   ${BASE_PATH}/parallel_mixtral.py --test_predict True > parallel_mixtral_predict.log 2>&1
fi
