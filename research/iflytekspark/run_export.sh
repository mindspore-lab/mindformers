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
if [ $# != 0 ]  && [ $# != 3 ]
then
  echo "Usage Help: bash run_export.sh For Single Devices"
  echo "Usage Help: bash run_export.sh [RANK_TABLE_FILE] [DEVICE_RANGE] [RANK_SIZE] For Multiple Devices"
  exit 1
fi

export GLOG_v=2
export MS_ENABLE_GE=1
export MS_GE_TRAIN=1
export MS_GE_ATOMIC_CLEAN_POLICY=1
export MS_ENABLE_FORMAT_MODE=1
export MS_ENABLE_REF_MODE=1
export MS_DEV_SIDE_EFFECT_LOAD_ELIM=3

if [ $# == 3 ]
then
  export PARALLEL=True
else
  export PARALLEL=False
fi

if [ ! -d "./mindir" ]; then
  mkdir mindir
fi

# According to the different machines, the yaml file that should be used is:
# Atlas 800 32G -> run_iflytekspark_13b_lite_infer_800_32G.yaml
# Atlas 800T A2 64G -> run_iflytekspark_13b_lite_infer_800T_A2_64G.yaml
export CONFIG_PATH=run_iflytekspark_13b_lite_infer_800T_A2_64G.yaml

# You can modify the following export parameters
batch_size=1      # batch_size
max_seq_len=32768 # seq_length
mindir_save_dir=./mindir/iflytekspark-bs${batch_size}-seq${max_seq_len}
rm -rf $mindir_save_dir ./akg_kernel_meta ./rank_* ./output

export PY_CMD="python run_iflytekspark.py \
               --config $CONFIG_PATH \
               --run_mode export \
               --mindir_save_dir $mindir_save_dir \
               --use_parallel $PARALLEL \
               --load_checkpoint '{your_ckpt_path}' \
               --predict_length $max_seq_len \
               --predict_batch $batch_size"

if [ $# == 3 ]
then
    export RANK_TABLE_FILE=$1
    DEVICE_RANGE=$2
    export RANK_SIZE=$3

    DEVICE_RANGE_LEN=${#DEVICE_RANGE}
    DEVICE_RANGE=${DEVICE_RANGE:1:DEVICE_RANGE_LEN-2}
    PREFIX=${DEVICE_RANGE%%","*}
    INDEX=${#PREFIX}
    START_DEVICE=${DEVICE_RANGE:0:INDEX}
    END_DEVICE=${DEVICE_RANGE:INDEX+1:DEVICE_RANGE_LEN-INDEX}

    if [[ ! $START_DEVICE =~ ^[0-9]+$ ]]; then
        echo "error: start_device=$START_DEVICE is not a number"
    exit 1
    fi

    if [[ ! $END_DEVICE =~ ^[0-9]+$ ]]; then
        echo "error: end_device=$END_DEVICE is not a number"
    exit 1
    fi
fi

if [ ! -d "./log" ]; then
  mkdir log
fi

if [ $# == 3 ]
then
    for((i=${START_DEVICE}; i<${END_DEVICE}; i++))
    do
      export DEVICE_ID=$i
      export RANK_ID=$((i-START_DEVICE))
      echo "Start distribute export for rank $RANK_ID, device $DEVICE_ID"
      eval $PY_CMD &> log/export_$i.log &
    done
else
  eval $PY_CMD &> log/export.log
fi
