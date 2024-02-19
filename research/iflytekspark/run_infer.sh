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
  echo "Usage Help: bash run_distribute.sh For Single Devices"
  echo "Usage Help: bash run_distribute.sh [RANK_TABLE_FILE] [DEVICE_RANGE] [RANK_SIZE] For Multiple Devices"
  exit 1
fi

if (lspci | grep d801);then
    export DEVICE_PLATFORM="Ascend910_32G"
elif (lspci | grep d802);then
    export DEVICE_PLATFORM="Ascend910_64G"
else
    echo "DEVICE_PLATFORM not supported"
    exit 1
fi

if [ $DEVICE_PLATFORM == "Ascend910_64G" ]
then
  export MS_DEV_SIDE_EFFECT_LOAD_ELIM=3
  export MS_ENABLE_GE=1
  export MS_GE_TRAIN=1
  export MS_GE_ATOMIC_CLEAN_POLICY=1
  export MS_ENABLE_FORMAT_MODE=1
  export MS_ENABLE_REF_MODE=1
fi

if [ $# == 3 ]
then
    export PARALLEL=True
else
    export PARALLEL=False
fi

# According to the different machines, the yaml file that should be used is:
# Atlas 800 32G -> run_iflytekspark_13b_infer_800_32G.yaml
# Atlas 800T A2 64G -> run_iflytekspark_13b_infer_800T_A2_64G.yaml
export CONFIG_PATH=run_iflytekspark_13b_infer_800T_A2_64G.yaml

# You can modify the following inference parameters
export PY_CMD="python run_iflytekspark.py \
               --config $CONFIG_PATH \
               --run_mode predict \
               --use_parallel $PARALLEL \
               --load_checkpoint '{your_ckpt_path}' \
               --predict_data '[为什么地球是独一无二的？##请问生抽和老抽有什么区别？]' \
               --predict_length 32768 \
               --predict_batch 1 \
               --prompt '<User> {}<end><Bot> ' \
               --tokenizer_file '{your_tokenizer_path}' \
               --streamer False"

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
        echo "Start distribute running for rank $RANK_ID, device $DEVICE_ID"
        eval $PY_CMD &> log/infer_$i.log &
    done
else
    eval $PY_CMD &> log/infer.log
fi
