#!/bin/bash
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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_distribute_train_ascend.sh DATA_DIR RANK_TABLE_FILE DEVICE_NUM"
echo "for example: bash scripts/run_distribute_train_ascend.sh /path/dataset /path/hccl.json 8"
echo "It is better to use absolute path."
echo "=============================================================================================================="

ROOT_PATH=`pwd`
DATA_DIR=$1
export RANK_TABLE_FILE=$2
RANK_SIZE=$3


for((i=0;i<${RANK_SIZE};i++));
do
    rm ${ROOT_PATH}/device$i/ -rf
    mkdir ${ROOT_PATH}/device$i
    cd ${ROOT_PATH}/device$i || exit
    export RANK_ID=$i
    export DEVICE_ID=$i
    python ${ROOT_PATH}/pretrain_gpt.py --distribute="true" --device_num=$RANK_SIZE --data_path=$DATA_DIR \
        --optimizer="adam" \
        --max_seq_length=1024 \
        --global_batch_size=8 \
        --vocab_size=50304 \
        --hidden_size=2048 \
        --num_hidden_layers=24 \
        --parallel_mode="semi_auto_parallel" \
        --num_attention_heads=16 \
        --data_parallel=8 \
        --model_parallel=1 \
        --device_target="Ascend">log$i.log 2>&1 &
done