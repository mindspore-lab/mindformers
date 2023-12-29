#!/bin/bash
# Copyright 2023 Huawei Technologies Co., Ltd
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

export RANK_TABLE_FILE=$1
CHECKPOINT_PATH=$2

# define variable
export RANK_SIZE=$3
export START_RANK=$4 # this server start rank
let END_RANK=START_RANK+RANK_SIZE # this server end rank

mkdir "output"

# run
for((i=${START_RANK}; i<${END_RANK}; i++))
do
    export RANK_ID=$((i-START_RANK))
    export DEVICE_ID=$i
    echo "Start distribute running for rank $RANK_ID, device $DEVICE_ID"
    python3 ./infer_pipeline.py --checkpoint_path $CHECKPOINT_PATH &> output/mindformers_$RANK_ID.log &
done
