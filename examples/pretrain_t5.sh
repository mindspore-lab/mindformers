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
echo "bash scripts/pretrain_t5.sh DEVICE_ID EPOCH_SIZE DATA_DIR"
echo "for example: bash scripts/run_standalone_train_gpu.sh 0 40 /path/zh-wiki/"
echo "=============================================================================================================="

DEVICE_ID=$1
EPOCH_SIZE=$2
DATA_DIR=$3

python pretrain_t5.py  \
    --distribute="false" \
    --epoch_size=$EPOCH_SIZE \
    --device_id=$DEVICE_ID \
    --data_path=$DATA_DIR \
    --optimizer="adam" \
    --max_seq_length=16 \
    --max_decode_length=16 \
    --max_position_embeddings=16 \
    --global_batch_size=96 \
    --vocab_size=36560 \
    --hidden_size=1024 \
    --num_hidden_layers=6 \
    --num_attention_heads=16 \
    --bucket_boundaries=16 \
    --device_target="GPU" > standalone_train_gpu_log.txt 2>&1 &
