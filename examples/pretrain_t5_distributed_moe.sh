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
echo "bash examples/pretrain_gpt_distributed.sh DATA_DIR RANK_TABLE_FILE DEVICE_NUM"
echo "for example: examples/pretrain_gpt_distributed.sh 8 hostfile /path/dataset"
echo "It is better to use absolute path."
echo "=============================================================================================================="

self_path=$(cd "$(dirname "$0")" || exit; pwd)
RANK_SIZE=$1
HOSTFILE=$2
DATASET=$3

mpirun --allow-run-as-root -n $RANK_SIZE --hostfile $HOSTFILE \
      --output-filename run_distributed_train_t5_moe \
      --mca btl tcp,self --mca btl_tcp_if_include 10.90.43.0/24,enp177s0f0 --merge-stderr-to-stdout \
python -s ${self_path}/../pretrain_t5.py  \
    --distribute="true" \
    --device_num=$RANK_SIZE \
    --data_path=$DATASET \
    --optimizer="adam" \
    --max_seq_length=16 \
    --max_decode_length=16 \
    --max_position_embeddings=16 \
    --global_batch_size=96 \
    --vocab_size=36560 \
    --hidden_size=1024 \
    --num_hidden_layers=6 \
    --num_attention_heads=16 \
    --parallel_mode="semi_auto_parallel" \
    --bucket_boundaries=16 \
    --expert_num=8 \
    --data_parallel=8 \
    --model_parallel=1 \
    --per_token_num_experts_chosen=4 \
    --device_target="GPU" > distribute_train_gpu_log.txt 2>&1 &

