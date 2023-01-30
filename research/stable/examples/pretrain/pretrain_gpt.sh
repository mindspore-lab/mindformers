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
echo "bash examples/pretrain/pretrain_gpt_distributed.sh DATA_DIR RANK_TABLE_FILE DEVICE_NUM"
echo "for example: examples/pretrain/pretrain_gpt_distributed.sh 8 hostfile /path/dataset"
echo "It is better to use absolute path."
echo "=============================================================================================================="
# export GLOG_v=3
export GLOG_v=3
export NCCL_DEBUG=INFO
RANK_SIZE=$1
HOSTFILE=$2
DATASET=$3

mpirun --allow-run-as-root -n $RANK_SIZE \
      --output-filename run_distributed_train_gpt \
python -m research.ntlb.transformer.core_gpt_trainer  \
    --device_num=$RANK_SIZE \
    --train_data_path=$DATASET \
    --seq_length=1024 \
    --global_batch_size=8 \
    --vocab_size=50257 \
    --parallel_mode="semi_auto_parallel" \
    --full_batch=True \
    --checkpoint_prefix="gpt" \
    --routing_policy="stable" \
    --routing_stage="s1" \
    --hidden_size=768 \
    --recompute=True \
    --mp_comm_recompute=False \
    --num_layers=12 \
    --num_heads=12 \
    --data_parallel=2 \
    --model_parallel=1 \
    --expert_parallel=2 \
    --expert_num=2 \
    --per_token_num_experts_chosen=1 \
    --device_target="GPU" > 0128_ntlb_train_gpu_log.txt 2>&1 &