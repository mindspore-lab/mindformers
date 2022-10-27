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
echo "bash examples/pretrain/pretrain_bert_distributed.sh  DEVICE_NUM HOST_FILE DATA_DIR"
echo "for example: examples/pretrain/pretrain_bert_distributed.sh 8 hostfile /path/dataset"
echo "It is better to use absolute path."
echo "=============================================================================================================="

RANK_SIZE=$1
HOSTFILE=$2
DATASET=$3
export NCCL_IB_HCA=mlx5_

mpirun --allow-run-as-root -n $RANK_SIZE --hostfile $HOSTFILE \
      --output-filename run_distributed_train_gpt \
      -x NCCL_IB_HCA -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x NCCL_SOCKET_IFNAME -n $RANK_SIZE \
      --mca btl tcp,self --mca btl_tcp_if_include 10.90.43.0/24,enp177s0f0 --merge-stderr-to-stdout \
python -m transformer.models.bert.bert_trainer \
    --device_num=$RANK_SIZE \
    --train_data_path=$DATASET \
    --seq_length=128 \
    --global_batch_size=4 \
    --vocab_size=30522 \
    --parallel_mode="data_parallel" \
    --checkpoint_prefix="bert" \
    --full_batch=False \
    --hidden_size=768 \
    --num_layers=12 \
    --num_heads=12 \
    --data_parallel=8 \
    --model_parallel=1 \
    --device_target="GPU" > distribute_train_gpu_log.txt 2>&1 &

