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
echo "bash scripts/run_squad_gpu.sh"
echo "for example: bash scripts/run_squad_gpu.sh"
echo "assessment_method include: [Accuracy]"
echo "=============================================================================================================="
export GLOG_v=3

RANK_SIZE=$1
HOSTFILE=$2
export NCCL_IB_HCA=mlx5_

mpirun --allow-run-as-root -n $RANK_SIZE --hostfile $HOSTFILE \
      --output-filename run_classifier \
      -x NCCL_IB_HCA -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x NCCL_SOCKET_IFNAME -n $RANK_SIZE \
      --mca btl tcp,self --mca btl_tcp_if_include 10.90.43.0/24,enp177s0f0 --merge-stderr-to-stdout \
python -m  transformer.trainer.trainer.py  \
    --auto_model="bert" \
    --device_target="GPU" \
    --do_train="true" \
    --do_eval="true" \
    --parallel_mode="data_parallel" \
    --epoch_num=3 \
    --num_class=2 \
    --vocab_size=30522 \
    --embedding_size=1024 \
    --num_layers=24 \
    --num_heads=16 \
    --seq_length=384 \
    --max_position_embeddings=512 \
    --train_data_shuffle="true" \
    --eval_data_shuffle="false" \
    --data_parallel=8 \
    --model_parallel=1 \
    --train_batch_size=12 \
    --eval_batch_size=24 \
    --vocab_file_path="./vocab.txt" \
    --save_checkpoint_path="./squad_ckpt" \
    --train_data_path="./squad/train.mindrecord" \
    --eval_json_path="./squad/dev-v1.1.json" \
    --schema_file_path="" > squad_log.txt 2>&1 &