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
echo "bash examples/pretrain/pretrain_vit_distributed_gpu.sh DEVICE_NUM HOST_FILE EPOCH_SIZE DATA_DIR"
echo "for example: examples/pretrain/pretrain_vit_distributed_gpu.sh 8 hostfile 300 /path/dataset"
echo "It is better to use absolute path."
echo "=============================================================================================================="

RANK_SIZE=$1
HOSTFILE=$2
EPOCH_SIZE=$3
DATA_DIR=$4
export NCCL_IB_HCA=mlx5_

mpirun --allow-run-as-root -n $RANK_SIZE --hostfile $HOSTFILE \
      --output-filename run_distributed_train_vit \
      -x NCCL_IB_HCA -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x NCCL_SOCKET_IFNAME -n $RANK_SIZE \
      --mca btl tcp,self --mca btl_tcp_if_include 10.90.43.0/24,enp177s0f0 --merge-stderr-to-stdout \
python -m mindtransformer.models.vit.vit_trainer \
      --device_num=$RANK_SIZE \
      --epoch_size=$EPOCH_SIZE \
      --train_data_path=$DATA_DIR \
      --mode=0 \
      --dataset_name=imagenet \
      --enable_graph_kernel=True \
      --recompute=False \
      --parallel_mode="semi_auto_parallel" \
      --data_parallel=8 \
      --model_parallel=1 \
      --optimizer="adamw" \
      --weight_decay=0.05 \
      --init_loss_scale_value=65536 \
      --loss_scale=1024 \
      --no_weight_decay_filter="beta,bias" \
      --gc_flag=0 \
      --lr_decay_mode="cosine" \
      --lr_max=0.00355 \
      --poly_power=2.0 \
      --warmup_epochs=40 \
      --global_batch_size=128 \
      --sink_size=1000 \
      --device_target="GPU" > distribute_train_gpu_log.txt 2>&1 &
tail -f distribute_train_gpu_log.txt

