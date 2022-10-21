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
echo "bash examples/pretrain/pretrain_vit_distributed_gpu.sh  DEVICE_NUM HOST_FILE"
echo "for example: examples/pretrain/pretrain_vit_distributed_gpu.sh 8 hostfile"
echo "It is better to use absolute path."
echo "=============================================================================================================="

RANK_SIZE=$1
HOSTFILE=$2
export NCCL_IB_HCA=mlx5_

mpirun --allow-run-as-root -n $RANK_SIZE --hostfile $HOSTFILE \
      --output-filename run_distributed_train_vit \
      -x NCCL_IB_HCA -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x NCCL_SOCKET_IFNAME -n $RANK_SIZE \
      --mca btl tcp,self --mca btl_tcp_if_include 10.90.43.0/24,enp177s0f0 --merge-stderr-to-stdout \
python -m transformer.models.vit.vit_trainer \
       --device_num=$RANK_SIZE \
       --dataset_name=imagenet \
       --train_data_path='/ms_test1/mindspore_dataset/ImageNet2012/train/' \
       --parallel_mode="data_parallel" \
       --data_parallel=8 \
       --model_parallel=1 \
       --lr_decay_mode='cosine' \
       --lr_max=0.00355 \
       --epoch_size=300 \
       --warmup_epochs=40 \
       --sink_size=100 \
       --device_target="GPU" > distribute_train_gpu_log.txt 2>&1 &

