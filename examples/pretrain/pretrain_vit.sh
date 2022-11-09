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
echo "bash examples/pretrain/pretrain_vit.sh DEVICE_ID EPOCH_SIZE DATA_DIR"
echo "for example: bash examples/pretrain/pretrain_vit.sh 0 300 /path/dataset"
echo "=============================================================================================================="

DEVICE_ID=$1
EPOCH_SIZE=$2
DATA_DIR=$3

python -m mindtransformer.models.vit.vit_trainer \
      --device_id=$DEVICE_ID  \
      --epoch_size=$EPOCH_SIZE \
      --train_data_path=$DATA_DIR \
      --mode=0 \
      --dataset_name=imagenet \
      --enable_graph_kernel=True \
      --recompute=False \
      --parallel_mode="stand_alone" \
      --optimizer="adamw" \
      --weight_decay=0.05 \
      --init_loss_scale_value=65536 \
      --loss_scale=1024 \
      --no_weight_decay_filter="beta,bias" \
      --gc_flag=0 \
      --lr_decay_mode='cosine' \
      --lr_max=0.00355 \
      --poly_power=2.0 \
      --warmup_epochs=40 \
      --global_batch_size=128 \
      --sink_size=100 > standalone_train_gpu_log.txt 2>&1 &
tail -f standalone_train_gpu_log.txt
