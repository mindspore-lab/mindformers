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
echo "bash examples/pretrain/pretrain_vit.sh DEVICE_ID"
echo "for example: bash examples/pretrain/pretrain_vit.sh 0"
echo "=============================================================================================================="

export DEVICE_ID=$1

python -m mindtransformer.models.vit.vit_trainer \
       --dataset_name=imagenet \
       --train_data_path='/ms_test1/mindspore_dataset/ImageNet2012/train/' \
       --lr_decay_mode='cosine' \
       --lr_max=0.00355 \
       --epoch_size=300 \
       --warmup_epochs=40 \
       --sink_size=10 \
       --device_target="GPU" \
       --device_id=$DEVICE_ID > standalone_train_gpu_log.txt 2>&1 &
tail -f standalone_train_gpu_log.txt
