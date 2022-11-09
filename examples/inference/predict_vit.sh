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
echo "bash examples/inference/predict_vit.sh DEVICE_ID DATA_DIR MODEL_DIR"
echo "for example: bash examples/inference/predict_vit.sh 0 /path/dataset /path/ckpt"
echo "=============================================================================================================="

DEVICE_ID=$1
DATA_DIR=$2
MODEL_DIR=$3

# use stand-alone inference

python -m mindtransformer.models.vit.vit_trainer \
     --device_id=$DEVICE_ID  \
     --eval_data_path=$DATA_DIR \
     --load_checkpoint_path=$MODEL_DIR \
     --mode=0 \
     --is_training=False \
     --enable_graph_kernel=False \
     --generate=False \
     --dataset_name=imagenet \
     --get_eval_dataset=True \
     --eval_batch_size=128 \
     --parallel_mode="stand_alone" > standalone_inference_gpu_log.txt 2>&1 &
tail -f standalone_inference_gpu_log.txt