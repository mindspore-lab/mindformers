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
echo "bash scripts/run_classifier_gpu.sh DEVICE_ID"
echo "DEVICE_ID is optional, default value is zero"
echo "for example: bash scripts/run_classifier_gpu.sh 0 MRPC"
echo "assessment_method include: [MCC, Spearman_correlation ,Accuracy]"
echo "=============================================================================================================="

if [ -z $1 ]
then
    export CUDA_VISIBLE_DEVICES=0
else
    export CUDA_VISIBLE_DEVICES="$1"
fi

 mkdir -p ms_log
CUR_DIR=`pwd`
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0

TASK=$2
python -m transformer.trainer.trainer  \
    --auto_model="bert_glue" \
    --device_target="GPU" \
    --dataset_format="tfrecord" \
    --assessment_method="accuracy" \
    --parallel_mode="stand_alone" \
    --epoch_size=3 \
    --num_labels=2 \
    --vocab_size=30522 \
    --embedding_size=768 \
    --num_layers=12 \
    --num_heads=12 \
    --seq_length=128 \
    --use_one_hot_embeddings=False \
    --model_type="bert" \
    --dropout_prob=0.1 \
    --train_data_shuffle="true" \
    --train_batch_size=32 \
    --start_lr=5e-5 \
    --save_checkpoint_path="./glue_ckpt/$TASK" \
    --load_checkpoint_path="./checkpoint/bertbase.ckpt" \
    --checkpoint_prefix='$TASK' \
    --train_data_path="./glue_data/$TASK/train.tf_record"

python transformer.tasks.text_classification \
    --auto_model="bert_glue" \
    --device_target="GPU" \
    --dataset_format="tfrecord" \
    --assessment_method="accuracy" \
    --parallel_mode="stand_alone" \
    --epoch_num=3 \
    --num_labels=2 \
    --vocab_size=30522 \
    --embedding_size=768 \
    --num_layers=12 \
    --num_heads=12 \
    --seq_length=128 \
    --use_one_hot_embeddings=False \
    --checkpoint_prefix='$TASK' \
    --model_type="bert" \
    --dropout_prob=0.1 \
    --eval_data_shuffle="false" \
    --eval_batch_size=1 \
    --load_checkpoint_path="./glue_ckpt/$TASK" \
    --eval_data_path="./glue_data/$TASK/eval.tf_record"




