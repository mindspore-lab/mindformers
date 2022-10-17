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
echo "bash scripts/run_language_model.sh"
echo "for example: bash scripts/run_language_model.sh"
echo "metric method: PPL"
echo "eval_type include: [zero-shot, finetuned]. Default: zero-shot"
echo "=============================================================================================================="

CUR_DIR=`pwd`
# checkpoint path
save_finetune_ckpt_path="./fine_ckpt/"
load_pretrain_ckpt_path="./pretrain_ckpt/gpt2.ckpt"
load_eval_ckpt_path="./fine_ckpt/"

# dataset path
train_data_file_path="./wikitext-2/train/train-mindrecord"
eval_data_file_path="./wikitext-2/test/test-mindrecord"

PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0
python -m tasks.nlp.language_modeling.run_language_model  \
    --config="transformer/configs/gpt/language_model.yaml" \
    --device_target="GPU" \
    --device_id=0 \
    --metric_method="PPL" \
    --do_train="true" \
    --do_eval="true" \
    --eval_type="finetuned" \
    --epoch_num=3 \
    --train_data_shuffle="true" \
    --eval_data_shuffle="false" \
    --optimizer="adam"  \
    --seq_length=1024 \
    --parallel_mode="stand_alone" \
    --train_batch_size=8 \
    --eval_batch_size=8 \
    --vocab_size=50257 \
    --hidden_size=768 \
    --num_layers=12 \
    --num_heads=12 \
    --start_lr=2e-4 \
    --save_finetune_ckpt_path=$save_finetune_ckpt_path \
    --load_pretrain_ckpt_path=$load_pretrain_ckpt_path \
    --load_finetune_ckpt_path=$load_eval_ckpt_path \
    --train_data_file_path=$train_data_file_path \
    --eval_data_file_path=$eval_data_file_path > language_log.txt 2>&1 &
