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
echo "bash examples/run_language_model.sh"
echo "for example: bash examples/run_language_model_distributed.sh 8 hostfile"
echo "metric method: PPL"
echo "eval_type include: [zero-shot, finetuned]. Default: zero-shot"
echo "=============================================================================================================="

CUR_DIR=`pwd`

RANK_SIZE=$1
HOSTFILE=$2


# checkpoint path
save_finetune_ckpt_path="/fine_ckpt/"
load_pretrain_ckpt_path="/checkpoint_path/gpt2.ckpt"

# dataset path
train_data_path="./wikitext-2/train/train-mindrecord"

export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0
export NCCL_IB_HCA=mlx5_

mpirun --allow-run-as-root -n $RANK_SIZE --hostfile $HOSTFILE \
      --output-filename run_classifier \
      -x NCCL_IB_HCA -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x NCCL_SOCKET_IFNAME -n $RANK_SIZE \
      --mca btl tcp,self --mca btl_tcp_if_include 10.90.43.0/24,enp177s0f0 --merge-stderr-to-stdout \
python -m mindtransformer.models.gpt.gpt_lm_trainer  \
    --device_target="GPU" \
    --device_num=$RANK_SIZE \
    --metric_method="PPL" \
    --do_train="true" \
    --eval_type="finetuned" \
    --epoch_size=3 \
    --train_data_shuffle="true" \
    --eval_data_shuffle="false" \
    --optimizer="adam"  \
    --seq_length=1024 \
    --parallel_mode="data_parallel" \
    --full_batch=False \
    --data_parallel=8 \
    --model_parallel=1 \
    --global_batch_size=16 \
    --vocab_size=50257 \
    --hidden_size=768 \
    --num_layers=12 \
    --num_heads=12 \
    --start_lr=2e-4 \
    --save_checkpoint_path=$save_finetune_ckpt_path \
    --load_checkpoint_path=$load_pretrain_ckpt_path \
    --checkpoint_prefix='language_model' \
    --train_data_path=$train_data_path \
