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

RANK_SIZE=$1
HOSTFILE=$2
TASK=$3
export NCCL_IB_HCA=mlx5_
mpirun --allow-run-as-root -n $RANK_SIZE --hostfile $HOSTFILE \
      --output-filename run_classifier \
      -x NCCL_IB_HCA -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x NCCL_SOCKET_IFNAME -n $RANK_SIZE \
      --mca btl tcp,self --mca btl_tcp_if_include 10.90.43.0/24,enp177s0f0 --merge-stderr-to-stdout \
python -m transformer.trainer.trainer  \
    --auto_model="bert_glue" \
    --device_target="GPU" \
    --device_num=$RANK_SIZE \
    --dataset_format="tfrecord" \
    --assessment_method="accuracy" \
    --parallel_mode="data_parallel" \
    --data_parallel=8 \
    --model_parallel=1 \
    --full_batch=False \
    --epoch_size=3 \
    --num_labels=2 \
    --vocab_size=30522 \
    --embedding_size=768 \
    --max_position_embeddings=512 \
    --num_layers=12 \
    --num_heads=12 \
    --seq_length=128 \
    --use_one_hot_embeddings=False \
    --model_type="bert" \
    --dropout_prob=0.1 \
    --train_data_shuffle="true" \
    --global_batch_size=4 \
    --start_lr=5e-5 \
    --save_checkpoint_path="./glue_ckpt/$TASK" \
    --load_checkpoint_path="/checkpoint/bertbase.ckpt" \
    --checkpoint_prefix="$TASK" \
    --train_data_path="/glue_path/$TASK/train.tf_record" \

python -m transformer.tasks.text_classification \
    --auto_model="bert_glue" \
    --device_target="GPU" \
    --dataset_format="tfrecord" \
    --assessment_method="accuracy" \
    --parallel_mode="stand_alone" \
    --num_labels=2 \
    --vocab_size=30522 \
    --embedding_size=768 \
    --num_layers=12 \
    --num_heads=12 \
    --seq_length=128 \
    --max_position_embeddings=512 \
    --use_one_hot_embeddings=False \
    --model_type="bert" \
    --dropout_prob=0.0 \
    --eval_data_shuffle="false" \
    --eval_batch_size=16 \
    --load_checkpoint_path="./glue_ckpt/$TASK" \
    --checkpoint_prefix="$TASK" \
    --eval_data_path="/glue_path/$TASK/eval.tf_record" 




