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
echo "bash scripts/run_classifier_gpu.sh RANK_SIZE HOSTFILE TASK "
echo "DEVICE_ID is optional, default value is zero"
echo "for example: bash scripts/run_classifier_gpu.sh 8 hostfile MRPC"
echo "assessment_method include: [MCC, Spearman_correlation ,Accuracy]"
echo "=============================================================================================================="

export GLOG_v=3

RANK_SIZE=$1
HOSTFILE=$2
TASK=$3
export NCCL_IB_HCA=mlx5_

mpirun --allow-run-as-root -n $RANK_SIZE --hostfile $HOSTFILE \
      --output-filename run_classifier \
      -x NCCL_IB_HCA -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x NCCL_SOCKET_IFNAME -n $RANK_SIZE \
      --mca btl tcp,self --mca btl_tcp_if_include 10.90.43.0/24,enp177s0f0 --merge-stderr-to-stdout \
python -m tasks.nlp.text_classification.run_classifier  \
    --config="./transformer/configs/nezha/task_classifier_config.yaml" \
    --device_target="GPU" \
    --device_num=$RANK_SIZE \
    --do_train="true" \
    --do_eval="true" \
    --assessment_method="accuracy" \
    --parallel_mode="data_parallel" \
    --epoch_num=3 \
    --num_class=16 \
    --vocab_size=21128 \
    --embedding_size=768 \
    --num_layers=12 \
    --num_heads=12 \
    --seq_length=128 \
    --train_data_shuffle="true" \
    --eval_data_shuffle="false" \
    --data_parallel=8 \
    --model_parallel=1 \
    --compute_dtype=fp16 \
    --train_batch_size=16 \
    --eval_batch_size=16 \
    --start_lr=6e-5 \
    --save_finetune_checkpoint_path="./glue_ckpt/$TASK" \
    --load_pretrain_checkpoint_path="./checkpoint/nezhabase.ckpt" \
    --load_finetune_checkpoint_path="./glue_ckpt/$TASK" \
    --train_data_path="./glue_data/$TASK/train.tf_record" \
    --eval_data_path="./glue_data/$TASK/eval.tf_record" \
    --schema_file_path="" > $TASK.txt 2>&1 &
