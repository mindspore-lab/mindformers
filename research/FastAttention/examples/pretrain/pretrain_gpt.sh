#!/bin/bash
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash examples/pretrain/pretrain_gpt.sh  DEVICE_ID EPOCH_SIZE DATA_DIR"
echo "for example: bash examples/pretrain/pretrain_gpt.sh 0 40 /path/zh-wiki/"
echo "=============================================================================================================="
export GLOG_v=3
export DEVICE_ID=$1
EPOCH_SIZE=$2
DATA_DIR=$3

python -m research.FastFFN.transformer.core_gpt_trainer \
    --epoch_size=$EPOCH_SIZE \
    --train_data_path=$DATA_DIR \
    --optimizer="adam"  \
    --seq_length=1024 \
    --parallel_mode="stand_alone" \
    --checkpoint_prefix="gpt" \
    --global_batch_size=32 \
    --vocab_size=50257 \
    --hidden_size=768 \
    --num_layers=12 \
    --num_heads=12 \
    --full_batch=False \
    --device_target="GPU" > standalone_train_gpu_log.txt 2>&1 &
