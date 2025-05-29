#!/bin/bash

set -e

dataset_dir=$1
ckpt_dir=$2

# download dataset
dataset_url="https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/dataset/wikitext-2/wikitext-2-v1.zip"
dataset_name="wikitext-2"

train_dir="$dataset_dir/train"
valid_dir="$dataset_dir/valid"
test_dir="$dataset_dir/test"

wget --no-check-certificate $dataset_url -O $dataset_name.zip

rm -rf "$dataset_dir"
mkdir -p "$dataset_dir"

# unzip dataset
yes | unzip "$dataset_name".zip
rm -f "$dataset_name".zip

# preprocess dataset
echo "Preprocessing dataset..."
echo "Generating MindRecord for training data..."
python "../../../tools/dataset_preprocess/llama/llama_preprocess.py" \
  --dataset_type wiki \
  --input_glob "$dataset_name/wiki.train.tokens" \
  --model_file "$ckpt_dir/tokenizer.model" \
  --seq_length 4096 \
  --output_file "$train_dir/wiki4096.mindrecord"

echo "Generating MindRecord for evaluation data..."
python "../../../tools/dataset_preprocess/llama/llama_preprocess.py" \
  --dataset_type wiki \
  --input_glob "$dataset_name/wiki.valid.tokens" \
  --model_file "$ckpt_dir/tokenizer.model" \
  --seq_length 4096 \
  --output_file "$valid_dir/wiki4096.mindrecord"

echo "Generating MindRecord for test data..."
python "../../../tools/dataset_preprocess/llama/llama_preprocess.py" \
  --dataset_type wiki \
  --input_glob "$dataset_name/wiki.test.tokens" \
  --model_file "$ckpt_dir/tokenizer.model" \
  --seq_length 4096 \
  --output_file "$test_dir/wiki4096.mindrecord"

# clean up
rm -rf "$dataset_name"