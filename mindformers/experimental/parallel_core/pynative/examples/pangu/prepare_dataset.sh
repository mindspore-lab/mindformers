#!/bin/bash

set -e

# download dataset
dataset_url="https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/dataset/wikitext-2/wikitext-2-v1.zip"
dataset_name="wikitext-2"

dataset_dir=$1
train_dir="$dataset_dir/train"
valid_dir="$dataset_dir/valid"
test_dir="$dataset_dir/test"

seq_length=$2

wget --no-check-certificate $dataset_url -O $dataset_name.zip

rm -rf "$dataset_dir"
mkdir -p "$dataset_dir"

# unzip dataset
unzip "$dataset_name".zip
rm -f "$dataset_name".zip

# preprocess dataset
echo "Preprocessing dataset..."
echo "Generating MindRecord for training data..."
python data_preprocess.py --input_glob  "$dataset_name/wiki.train.tokens" --dataset_type "wiki"  --output_file "$train_dir/data.mindrecord" --eot 50256 --data_column_name input_ids --seq_length "$seq_length"
echo "Generating MindRecord for evaluation data..."
python data_preprocess.py --input_glob  "$dataset_name/wiki.valid.tokens" --dataset_type "wiki"  --output_file "$valid_dir/data.mindrecord" --eot 50256 --data_column_name input_ids --seq_length "$seq_length"
echo "Generating MindRecord for test data..."
python data_preprocess.py --input_glob  "$dataset_name/wiki.test.tokens" --dataset_type "wiki"  --output_file "$test_dir/data.mindrecord" --eot 50256 --data_column_name input_ids --seq_length "$seq_length"
