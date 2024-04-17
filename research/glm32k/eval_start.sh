#!/bin/bash
# Copyright 2023 Huawei Technologies Co., Ltd
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

start_time=$(date +%s)
echo "Start eval, current date is $start_time"

input_dataset_file=/path/to/eval_dataset/dureader.jsonl
checkpoint_path=/path/to/glm3_32k.ckpt
ge_config_path="/research/glm32k/910b_ge_prefill_pa.cfg,/research/glm32k/910b_ge_inc_pa.cfg"
tokenizer_path=//path/to/tokenizer.model
full_model_path=/research/output/mindir_full_checkpoint/rank_0_graph.mindir
inc_model_path=/research/output/mindir_inc_checkpoint/rank_0_graph.mindir


model='glm32k'
mode='mslite'
output_path=eval_result/${model}_${mode}
gen_result_path=${output_path}/predict_result
log_path=${output_path}/log
merge_path=${output_path}/merge_result

mkdir -p ${gen_result_path}
mkdir -p ${log_path}
mkdir -p ${merge_path}
echo 'Output path: 'output_path

# 200条测试数据，如果使用8卡，每张卡分配25条数据
npu_num=8
first_npu_id=0
step=25
for ((i = 0; i < ${npu_num}; i++)); do
  start_index=$((i * step))
  end_index=$(((i + 1) * step))
  npu=$((i + first_npu_id))
  echo 'Running process #' ${i} 'from' ${start_index} 'to' ${end_index} 'on NPU' ${npu}
  if [ ${mode} == "online" ]; then
    python eval_gen_online.py \
    --start_index ${start_index} \
    --end_index ${end_index} \
    --output_file ${gen_result_path} \
    --input_dataset_file ${input_dataset_file} \
    --device_id ${npu} \
    --checkpoint_path ${checkpoint_path} \
    &> ./${log_path}/longbench_${npu}.log &
  elif [ ${mode} == "mslite" ]; then
    python eval_gen_mslite.py \
    --start_index ${start_index} \
    --end_index ${end_index} \
    --output_file ${gen_result_path} \
    --input_dataset_file ${input_dataset_file} \
    --device_id ${npu} \
    --do_sample False \
    --batch_size 1 \
    --model_name glm3 \
    --tokenizer_path ${tokenizer_path} \
    --prefill_model_path ${full_model_path} \
    --increment_model_path ${inc_model_path} \
    --config_path ${ge_config_path} \
    --dynamic False \
    --paged_attention True \
    --pa_block_size 128 \
    --pa_num_blocks 512 \
    --seq_length 32640 \
    --max_length 32640 \
    &> ./${log_path}/longbench_${npu}.log &
  else
    echo "Unknown mode: $mode"
  fi
done
