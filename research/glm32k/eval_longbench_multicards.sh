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

input_dataset_file=/path/eval_dataset/dureader.jsonl
output_file=pred
checkpoint_path=/path/mindspore_models/glm32k.ckpt

mkdir -p ${output_file}
echo 'Output path: 'output_file

# 200条测试数据，如果使用8卡，每张卡分配25条数据
npu_num=8
step=25
for ((i = 0; i < $npu_num; i++)); do
  start_index=$((i * step))
  end_index=$(((i + 1) * step))
  npu=$((i))
  echo 'Running process #' ${i} 'from' $start_index 'to' $end_index 'on NPU' ${npu}
  python eval_longbench_generate.py --start_index ${start_index} --end_index ${end_index} --output_file ${output_file} --input_dataset_file ${input_dataset_file} --device_id ${npu} --checkpoint_path ${checkpoint_path} &> longbench_$npu.log &
 done