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

model='glm32k'
mode='mslite'
output_path=eval_result/${model}_${mode}
gen_result_path=${output_path}/predict_result
merge_path=${output_path}/merge_result


python eval_postprocess.py \
--need_merge_path ${gen_result_path} \
--merged_path ${merge_path} \
--predict_file ${merge_path}/dureader.jsonl

echo 'evaluation completed!'