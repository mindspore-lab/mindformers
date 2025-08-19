#!/bin/bash
# Copyright 2025 Huawei Technologies Co., Ltd
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

# usage: bash run_osl_deepseek_4layers.sh <model_dir> [worker_num(default: 4)]
model_dir=$1
worker_num=${2:-4}
basename=$(cd "$(dirname $0)"; pwd)
mf_path=$basename/../../../..
port=8919

# Write yaml
sed -i "s|model_parallel: [0-9]*|model_parallel: $worker_num|" osl_deepseek.yaml
sed -i "s|pretrained_model_dir: '.*'|pretrained_model_dir: \'$model_dir\'|" osl_deepseek.yaml

# Run prediction
bash $mf_path/scripts/msrun_launcher.sh "$mf_path/run_mindformer.py \
    --config $basename/osl_deepseek.yaml \
    --run_mode predict \
    --use_parallel True \
    --predict_data '介绍下北京故宫'" \
    $worker_num $port output/msrun_log True 7200

# Check result
grep 介绍下北京故宫博物院OD text_generation_result.txt
if [ $? -ne 0 ]; then
    echo "First token check failed."
    exit 1
fi

echo "Passed."
