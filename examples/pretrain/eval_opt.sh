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
echo "bash examples/pretrain/eval_opt.sh 'HELLO WORLD'"
echo "for example: bash examples/pretrain/eval_opt.sh 'HELLOW WORD'"
echo "=============================================================================================================="
export GLOG_v=3
SAMPLES=$1

python -m transformer.predict \
    --config=./transformer/configs/opt/opt.yaml \
    --seq_length=1024 \
    --parallel_mode="stand_alone" \
    --global_batch_size=1 \
    --vocab_size=50272 \
    --hidden_size=2560 \
    --load_checkpoint_path="./converted_mindspore_opt.ckpt" \
    --vocab_path="./vocab.json" \
    --num_layers=32 \
    --eval=True \
    --input_samples="${SAMPLES}" \
    --num_heads=32 \
    --generate=True \
    --full_batch=False \
    --device_target="Ascend" > eval_opt.log 2>&1 &
