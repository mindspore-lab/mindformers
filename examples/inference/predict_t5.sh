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
echo "bash scripts/pretrain/pretrain_t5.sh"
echo "for example: bash scripts/pretrain/pretrain_t5.sh"
echo "=============================================================================================================="

python -m transformer.models.t5.t5_predict \
     --optimizer="adam" \
     --seq_length=20 \
     --max_decode_length=128 \
     --parallel_mode="stand_alone" \
     --max_position_embeddings=1024 \
     --d_kv=64 \
     --global_batch_size=1 \
     --attention_dropout_prob=0.1 \
     --hidden_dropout_prob=0.1 \
     --vocab_size=32128 \
     --hidden_size=512 \
     --intermediate_size=2048 \
     --num_hidden_layers=6 \
     --num_heads=8 \
     --load_checkpoint_path=mindspore_t5_small.ckpt \
     --generate=True \
     --bucket_boundaries=16 \
     --has_relative_bias=True \
     --device_target="GPU"
