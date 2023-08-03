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
echo "bash scripts/examples/translation/t5_small_translation_finetune_on_wmt16.sh /your_wmt_path"
echo "The data setting could refer to ./docs/model_cards/t5.md"
echo "It is better to use absolute path."
echo "Please make the src_max_length and the tgt_max_length in config/t5/run_t5_small_on_wmt16.yaml"
echo "to be consistent with the seq_length and max_decoder_length in config/t5/run_t5_small_on_wmt16.yaml"
echo "Or it will raise error like "
echo "TransformerEncoderLayer x shape must be one of [[1, 1024, 512], [1024, 512]],but got [1, 32, 512]"
echo "=============================================================================================================="

data_path=$1

python run_mindformer.py --config ./configs/t5/run_t5_small_on_wmt16.yaml --train_dataset_dir $data_path