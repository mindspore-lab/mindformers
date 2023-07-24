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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/examples/token_classification/tokcls_bert_base_chinese_predict_on_cluener.sh"
echo "The data setting could refer to ./docs/task_cards/token_classification.md"
echo "It is better to use absolute path."
echo "=============================================================================================================="

python run_mindformer.py --config ./configs/tokcls/run_tokcls_bert_base_chinese.yaml                                \
                         --run_mode predict --load_checkpoint tokcls_bert_base_chinese_cluener                      \
                         --predict_data "结果上周六他们主场0：3惨败给了中游球队瓦拉多利德，近7个多月以来西甲首次输球。"