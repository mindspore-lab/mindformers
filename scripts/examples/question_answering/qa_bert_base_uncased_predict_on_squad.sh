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
echo "bash scripts/examples/question_answering/qa_bert_base_uncased_predict_on_squad.sh"
echo "The data setting could refer to ./docs/task_cards/question_answering.md"
echo "It is better to use absolute path."
echo "=============================================================================================================="

python run_mindformer.py --config ./configs/qa/run_qa_bert_base_uncased.yaml          \
                         --run_mode eval --load_checkpoint qa_bert_base_uncased_squad \
                         --predict_data "My name is Wolfgang and I live in Berlin - Where do I live?"

