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
echo "bash scripts/examples/image_classification/vit_base_p16_predict_on_imagenet1k.sh"
echo "The data setting could refer to ./docs/task_cards/image_classification.md"
echo "It is better to use absolute path."
echo "=============================================================================================================="

python run_mindformer.py --config ./configs/vit/run_vit_base_p16_224_100ep.yaml --run_mode predict \
                         --predict_data "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/
                                         XFormer_for_mindspore/clip/sunflower.png"