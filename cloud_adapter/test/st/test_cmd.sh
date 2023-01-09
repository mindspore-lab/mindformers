#!/bin/bash
# 
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
set -e

# init
fm registry

# modelarts
fm show --scenario modelarts --app_config obs://**/app_config_test.yaml
fm show --scenario modelarts --app_config obs://**/app_config_test.yaml --job_id 1203c47b-05b0-4056-8364-5fe6ae1bde45
fm stop --scenario modelarts --app_config obs://**/app_config_test.yaml --job_id 1203c47b-05b0-4056-8364-5fe6ae1bde45
fm delete --scenario modelarts --app_config obs://**/app_config_test.yaml --job_id aaeeb71f-6f90-4cc1-a15f-f4890dd877f1
fm job-status --scenario modelarts --app_config obs://**/app_config_test.yaml --job_id 1203c47b-05b0-4056-8364-5fe6ae1bde45

fm finetune --scenario modelarts  --app_config obs://**/app_config_test.yaml --job_name 56556443334444 --model_config_path obs://**/model_config_test.yaml

fm evaluate --scenario modelarts --model_config_path 's3://**/opt_caption_finetune_end2end.yml' --ckpt_path obs://**/ckpt_path/
fm evaluate --scenario modelarts --app_config obs://**/app_config_220802_mav2.yaml --model_config_path 's3://**/opt_caption_finetune_end2end.yml' --ckpt_path obs://**/ckpt_path/
fm finetune --scenario modelarts  --app_config obs://**/app_config_test.yaml --job_name --job_name 56556443334444 --model_config_path obs://**/model_config_test.yaml

fm evaluate --scenario modelarts --model_config_path 's3://lcj/yml/opt_caption_finetune_end2end.yml' --ckpt_path obs://hemuhui-test/ckpt_path/
fm evaluate --scenario modelarts --app_config obs://hemuhui-test/my-test/app_config_220802_mav2.yaml --model_config_path 's3://lcj/yml/opt_caption_finetune_end2end.yml' --ckpt_path obs://hemuhui-test/ckpt_path/