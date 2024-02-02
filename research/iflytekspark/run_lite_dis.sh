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

clear

##### rm and reset
unset MS_ENABLE_REF_MODE
rm -rf kernel_meta_* rank_* tmp_weight*

##### run mode
device_id=0
batch_size=1
stream_return=False
max_seq_len=32768
model_root=/your/path/to/exported_mindir
device_num=2
start_device_id=0

##### path join
model_dir=$model_root/iflytekspark-bs${batch_size}-seq${max_seq_len}
full_model_name=full_bs${batch_size}_seq${max_seq_len}_rank{}_graph.mindir
inc_model_name=inc_bs${batch_size}_seq${max_seq_len}_rank{}_graph.mindir
sample_model_path=$model_dir/sample_bs${batch_size}_rank{}.mindir
full_cfg_path=./lite_config/full_dyn_infer_cfg_dis.ini
inc_cfg_path=./lite_config/inc_dyn_infer_cfg_akg_dis.ini
sample_config_path=./lite_config/inc_dyn_infer_cfg_dis.ini
postprocess_config_path=./run_iflytekspark_13b_lite_infer_800T_A2_64G_dis.yaml

##### call main.py
nohup mpiexec --allow-run-as-root -np $device_num python3 ./lite_infer_main.py \
                        --device_id $device_id  \
                        --batch_size $batch_size \
                        --max_seq_len $max_seq_len \
                        --prefill_model_path $model_dir/$full_model_name \
                        --decode_model_path $model_dir/$inc_model_name \
                        --prefill_model_config_path $full_cfg_path \
                        --decode_model_config_path $inc_cfg_path \
                        --sample_model_path $sample_model_path \
                        --sample_model_config_path $sample_config_path \
                        --postprocess_config_path $postprocess_config_path \
                        --stream_return $stream_return \
                        --max_out_lenth $max_seq_len \
                        --tokenizer_file "/your/path/to/tokenizer" \
                        --input_file "/your/path/to/input_file" \
                        --start_device_id $start_device_id \
                        --prompt "" > log/lite_dis.log &
    
