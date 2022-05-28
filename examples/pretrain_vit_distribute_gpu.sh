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

if [ $# != 1 ]
then
  echo "Usage: bash run_distribute_train.sh [HOSTFILE]"
  exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

HOSTFILE=$(get_real_path $1)
CONFIG_FILE=$(get_real_path ../transformer/configs/vit/vit_imagenet2012_config.yml)

if [ ! -f $HOSTFILE ]
then
    echo "error: HOSTFILE=$HOSTFILE is not a file"
exit 1
fi

if [ ! -f $CONFIG_FILE ]
then
    echo "error: config_path=$CONFIG_FILE is not a file"
exit 1
fi

export DEVICE_NUM=8
export RANK_SIZE=8

rm -rf ./train_parallel
mkdir ./train_parallel
cp ../pretrain_vit.py ./train_parallel
cp pretrain_vit*.sh ./train_parallel
cp -r ../transformer/configs/vit/*.yml ./train_parallel
cp -r ../transformer ./train_parallel
mkdir ./train_parallel/tasks
cp -r ../tasks/vision ./train_parallel/tasks
cd ./train_parallel || exit
echo "start training"

mpirun --allow-run-as-root -n 8 --hostfile $HOSTFILE --output-filename log_output python pretrain_vit.py --config_path=$CONFIG_FILE \
       --device_target=GPU &> log &
