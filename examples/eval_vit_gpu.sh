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

if [ $# != 2 ]
then
  echo "Usage: bash run_distribute_train.sh [HOSTFILE] [CONFIG_PATH]"
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
CONFIG_FILE=$(get_real_path $2)

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

rm -rf ./eval
mkdir ./eval
cp ../*.py ./eval
cp *.sh ./eval
cp -r ../config/*.yml ./eval
cp -r ../src ./eval
cd ./eval || exit

mpirun --allow-run-as-root -n 8 --hostfile $HOSTFILE python eval.py --config_path=$CONFIG_FILE \
       --device_target=GPU &> log &