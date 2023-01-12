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

if [ $# != 3 ]
then
  echo "Usage Help: bash run_standalone.sh [CONFIG_PATH] [DEVICE_ID] [RUN_STATUS] "
  exit 1
fi

check_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

CONFIG_FILE=$(check_real_path $1)
DEVICE_ID=$2
RUN_STATUS=$3

if [ ! -f $CONFIG_FILE ]
then
    echo "error: config_path=$CONFIG_FILE is not a file"
exit 1
fi

ulimit -u unlimited
export DEVICE_ID=${DEVICE_ID}

rm -rf ./mf_standalone
mkdir ./mf_standalone
cp ../*.py ./mf_standalone
cp -r ../configs ./mf_standalone
cp -r ../mindformers ./mf_standalone
cd ./mf_standalone || exit
echo "start training for device $DEVICE_ID"
env > env.log
python run_mindformer.py --config=$CONFIG_FILE --use_parallel=False --run_mode=$RUN_STATUS &> mindformer.log &
cd ..

# if you want kill current job, you can use as follow:
# kill -9 $(ps aux | grep "python run_mindformer.py" | grep -v grep | awk '{print $2}')