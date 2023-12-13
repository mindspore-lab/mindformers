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

output_dir=$(cat $CONFIG_FILE | grep output_dir)
output_dir=$(echo "$output_dir" | awk '{print $2}')
if [ ! -n "$output_dir" ]; then
  echo "Error: No output_dir in $CONFIG_FILE"
  exit 1
fi
if [[ $output_dir =~ "'" ]]; then
  output_dir=${output_dir#*\'}
  output_dir=${output_dir%\'*}
else
  output_dir=${output_dir#*\"}
  output_dir=${output_dir%\"*}
fi
if [[ $output_dir == "./output" ]]
then
  echo "output_dir is ./output"
  export LOCAL_DEFAULT_PATH="../../output"
elif [[ ! $output_dir =~ ^/ ]]; then
  echo "Error: output_dir should be absolute path, but get $output_dir."
  echo "The default value of output_dir should be './output'. Replace it with an absolute path if you want to customize it."
  exit 1
else
  echo "output_dir is $output_dir"
  export LOCAL_DEFAULT_PATH=$output_dir
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
mkdir -p $LOCAL_DEFAULT_PATH/log/rank_0
python run_mindformer.py --config=$CONFIG_FILE --use_parallel=False --run_mode=$RUN_STATUS \
--output_dir=$LOCAL_DEFAULT_PATH &> $LOCAL_DEFAULT_PATH/log/rank_0/mindformer.log &
cd ..

# if you want kill current job, you can use as follow:
# kill -9 $(ps aux | grep "python run_mindformer.py" | grep -v grep | awk '{print $2}')