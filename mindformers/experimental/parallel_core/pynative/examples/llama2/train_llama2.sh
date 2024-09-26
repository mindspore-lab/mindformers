#!/bin/bash

set -e

export HCCL_BUFFSIZE=200

export HCCL_EXEC_TIMEOUT=600

export ASCEND_RT_VISIBLE_DEVICES='0'

# 显存分析
# export MS_MEMORY_STATISTIC=2

port=8828

# kill process
PIDS=$(sudo lsof -i :$port | awk 'NR>1 {print $2}')
if [ -n "$PIDS" ]; then
    for pid in $PIDS; do
        kill -9 $pid
        echo "Killed process $pid"
    done
else
    echo "No processes found listening on port $port."
fi

project_dir=$(cd "$(dirname "$0")" || exit; pwd)
log_path="msrun_log"

echo "in project_dir: $project_dir"

rm -rf "${log_path}"
mkdir "${log_path}"
echo "train start, log path: ${log_path}"

# 计算设备数量
IFS=',' read -r -a devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
work_num=${#devices[@]}

config_path=$1

if [ -z "$config_path" ]; then
    config_path="pretrain_llama2.yaml"
fi

msrun --worker_num "$work_num" --local_worker_num="$work_num" --master_port=$port --log_dir="$log_path" --join=True --cluster_time_out=300 train.py --config_path="${config_path}"
