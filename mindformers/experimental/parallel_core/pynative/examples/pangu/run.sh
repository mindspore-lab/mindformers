#!/bin/bash

set -e

export HCCL_BUFFSIZE=1

export HCCL_EXEC_TIMEOUT=600

export ASCEND_RT_VISIBLE_DEVICES='1'

# 显存分析
# export MS_MEMORY_STATISTIC=2

port=8848

# kill process
PIDS=$(sudo lsof -i :$port | awk 'NR>1 {print $2}')
if [ -n "$PIDS" ]; then
    for pid in $PIDS; do
        sudo kill -9 $pid
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

msrun --worker_num "$work_num" --local_worker_num="$work_num" --master_port=$port --log_dir="$log_path" --join=True --cluster_time_out=300 train.py --model_type ori

