#!/bin/bash
# Copyright (c) Huawei Technologies Co. Ltd. 2021-2021. All rights reserved.
# ===============================================================================
set -e

cd $(cd "$(dirname "$0")"; pwd)

rm -rf ./mxLaunchKit
cp -r ../../../mxLaunchKit ./

current_time=$(date +%Y%m%d%H%M%S)

docker build --rm=true --no-cache -t fmtk-ma_base:${current_time} -f Dockerfile .