# !/bin/bash
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

CUR_DIR=$(dirname $(readlink -f $0))
TOP_DIR=$(readlink -f $CUR_DIR/../)

FM_DIR=$TOP_DIR/fm

function pull_setup() {
  cd $TOP_DIR
  pip3 install --upgrade setuptools wheel
  pip3 install pyyaml
  python3 $TOP_DIR/setup.py bdist_wheel
  cd -
}

function main() {
  pull_setup
}

main
