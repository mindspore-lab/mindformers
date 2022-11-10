#!/bin/bash
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

set -e

SCRIPT_BASEDIR=$(realpath "$(dirname "$0")")

PROJECT_DIR=$(realpath "$SCRIPT_BASEDIR/../../")
ST_PATH="$PROJECT_DIR/tests/st"

run_test() {
    OS_NAME=$(uname)
    echo "Start to run test on $OS_NAME"
    cd "$PROJECT_DIR" || exit
    echo "python -m pytest -v '$ST_PATH'"
    python -m pytest -v "$ST_PATH"
    echo "Test all use cases success."
}

run_test
