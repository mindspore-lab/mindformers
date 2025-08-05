# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Dataset Utils."""

import mindspore as ms

from mindformers.tools.utils import (
    get_dp_from_dataset_strategy,
    get_real_group_size,
    get_real_rank
)


def is_dataset_built_on_rank() -> bool:
    """check which rank need to build dataset."""
    global_rank_id = get_real_rank()
    stage_num = ms.get_auto_parallel_context("pipeline_stages")
    total_device_num = get_real_group_size() // stage_num
    dp = get_dp_from_dataset_strategy()
    tp = int(total_device_num // dp)

    local_stage_num = int(global_rank_id // (dp * tp))

    # when not stage 0 or last stage, no need to build dataset.
    if 0 < local_stage_num < (stage_num - 1):
        return False

    # In tp group, only need one card to build dataset, others don't need to build dataset.
    if global_rank_id % tp != 0:
        return False

    return True
