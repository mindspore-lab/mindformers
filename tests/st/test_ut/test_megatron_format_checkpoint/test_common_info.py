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
"""Test save/load common info."""

import os
import shutil
import pytest

from mindformers.checkpoint.checkpoint import CommonInfo
from mindformers.checkpoint.utils import (
    get_checkpoint_iter_dir,
    get_common_filename
)

CHECKPOINT_ROOT_DIR = "./output_megatron_format_checkpoint"

FIRST_ITER = 15
SECOND_ITER = 30
NOT_EXISTS = False


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_save_and_load_common_info_case():
    """
    Feature: Test save common info, then load them.
    Description: Simulate saving common.json twice in succession
        to ensure that the paths and contents of both accesses are normal.
        Then load the saved commonInfo twice to check whether the load function can obtain the value normally.
    Expectation: The first save is normal, and the second save can save the updated CommonInfo content.
        And no error is reported during the two loads, and the content is as expected.
    """
    # Test save common info part.
    base_common_info = CommonInfo(
        epoch_num=1,
        step_num=FIRST_ITER,
        global_step=FIRST_ITER,
        loss_scale=2.5,
        global_batch_size=128
    )

    # First save
    first_checkpoint_path = get_checkpoint_iter_dir(CHECKPOINT_ROOT_DIR, FIRST_ITER)
    os.makedirs(first_checkpoint_path, exist_ok=True)

    first_common_info_path = get_common_filename(CHECKPOINT_ROOT_DIR, FIRST_ITER)
    base_common_info.save_common(first_common_info_path)

    assert os.path.isfile(first_common_info_path)
    with open(first_common_info_path, "r", encoding="utf-8") as f:
        data = f.read()
        assert '"epoch_num": 1' in data
        assert '"step_num": 15' in data

    # Second save
    base_common_info.step_num = SECOND_ITER
    base_common_info.global_step = SECOND_ITER
    base_common_info.loss_scale = 1.0
    base_common_info.global_batch_size = 256

    second_checkpoint_path = get_checkpoint_iter_dir(CHECKPOINT_ROOT_DIR, SECOND_ITER)
    os.makedirs(second_checkpoint_path, exist_ok=True)

    second_common_info_path = get_common_filename(CHECKPOINT_ROOT_DIR, SECOND_ITER)
    base_common_info.save_common(second_common_info_path)

    assert os.path.isfile(first_common_info_path)  # ensure first save not be removed
    assert os.path.isfile(second_common_info_path)
    with open(second_common_info_path, "r", encoding="utf-8") as f:
        data = f.read()
        assert '"global_step": 30' in data
        assert '"step_num": 30' in data
        assert '"loss_scale": 1.0' in data
        assert '"global_batch_size": 256' in data

    # Test load common info part.
    # First load
    first_common_info_path = get_common_filename(CHECKPOINT_ROOT_DIR, FIRST_ITER)
    assert os.path.isfile(first_common_info_path)

    first_loaded = CommonInfo.load_common(first_common_info_path)
    assert first_loaded.epoch_num == 1
    assert first_loaded.step_num == FIRST_ITER
    assert first_loaded.global_step == FIRST_ITER
    assert first_loaded.loss_scale == 2.5
    assert first_loaded.global_batch_size == 128

    # Second load
    second_common_info_path = get_common_filename(CHECKPOINT_ROOT_DIR, SECOND_ITER)
    assert os.path.isfile(second_common_info_path)

    second_loaded = CommonInfo.load_common(second_common_info_path)
    assert second_loaded.epoch_num == 1
    assert second_loaded.step_num == SECOND_ITER
    assert second_loaded.global_step == SECOND_ITER
    assert second_loaded.loss_scale == 1.0
    assert second_loaded.global_batch_size == 256

    # Clear all save files for test
    shutil.rmtree(CHECKPOINT_ROOT_DIR)
    assert os.path.exists(CHECKPOINT_ROOT_DIR) == NOT_EXISTS
    assert os.path.exists(first_common_info_path) == NOT_EXISTS
    assert os.path.exists(second_common_info_path) == NOT_EXISTS
