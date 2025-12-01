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
import json
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_common_info_with_partial_fields(tmp_path):
    """
    Feature: Test CommonInfo with only partial fields set.
    Description: Create CommonInfo with only some fields set, save and load it.
    Expectation: The fields that are not set should be None after loading.
    """
    # Create CommonInfo with only partial fields
    common_info = CommonInfo(
        epoch_num=5,
        global_step=1000
    )

    # Save and load
    common_path = os.path.join(tmp_path, "common.json")
    common_info.save_common(common_path)

    loaded_info = CommonInfo.load_common(common_path)

    # Verify set fields are preserved
    assert loaded_info.epoch_num == 5
    assert loaded_info.global_step == 1000

    # Verify unset fields are None
    assert loaded_info.step_num is None
    assert loaded_info.loss_scale is None
    assert loaded_info.global_batch_size is None


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_common_info_with_none_fields(tmp_path):
    """
    Feature: Test CommonInfo with explicit None fields.
    Description: Create CommonInfo with explicit None values, save and load it.
    Expectation: The None values should be preserved after loading.
    """
    # Create CommonInfo with explicit None values
    common_info = CommonInfo(
        epoch_num=None,
        step_num=None,
        global_step=2000,
        loss_scale=None,
        global_batch_size=None
    )

    # Save and load
    common_path = os.path.join(tmp_path, "common.json")
    common_info.save_common(common_path)

    loaded_info = CommonInfo.load_common(common_path)

    # Verify values are preserved
    assert loaded_info.epoch_num is None
    assert loaded_info.step_num is None
    assert loaded_info.global_step == 2000
    assert loaded_info.loss_scale is None
    assert loaded_info.global_batch_size is None


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_load_nonexistent_file(tmp_path):
    """
    Feature: Test loading from a nonexistent file.
    Description: Attempt to load CommonInfo from a file that doesn't exist.
    Expectation: A FileNotFoundError should be raised.
    """
    nonexistent_path = os.path.join(tmp_path, "nonexistent.json")

    with pytest.raises(FileNotFoundError):
        CommonInfo.load_common(nonexistent_path)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_load_invalid_json(tmp_path):
    """
    Feature: Test loading from an invalid JSON file.
    Description: Attempt to load CommonInfo from a file with invalid JSON.
    Expectation: A ValueError should be raised.
    """
    invalid_path = os.path.join(tmp_path, "invalid.json")

    # Create file with invalid JSON
    with open(invalid_path, "w", encoding='utf-8') as f:
        f.write("invalid json content")

    with pytest.raises(ValueError):
        CommonInfo.load_common(invalid_path)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_save_to_nonexistent_directory(tmp_path):
    """
    Feature: Test saving to a nonexistent directory.
    Description: Attempt to save CommonInfo to a directory that doesn't exist.
    Expectation: An error should be raised when trying to save to a nonexistent directory.
    """
    # Create a subdirectory that doesn't exist
    nonexistent_dir = os.path.join(tmp_path, "nonexistent_dir")
    common_path = os.path.join(nonexistent_dir, "common.json")

    common_info = CommonInfo(epoch_num=1, global_step=500)

    # This should raise an error because the directory doesn't exist
    with pytest.raises(FileNotFoundError):
        common_info.save_common(common_path)

    # Now create the directory and try again
    os.makedirs(nonexistent_dir, exist_ok=True)
    common_info.save_common(common_path)

    assert os.path.exists(common_path)
    assert os.path.isfile(common_path)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_save_overwrite_existing_file(tmp_path):
    """
    Feature: Test overwriting an existing file.
    Description: Save CommonInfo to a file, then save again with different values.
    Expectation: The file should be overwritten with the new values.
    """
    common_path = os.path.join(tmp_path, "common.json")

    # First save
    common_info1 = CommonInfo(epoch_num=1, global_step=100)
    common_info1.save_common(common_path)

    # Verify first save
    with open(common_path, "r", encoding='utf-8') as f:
        data = json.load(f)
        assert data["epoch_num"] == 1
        assert data["global_step"] == 100

    # Second save with different values
    common_info2 = CommonInfo(epoch_num=2, global_step=200)
    common_info2.save_common(common_path)

    # Verify overwrite
    with open(common_path, "r", encoding='utf-8') as f:
        data = json.load(f)
        assert data["epoch_num"] == 2
        assert data["global_step"] == 200


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_common_info_equality(tmp_path):
    """
    Feature: Test equality of CommonInfo objects.
    Description: Create two identical CommonInfo objects and verify they are equal.
    Expectation: The objects should have the same attribute values.
    """
    common_info1 = CommonInfo(
        epoch_num=3,
        step_num=50,
        global_step=1500,
        loss_scale=3.0,
        global_batch_size=256
    )

    common_info2 = CommonInfo(
        epoch_num=3,
        step_num=50,
        global_step=1500,
        loss_scale=3.0,
        global_batch_size=256
    )

    # Save and load one of them
    common_path = os.path.join(tmp_path, "common.json")
    common_info1.save_common(common_path)
    loaded_info = CommonInfo.load_common(common_path)

    # Verify all attributes are equal
    assert loaded_info.epoch_num == common_info2.epoch_num
    assert loaded_info.step_num == common_info2.step_num
    assert loaded_info.global_step == common_info2.global_step
    assert loaded_info.loss_scale == common_info2.loss_scale
    assert loaded_info.global_batch_size == common_info2.global_batch_size


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_common_info_json_format(tmp_path):
    """
    Feature: Test JSON format of saved CommonInfo.
    Description: Save CommonInfo and verify the JSON format is correct.
    Expectation: The JSON should be properly formatted with all fields.
    """
    common_info = CommonInfo(
        epoch_num=4,
        step_num=100,
        global_step=2000,
        loss_scale=1.5,
        global_batch_size=512
    )

    common_path = os.path.join(tmp_path, "common.json")
    common_info.save_common(common_path)

    # Load and verify JSON structure
    with open(common_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Verify all fields are present
    assert "epoch_num" in data
    assert "step_num" in data
    assert "global_step" in data
    assert "loss_scale" in data
    assert "global_batch_size" in data

    # Verify values are correct
    assert data["epoch_num"] == 4
    assert data["step_num"] == 100
    assert data["global_step"] == 2000
    assert data["loss_scale"] == 1.5
    assert data["global_batch_size"] == 512
