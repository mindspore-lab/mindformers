# Copyright 2024 Huawei Technologies Co., Ltd
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
"""test utils"""
import os
import stat
import tempfile
from pathlib import Path
from unittest import mock
import pytest

from mindspore import context

from mindformers.tools.utils import (
    check_in_modelarts,
    get_output_root_path,
    is_version_le,
    is_version_ge,
    get_epoch_and_step_from_ckpt_name,
    str2bool,
    parse_value,
    set_safe_mode_for_file_or_dir,
    PARALLEL_MODE,
    MODE,
    Validator,
    check_obs_url,
    get_rank_id_from_ckpt_name,
    replace_rank_id_in_ckpt_name,
    get_ascend_log_path,
    calculate_pipeline_stage,
    divide,
    is_pynative,
    create_and_write_info_to_txt,
    check_ckpt_file_name,
    get_times_epoch_and_step_from_ckpt_name,
    is_last_pipeline_stage,
    get_dp_from_dataset_strategy,
    check_shared_disk,
    replace_tk_to_mindpet,
    get_num_nodes_devices,
)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_check_in_modelarts_true():
    """
    Feature: Utils functions
    Description: Test utils functions.
    Expectation: Run successfully.
    """
    with mock.patch.dict(os.environ, {"MA_LOG_DIR": "/tmp"}):
        assert check_in_modelarts() is True

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_check_in_modelarts_false():
    """
    Feature: Utils functions
    Description: Test utils functions.
    Expectation: Run successfully.
    """
    with mock.patch.dict(os.environ, clear=True):
        assert check_in_modelarts() is False

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_output_root_path_default():
    """
    Feature: Utils functions
    Description: Test utils functions.
    Expectation: Run successfully.
    """
    with mock.patch.dict(os.environ, {}, clear=True):
        path = get_output_root_path()
        assert path == os.path.realpath("./output")

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_output_root_path_env():
    """
    Feature: Utils functions
    Description: Test utils functions.
    Expectation: Run successfully.
    """
    with mock.patch.dict(os.environ, {"LOCAL_DEFAULT_PATH": "/custom/output"}):
        path = get_output_root_path()
        assert path == "/custom/output"

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_is_version_le():
    """
    Feature: Utils functions
    Description: Test utils functions.
    Expectation: Run successfully.
    """
    assert is_version_le("1.8.1", "1.11.0") is True
    assert is_version_le("1.11.0", "1.11.0") is True
    assert is_version_le("2.0.0", "1.11.0") is False

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_is_version_ge():
    """
    Feature: Utils functions
    Description: Test utils functions.
    Expectation: Run successfully.
    """
    assert is_version_ge("1.11.0", "1.8.1") is True
    assert is_version_ge("1.11.0", "1.11.0") is True
    assert is_version_ge("1.8.1", "1.11.0") is False

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_epoch_and_step_from_ckpt_name():
    """
    Feature: Utils functions
    Description: Test utils functions.
    Expectation: Run successfully.
    """
    epoch, step = get_epoch_and_step_from_ckpt_name("model-5_100.ckpt")
    assert epoch == 5
    assert step == 100

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_epoch_and_step_invalid():
    """
    Feature: Utils functions
    Description: Test utils functions.
    Expectation: Run successfully.
    """
    with pytest.raises(ValueError, match="Can't match epoch and step"):
        get_epoch_and_step_from_ckpt_name("invalid_name.txt")

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_str2bool():
    """
    Feature: Utils functions
    Description: Test utils functions.
    Expectation: Run successfully.
    """
    assert str2bool("True") is True
    assert str2bool("False") is False

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_str2bool_invalid():
    """
    Feature: Utils functions
    Description: Test utils functions.
    Expectation: Run successfully.
    """
    with pytest.raises(Exception, match="Invalid Bool Value"):
        str2bool("maybe")

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_parse_value():
    """
    Feature: Utils functions
    Description: Test utils functions.
    Expectation: Run successfully.
    """
    assert parse_value("123") == 123
    assert parse_value("3.14") == 3.14
    assert parse_value("True") is True
    assert parse_value('{"a": 1}') == {"a": 1}
    assert parse_value("hello") == "hello"

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_safe_mode_for_file_or_dir():
    """
    Feature: Utils functions
    Description: Test utils functions.
    Expectation: Run successfully.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test.txt"
        dir_path = Path(tmpdir) / "subdir"
        dir_path.mkdir()

        file_path.write_text("test")
        set_safe_mode_for_file_or_dir([str(file_path), str(dir_path)])

        assert (file_path.stat().st_mode & stat.S_IRUSR) != 0
        assert (file_path.stat().st_mode & stat.S_IWUSR) != 0
        assert (dir_path.stat().st_mode & stat.S_IXUSR) != 0

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_parallel_mode_mapping():
    """
    Feature: Utils functions
    Description: Test utils functions.
    Expectation: Run successfully.
    """
    assert PARALLEL_MODE["DATA_PARALLEL"] == context.ParallelMode.DATA_PARALLEL
    assert PARALLEL_MODE[0] == context.ParallelMode.DATA_PARALLEL

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_mode_mapping():
    """
    Feature: Utils functions
    Description: Test utils functions.
    Expectation: Run successfully.
    """
    assert MODE["GRAPH_MODE"] == context.GRAPH_MODE
    assert MODE[0] == context.GRAPH_MODE

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_validator_check_type():
    """
    Feature: Utils functions
    Description: Test utils functions.
    Expectation: Run successfully.
    """
    Validator.check_type(42, int)
    with pytest.raises(TypeError):
        Validator.check_type("42", int)

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_check_obs_url_valid():
    """
    Feature: Utils functions
    Description: Test utils functions.
    Expectation: Run successfully.
    """
    check_obs_url("obs://bucket/path")
    check_obs_url("s3://bucket/path")

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_check_obs_url_invalid():
    """
    Feature: Utils functions
    Description: Test utils functions.
    Expectation: Run successfully.
    """
    with pytest.raises(TypeError, match="should be start with obs:// or s3://"):
        check_obs_url("/local/path")

    with pytest.raises(TypeError, match="type should be a str"):
        check_obs_url(123)

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_rank_id_from_ckpt_name():
    """
    Feature: Utils functions
    Description: Test utils functions.
    Expectation: Run successfully.
    """
    rank = get_rank_id_from_ckpt_name("llama_7b_rank_3-5_100.ckpt")
    assert rank == 3

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_replace_rank_id_in_ckpt_name():
    """
    Feature: Utils functions
    Description: Test utils functions.
    Expectation: Run successfully.
    """
    new_name = replace_rank_id_in_ckpt_name("model_rank_2-1_50.ckpt", 5)
    assert new_name == "model_rank_5-1_50.ckpt"

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_ascend_log_path():
    """
    Feature: Utils functions
    Description: Test utils functions.
    Expectation: Run successfully.
    """
    os.environ['ASCEND_PROCESS_LOG_PATH'] = '/home/log'
    assert get_ascend_log_path() == '/home/log'

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_calculate_pipeline_stage():
    """
    Feature: Utils functions
    Description: Test utils functions.
    Expectation: Run successfully.
    """
    layers_per_stage = [4, 4]
    model_layers = [6]
    input_layers_per_stage = layers_per_stage.copy()
    result = calculate_pipeline_stage(input_layers_per_stage, model_layers)
    expected = [
        {
            "offset": [1, -1],      # [4-3, 2-3]
            "start_stage": 0,
            "stage_num": 2
        }
    ]
    assert result == expected

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_divide():
    """
    Feature: Utils functions
    Description: Test utils functions.
    Expectation: Run successfully.
    """
    res = divide(10, 2)
    assert res == 5

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_is_pynative():
    """
    Feature: Utils functions
    Description: Test utils functions.
    Expectation: Run successfully.
    """
    os.environ['ENFORCE_EAGER'] = 'true'
    res = is_pynative()
    assert res

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_create_and_write_info_to_txt():
    """
    Feature: Utils functions
    Description: Test utils functions.
    Expectation: Run successfully.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        txt_path = os.path.join(tmpdir, "output.txt")
        info = "Hello, world!"

        create_and_write_info_to_txt(txt_path, info)

        assert os.path.exists(txt_path)
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert content == "Hello, world!"

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_check_ckpt_file_name():
    """
    Feature: Utils functions
    Description: Test utils functions.
    Expectation: Run successfully.
    """
    ckpt_name = "llama_0-3_1.ckpt"
    res = check_ckpt_file_name(ckpt_name)
    assert res
    ckpt_name = "dsadsdsadasd"
    res = check_ckpt_file_name(ckpt_name)
    assert not res

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_times_epoch_and_step_from_ckpt_name():
    """
    Feature: Utils functions
    Description: Test utils functions.
    Expectation: Run successfully.
    """
    ckpt_name = "llama_0-3_1.ckpt"
    res = check_ckpt_file_name(ckpt_name)
    if res:
        times, epcoh, step = get_times_epoch_and_step_from_ckpt_name(ckpt_name)
        assert times == 0
        assert epcoh == 3
        assert step == 1

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_is_last_pipeline_stage():
    """
    Feature: Utils functions
    Description: Test utils functions.
    Expectation: Run successfully.
    """
    with mock.patch("mindformers.tools.utils.get_real_group_size", return_value=8), \
         mock.patch("mindformers.tools.utils.get_real_rank", return_value=6), \
         mock.patch("mindspore.get_auto_parallel_context", return_value=2):
        assert is_last_pipeline_stage() is True

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_dp_from_dataset_strategy():
    """
    Feature: Utils functions
    Description: Test utils functions.
    Expectation: Run successfully.
    """
    with mock.patch("mindspore.get_auto_parallel_context", return_value=[[2, 1]]):
        dp = get_dp_from_dataset_strategy()
        assert dp == 2

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_check_shared_disk():
    """
    Feature: Utils functions
    Description: Test utils functions.
    Expectation: Run successfully.
    """
    disk_path = "/home/workspace"
    res = check_shared_disk(disk_path=disk_path)
    assert not res

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_replace_tk_to_mindpet():
    """
    Feature: Utils functions
    Description: Test utils functions.
    Expectation: Run successfully.
    """
    ckpt_dict = {"tk_delta": 1}
    new_ckpt = replace_tk_to_mindpet(ckpt_dict)
    assert new_ckpt['mindpet_delta'] == 1

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_num_nodes_devices():
    """
    Feature: Utils functions
    Description: Test utils functions.
    Expectation: Run successfully.
    """
    rank_size = 7
    with mock.patch("mindformers.tools.utils.get_device_num_per_node", return_value=8):
        num_nodes, num_devices = get_num_nodes_devices(rank_size=rank_size)
        assert num_nodes == 1
        assert num_devices == rank_size
