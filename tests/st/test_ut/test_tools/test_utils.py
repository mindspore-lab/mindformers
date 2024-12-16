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
import json
import tempfile
from unittest import mock
import pytest
import mindspore as ms
from mindformers.tools.utils import (
    check_obs_url,
    check_list,
    check_file,
    get_net_outputs,
    get_num_nodes_devices,
    generate_rank_list,
    convert_nodes_devices_input,
    str2bool,
    parse_value,
    create_and_write_info_to_txt,
    is_version_le,
    get_pipeline_rank_ids
)


tmp_dir = tempfile.TemporaryDirectory()
tmp_path = tmp_dir.name


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_check_obs_url():
    """
    Feature: utils.check_obs_url
    Description: test check_obs_url function
    Expectation: success
    """
    with pytest.raises(TypeError):
        assert check_obs_url(1)
    with pytest.raises(TypeError):
        assert check_obs_url("mock://")
    assert check_obs_url("obs://")


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_check_list():
    """
    Feature: utils.check_list
    Description: test check_list function
    Expectation: success
    """
    with pytest.raises(ValueError):
        assert check_list(var_name="mock", list_var=[3], num=1)\


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_check_file():
    """
    Feature: utils.check_file
    Description: test check_file function
    Expectation: success
    """
    with pytest.raises(ValueError):
        assert check_file("not_a_dir")
    with pytest.raises(ValueError):
        assert check_file(tmp_path)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_net_outputs():
    """
    Feature: utils.get_net_outputs
    Description: test get_net_outputs function
    Expectation: success
    """
    tensor_0, tensor_1 = ms.Tensor([0]), ms.Tensor([1])
    assert get_net_outputs(tensor_0) == 0.0
    assert get_net_outputs([tensor_0, tensor_1]).asnumpy().tolist() == [0]


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_num_nodes_devices():
    """
    Feature: utils.get_num_nodes_devices
    Description: test get_num_nodes_devices function
    Expectation: success
    """
    assert get_num_nodes_devices(2) == (1, 2)
    assert get_num_nodes_devices(3) == (0, 8)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_generate_rank_list():
    """
    Feature: utils.generate_rank_list
    Description: test generate_rank_list function
    Expectation: success
    """
    assert generate_rank_list(stdout_nodes=[0], stdout_devices=[0, 1, 2, 3]) == [0, 1, 2, 3]


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_convert_nodes_devices_input():
    """
    Feature: utils.convert_nodes_devices_input
    Description: test convert_nodes_devices_input function
    Expectation: success
    """
    assert convert_nodes_devices_input(var=None, num=4) == (0, 1, 2, 3)
    assert convert_nodes_devices_input(var={"start": 0, "end": 4}, num=0) == (0, 1, 2, 3)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_str2bool():
    """
    Feature: utils.str2bool
    Description: test str2bool function
    Expectation: success
    """
    assert not str2bool("FALSE")
    assert str2bool("TRUE")
    with pytest.raises(Exception):
        assert str2bool("mock")


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_parse_value():
    """
    Feature: utils.parse_value
    Description: test parse_value function
    Expectation: success
    """
    assert parse_value("2")
    assert parse_value("2.1")
    assert parse_value("True")
    assert parse_value(json.dumps({"mock": 0}))
    assert parse_value("mock")


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_create_and_write_info_to_txt():
    """
    Feature: utils.create_and_write_info_to_txt
    Description: test create_and_write_info_to_txt function
    Expectation: success
    """
    mock_path = os.path.join(tmp_path, "mock.txt")
    create_and_write_info_to_txt(txt_path=mock_path, info="mock")
    with open(mock_path, "r", encoding="utf-8") as r:
        assert r.read().strip() == "mock"


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_is_version_le():
    """
    Feature: utils.is_version_le
    Description: test is_version_le function
    Expectation: success
    """
    assert is_version_le(current_version="5.2.0", base_version="5.3.0")
    assert is_version_le(current_version="5.3.0", base_version="5.3.0")



class TestGetPipelineRankMethod:
    """A test class for testing tools utils method."""

    @pytest.mark.run(order=1)
    @mock.patch('mindformers.tools.utils.get_real_group_size')
    @mock.patch('mindformers.tools.utils.ms.get_auto_parallel_context')
    def test_get_pipeline_rank_ids_with_valid(self, mock_get_parallal_stage_num, mock_get_real_group_size):
        """test get pipeline rank ids in normal condition."""
        mock_get_real_group_size.return_value = 8
        mock_get_parallal_stage_num.return_value = 2

        test_rank_ids = get_pipeline_rank_ids()
        expected_ids = [0, 4]

        assert len(test_rank_ids) == len(expected_ids)
        assert test_rank_ids == expected_ids

    @pytest.mark.run(order=2)
    @mock.patch('mindformers.tools.utils.get_real_group_size')
    @mock.patch('mindformers.tools.utils.ms.get_auto_parallel_context')
    def test_get_pipeline_rank_ids_with_invalid(self, mock_get_parallal_stage_num, mock_get_real_group_size):
        """test get pipeline rank ids in normal condition."""
        mock_get_real_group_size.return_value = 8
        mock_get_parallal_stage_num.return_value = 3

        test_rank_ids = get_pipeline_rank_ids()

        expected_ids = [-1]

        assert len(test_rank_ids) == len(expected_ids)
        assert test_rank_ids == expected_ids
