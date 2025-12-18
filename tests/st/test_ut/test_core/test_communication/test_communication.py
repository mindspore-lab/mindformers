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
"""Test communication utilities for parallel training."""

from unittest.mock import patch

import pytest

from mindformers.parallel_core.training_graph import communication
from mindformers.parallel_core.training_graph.communication import (
    get_op_group_name,
    get_cp_group_name,
    get_dp_group_name,
)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("mindformers.parallel_core.training_graph.communication.create_communication_group")
def test_get_op_group_name_with_mock(mock_create_group):
    """
    Feature: get_op_group_name()
    Description: Test the get op group name with mock.
    Expectation: The get op group name with mock should be correct.
    """
    mock_create_group.return_value = "mock_group"
    communication.OP_GROUP_NAME.clear()

    # case 0: model_parallel_size > 1
    result = get_op_group_name(rank_id=3, real_op_size=2, model_parallel_size=2)
    assert result == ("mock_group", [1, 3])
    mock_create_group.assert_called_once_with([1, 3])

    second_result = get_op_group_name(rank_id=3, real_op_size=2, model_parallel_size=2)
    assert second_result == result
    mock_create_group.assert_called_once()

    # case 1: model_parallel_size = 1
    result = get_op_group_name(rank_id=3, real_op_size=2, model_parallel_size=1)
    assert result == ("mock_group", [2, 3])

    # case 2: model_parallel_size = 4
    result = get_op_group_name(rank_id=3, real_op_size=2, model_parallel_size=4)
    assert result == ("mock_group", [3, 7])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("mindformers.parallel_core.training_graph.communication.create_communication_group")
def test_get_cp_group_name_with_mock(mock_create_group):
    """
    Feature: get_cp_group_name()
    Description: Test the get cp group name with mock.
    Expectation: The get cp group name with mock should be correct.

    For pp=2, dp=2, cp=2, tp=2 (16 cards):
    PP stage 0: ranks 0-7
    PP stage 1: ranks 8-15 (mirrors stage 0)

    Within each PP stage, layout is same as 8-card case:
        rank | dp | cp | tp
        0/8  | 0  | 0  | 0
        1/9  | 0  | 0  | 1
        2/10 | 0  | 1  | 0
        3/11 | 0  | 1  | 1
        4/12 | 1  | 0  | 0
        5/13 | 1  | 0  | 1
        6/14 | 1  | 1  | 0
        7/15 | 1  | 1  | 1

    CP group: ranks with same (dp, tp), different cp
    """
    mock_create_group.return_value = "mock_group"
    communication.CP_GROUP_NAME.clear()

    # PP stage 0 (8 cards): dp=2, tp=2, cp=2
    assert get_cp_group_name(rank_id=0, dp=2, tp=2, cp=2) == ("mock_group", [0, 2])
    assert get_cp_group_name(rank_id=1, dp=2, tp=2, cp=2) == ("mock_group", [1, 3])
    assert get_cp_group_name(rank_id=4, dp=2, tp=2, cp=2) == ("mock_group", [4, 6])
    assert get_cp_group_name(rank_id=5, dp=2, tp=2, cp=2) == ("mock_group", [5, 7])

    # PP stage 1 (16 cards): ranks 8-15 mirror stage 0
    assert get_cp_group_name(rank_id=8, dp=2, tp=2, cp=2) == ("mock_group", [8, 10])
    assert get_cp_group_name(rank_id=9, dp=2, tp=2, cp=2) == ("mock_group", [9, 11])
    assert get_cp_group_name(rank_id=12, dp=2, tp=2, cp=2) == ("mock_group", [12, 14])
    assert get_cp_group_name(rank_id=13, dp=2, tp=2, cp=2) == ("mock_group", [13, 15])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("mindformers.parallel_core.training_graph.communication.create_communication_group")
def test_get_dp_group_name_with_mock(mock_create_group):
    """
    Feature: get_dp_group_name()
    Description: Test the get dp group name with mock.
    Expectation: The get dp group name with mock should be correct.

    For pp=2, dp=2, cp=2, tp=2 (16 cards):
    PP stage 0: ranks 0-7
    PP stage 1: ranks 8-15 (mirrors stage 0)

    Within each PP stage, layout is same as 8-card case:
        rank | dp | cp | tp
        0/8  | 0  | 0  | 0
        1/9  | 0  | 0  | 1
        2/10 | 0  | 1  | 0
        3/11 | 0  | 1  | 1
        4/12 | 1  | 0  | 0
        5/13 | 1  | 0  | 1
        6/14 | 1  | 1  | 0
        7/15 | 1  | 1  | 1

    DP group: ranks with same (cp, tp), different dp
    """
    mock_create_group.return_value = "mock_group"
    communication.DP_GROUP_NAME.clear()

    # PP stage 0 (8 cards): dp=2, tp=2, cp=2
    assert get_dp_group_name(rank_id=0, dp=2, tp=2, cp=2) == ("mock_group", [0, 4])
    assert get_dp_group_name(rank_id=1, dp=2, tp=2, cp=2) == ("mock_group", [1, 5])
    assert get_dp_group_name(rank_id=2, dp=2, tp=2, cp=2) == ("mock_group", [2, 6])
    assert get_dp_group_name(rank_id=3, dp=2, tp=2, cp=2) == ("mock_group", [3, 7])

    # PP stage 1 (16 cards): ranks 8-15 mirror stage 0
    assert get_dp_group_name(rank_id=8, dp=2, tp=2, cp=2) == ("mock_group", [8, 12])
    assert get_dp_group_name(rank_id=9, dp=2, tp=2, cp=2) == ("mock_group", [9, 13])
    assert get_dp_group_name(rank_id=10, dp=2, tp=2, cp=2) == ("mock_group", [10, 14])
    assert get_dp_group_name(rank_id=11, dp=2, tp=2, cp=2) == ("mock_group", [11, 15])
