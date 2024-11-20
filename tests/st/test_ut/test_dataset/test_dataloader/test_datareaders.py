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
"""test datareaders"""
import os
import tempfile
import pytest
from mindformers.dataset.dataloader.datareaders import squad_reader, cmrc2018_reader, agnews_reader
from tests.st.test_ut.test_dataset.get_test_data import get_agnews_data, get_squad_data, get_cmrc_data


temp_dir = tempfile.TemporaryDirectory()
path = temp_dir.name


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_squad_reader():
    """
    Feature: datareaders.squad_reader
    Description: test squad_reader function
    Expectation: success
    """
    get_squad_data(path)
    data_path = os.path.join(path, "train-v1.1.json")
    res = squad_reader(data_path)
    assert len(res) == 2
    assert res["sources"] == ['Read the passage and answer the question below.\n\n### Instruction:\n'
                              'An increasing sequence: one, two, three.\n\n### Input:\n'
                              '华为是一家总部位于中国深圳的多元化科技公司\n\n### Response:']
    assert res["targets"] == ['one']


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cmrc2018_reader():
    """
    Feature: datareaders.cmrc2018_reader
    Description: test cmrc2018_reader function
    Expectation: success
    """
    get_cmrc_data(path)
    data_path = os.path.join(path, "train.json")
    res = cmrc2018_reader(data_path)
    assert len(res) == 2
    assert res["prompts"] == ['阅读文章：华为是一家总部位于中国深圳的多元化科技公司,成立于1987年,是全球最大的电信设备制造商之一。\n'
                              '问：An increasing sequence: one, two, three, five, six, seven, nine, 10, 11, 12, 13.\n'
                              '答：']
    assert res["answers"] == ['华为']


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_agnews_reader():
    """
    Feature: datareaders.agnews_reader
    Description: test agnews_reader function
    Expectation: success
    """
    get_agnews_data(path)
    data_path = os.path.join(path, "agnews")
    res = agnews_reader(data_path)
    assert len(res) == 2
    assert res["sentence"] == ['华为是一家总部位于中国深圳的多元化科技公司,成立于1987年,是全球最大的电信设备制造商之一。. '
                               'An increasing sequence: one, two, three, five, six, seven, nine, 10, 11, 12, 13.']
    assert res["label"] == [0]
