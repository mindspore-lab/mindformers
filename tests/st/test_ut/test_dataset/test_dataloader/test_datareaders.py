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
"""test datareaders"""
import os
import tempfile
import pytest
from mindformers.dataset.dataloader.datareaders import agnews_reader
from tests.st.test_ut.test_dataset.get_test_data import get_agnews_data


temp_dir = tempfile.TemporaryDirectory()
path = temp_dir.name


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
