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
"""test multi-source dataloader."""
import os
import pytest

from mindformers import ToolAlpacaDataLoader

def make_test_tool_alpaca_dataset(dataset_root="./checkpoint_download", valid_num=8):
    """make a fake tool_alpaca dataset"""
    valid_data = u'{"tools": ["tools"], "conversations": [' + \
                 u'{"role": "user", "content": "content"}, ' + \
                 u'{"role": "assistant", "content": "content"}, ' + \
                 u'{"role": "tool", "name": "name", "parameters": {}, "observation": "observation"}, ' + \
                 u'{"role": "assistant", "content": "content"}]}'
    # invalid json format
    invalid_data1 = u'{"tools": ["tools"], "conversations": [' + \
                    u'{"role": "user", "content": "content"}, ' + \
                    u'{"role": "tool", "name": "name", "parameters": }, "observation": "observation"}, ' + \
                    u'{"role": "assistant", "content": "content"}]}'
    # invalid keys `conv`
    invalid_data2 = u'{"tools": ["tools"], "conv": [' + \
                    u'{"role": "user", "content": "content"}, ' + \
                    u'{"role": "assistant", "content": "content"}, ' + \
                    u'{"role": "tool", "name": "name", "parameters": {}, "observation": "observation"}, ' + \
                    u'{"role": "assistant", "content": "content"}]}'
    # invalid first conv
    invalid_data3 = u'{"tools": ["tools"], "conversations": [' + \
                    u'{"role": "assistant", "content": "content"}, ' + \
                    u'{"role": "user", "content": "content"}, ' + \
                    u'{"role": "tool", "name": "name", "parameters": {}, "observation": "observation"}, ' + \
                    u'{"role": "assistant", "content": "content"}]}'
    # invalid last conv
    invalid_data4 = u'{"tools": ["tools"], "conversations": [' + \
                    u'{"role": "user", "content": "content"}, ' + \
                    u'{"role": "assistant", "content": "content"}, ' + \
                    u'{"role": "tool", "name": "name", "parameters": {}, "observation": "observation"}, ' + \
                    u'{"role": "user", "content": "content"}]}'
    # unpaired middle conv: odd num
    invalid_data5 = u'{"tools": ["tools"], "conversations": [' + \
                    u'{"role": "user", "content": "content"}, ' + \
                    u'{"role": "assistant", "content": "content"}, ' + \
                    u'{"role": "tool", "name": "name", "parameters": {}, "observation": "observation"}, ' + \
                    u'{"role": "tool", "name": "name", "parameters": {}, "observation": "observation"}, ' + \
                    u'{"role": "assistant", "content": "content"}]}'
    # unpaired middle conv
    invalid_data6 = u'{"tools": ["tools"], "conversations": [' + \
                    u'{"role": "user", "content": "content"}, ' + \
                    u'{"role": "assistant", "content": "content"}, ' + \
                    u'{"role": "assistant", "name": "name", "parameters": {}, "observation": "observation"}, ' + \
                    u'{"role": "assistant", "content": "content"}]}'
    invalid_data = [invalid_data1, invalid_data2, invalid_data3, invalid_data4, invalid_data5, invalid_data6]

    if not os.path.exists(dataset_root):
        os.mkdir(dataset_root)
    file_name = os.path.join(dataset_root, 'tool_alpaca.jsonl')
    with open(file_name, mode='w') as fp:
        for _ in range(valid_num):
            fp.write(valid_data + '\n')
        for line in invalid_data:
            fp.write(line + '\n')

    return file_name

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tool_alpaca_dataset_correct():
    """
    Feature: Test tool_alpaca dataloader correction
    Description: Create tool_alpaca dataloader and iter it
    Expectation: The output data is different from expect data
    """
    file_name = make_test_tool_alpaca_dataset(valid_num=8)
    data_loader = ToolAlpacaDataLoader(file_name)
    data_loader = data_loader.batch(1)

    tgt_line = u"[{'tools': [Tensor(shape=[], dtype=String, value= 'tools')], " + \
               u"'conversations': [{'role': Tensor(shape=[], dtype=String, value= 'user'), " + \
               u"'content': Tensor(shape=[], dtype=String, value= 'content')}, " + \
               u"{'role': Tensor(shape=[], dtype=String, value= 'assistant'), " + \
               u"'content': Tensor(shape=[], dtype=String, value= 'content')}, " + \
               u"{'role': Tensor(shape=[], dtype=String, value= 'tool'), " + \
               u"'name': Tensor(shape=[], dtype=String, value= 'name'), 'parameters': {}, " + \
               u"'observation': Tensor(shape=[], dtype=String, value= 'observation')}, " + \
               u"{'role': Tensor(shape=[], dtype=String, value= 'assistant'), " + \
               u"'content': Tensor(shape=[], dtype=String, value= 'content')}]}]"
    for line in data_loader:
        assert str(line) == tgt_line, f"test ToolAlpacaDataLoader load correction failed, please check your code."

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tool_alpaca_dataset_invalid():
    """
    Feature: Test tool_alpaca dataloader skip invalid data ability
    Description: Create tool_alpaca dataloader and get it's length
    Expectation: len(data_loader) is different from valid_num
    """
    valid_num = 7
    file_name = make_test_tool_alpaca_dataset(valid_num=valid_num)
    data_loader = ToolAlpacaDataLoader(file_name)
    data_loader = data_loader.batch(1)

    assert len(data_loader) == valid_num, f"test ToolAlpacaDataLoader `skip invalid data function` failed, " + \
                                          f"please check your code."

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tool_alpaca_dataset_zero_data():
    """
    Feature: Test tool_alpaca dataloader zero data assertion
    Description: Create zero tool_alpaca dataloader
    Expectation: dataloader doesn't raise AssertionError
    """
    file_name = make_test_tool_alpaca_dataset(valid_num=0)
    with pytest.raises(AssertionError):
        _ = ToolAlpacaDataLoader(file_name)
