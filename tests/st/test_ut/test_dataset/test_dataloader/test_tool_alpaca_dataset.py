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
"""test ToolAlpaca Dataset."""
import os
import pytest

import mindspore as ms
from mindformers import ToolAlpacaDataLoader

ms.set_context(mode=1, device_target='CPU')


def make_test_tool_alpaca_dataset(dataset_dir="./checkpoint_download", valid_num=8):
    """generate a fake ToolAlpaca Dataset"""
    valid_data = '{"tools": ["tools"], "conversations": [' + \
                 '{"role": "user", "content": "content"}, ' + \
                 '{"role": "assistant", "content": "content"}, ' + \
                 '{"role": "tool", "name": "name", "parameters": {}, "observation": "observation"}, ' + \
                 '{"role": "assistant", "content": "content"}]}'
    # invalid json format
    invalid_data1 = '{"tools": ["tools"], "conversations": [' + \
                    '{"role": "user", "content": "content"}, ' + \
                    '{"role": "tool", "name": "name", "parameters": }, "observation": "observation"}, ' + \
                    '{"role": "assistant", "content": "content"}]}'
    # invalid keys `conv`
    invalid_data2 = '{"tools": ["tools"], "conv": [' + \
                    '{"role": "user", "content": "content"}, ' + \
                    '{"role": "assistant", "content": "content"}, ' + \
                    '{"role": "tool", "name": "name", "parameters": {}, "observation": "observation"}, ' + \
                    '{"role": "assistant", "content": "content"}]}'
    # invalid first conv
    invalid_data3 = '{"tools": ["tools"], "conversations": [' + \
                    '{"role": "assistant", "content": "content"}, ' + \
                    '{"role": "user", "content": "content"}, ' + \
                    '{"role": "tool", "name": "name", "parameters": {}, "observation": "observation"}, ' + \
                    '{"role": "assistant", "content": "content"}]}'
    # invalid last conv
    invalid_data4 = '{"tools": ["tools"], "conversations": [' + \
                    '{"role": "user", "content": "content"}, ' + \
                    '{"role": "assistant", "content": "content"}, ' + \
                    '{"role": "tool", "name": "name", "parameters": {}, "observation": "observation"}, ' + \
                    '{"role": "user", "content": "content"}]}'
    # unpaired middle conv: odd num
    invalid_data5 = '{"tools": ["tools"], "conversations": [' + \
                    '{"role": "user", "content": "content"}, ' + \
                    '{"role": "assistant", "content": "content"}, ' + \
                    '{"role": "tool", "name": "name", "parameters": {}, "observation": "observation"}, ' + \
                    '{"role": "tool", "name": "name", "parameters": {}, "observation": "observation"}, ' + \
                    '{"role": "assistant", "content": "content"}]}'
    # unpaired middle conv
    invalid_data6 = '{"tools": ["tools"], "conversations": [' + \
                    '{"role": "user", "content": "content"}, ' + \
                    '{"role": "assistant", "content": "content"}, ' + \
                    '{"role": "assistant", "name": "name", "parameters": {}, "observation": "observation"}, ' + \
                    '{"role": "assistant", "content": "content"}]}'
    invalid_data = [invalid_data1, invalid_data2, invalid_data3, invalid_data4, invalid_data5, invalid_data6]

    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    file_name = os.path.join(dataset_dir, f'tool_alpaca_{valid_num}.jsonl')
    with open(file_name, mode='w', encoding='utf-8') as fp:
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
    Feature: ToolAlpacaDataLoader correction
    Description: Create ToolAlpacaDataLoader and iter it
    Expectation: The output data is different from expect data
    """
    file_name = make_test_tool_alpaca_dataset(
        dataset_dir="./checkpoint_download_tool_alpaca_dataset_correct", valid_num=8
    )
    data_loader = ToolAlpacaDataLoader(file_name)

    target = {
        'conversations': [
            {'content': 'content', 'role': 'user'},
            {'content': 'content', 'role': 'assistant'},
            {'name': 'name', 'observation': 'observation', 'parameters': {}, 'role': 'tool'},
            {'content': 'content', 'role': 'assistant'}
        ], 'tools': ['tools']
    }

    for _, line in enumerate(data_loader):
        sample = line[0]
        # compare 'conversations'
        assert len(sample['conversations']) == 4
        for idx, conversation in enumerate(sample['conversations']):
            for k, v in conversation.items():
                if isinstance(target['conversations'][idx][k], str):
                    assert target['conversations'][idx][k] == str(v)
                else:
                    assert target['conversations'][idx][k] == v
        # compare 'tools'
        assert target['tools'][0] == str(sample['tools'][0])
        break


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tool_alpaca_dataset_invalid():
    """
    Feature: ToolAlpacaDataLoader skip invalid data ability
    Description: Create ToolAlpacaDataLoader and get its length
    Expectation: len(data_loader) is different from valid_num
    """
    valid_num = 7
    file_name = make_test_tool_alpaca_dataset(dataset_dir="./checkpoint_download_tool_alpaca_dataset_invalid",
                                              valid_num=valid_num)
    data_loader = ToolAlpacaDataLoader(file_name)
    data_loader = data_loader.batch(1)

    assert len(data_loader) == valid_num, \
        "test ToolAlpacaDataLoader `skip invalid data function` failed, please check your code."


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tool_alpaca_dataset_zero_data():
    """
    Feature: ToolAlpacaDataLoader zero data assertion
    Description: Create zero ToolAlpacaDataLoader
    Expectation: ValueError
    """
    file_name = make_test_tool_alpaca_dataset(valid_num=0)
    with pytest.raises(ValueError):
        _ = ToolAlpacaDataLoader(file_name)
