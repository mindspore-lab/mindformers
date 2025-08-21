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
"""test hf dataloader"""

import os
import json
from unittest.mock import patch
import pytest
import numpy as np

from mindformers.tools.register.config import MindFormerConfig
from mindformers.dataset.dataloader.hf_dataloader import HFDataLoader


WORK_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_DATASET_PATH = os.path.join(WORK_DIR, 'alpaca.json')

DATASET_CONFIG = dict(
    data_loader=dict(
        type='HFDataLoader',
        load_func='load_dataset',
        path='',
        data_files='',
        handler=None,
        use_broadcast_data=False,
        create_compressed_eod_mask=False,
        compressed_eod_mask_length=128,
        shuffle=False),
    input_columns=["input_ids", "labels"],
    construct_args_key=["input_ids", "labels"],
    seed=42,
    python_multiprocessing=False,
    num_parallel_workers=8,
    drop_remainder=True,
    prefetch_size=1,
    numa_enable=False,
)
GLOBAL_CONFIG = MindFormerConfig(**DATASET_CONFIG)


class MockTokenizer:

    def __call__(self, sample):
        length = min(256, len(sample))
        input_ids = np.random.randint(0, 1024, length).tolist()
        return dict(input_ids=input_ids)


def generate_json():
    """generate alpaca samples and save to json"""
    num_samples = 1024
    sample = [{
        "instruction": "Explain why the following fraction is equivalent to 1/4",
        "input": "4/16",
        "output": "The fraction 4/16 is equivalent to 1/4 because both fractions represent the same value. "
                  "A fraction can be simplified by dividing both the numerator and the denominator by a common factor. "
                  "In this case, 4 is a common factor of both the numerator and the denominator of 4/16. "
                  "When we divide both by 4, we get 4/4 = 1 and 16/4 = 4, so the simplified fraction is 1/4. "
                  "Alternatively, we can think of this in terms of multiplication. For example, if we multiply "
                  "the numerator and denominator of the fraction 1/4 by 4, we get (1x4)/(4x4), or 4/16. Since "
                  "both fractions can be derived from the other through multiplication or division by the same number, "
                  "they represent the same value and are equivalent."
    }] * num_samples
    with open(JSON_DATASET_PATH, 'w') as fp:
        json.dump(sample, fp, indent=2)


@patch('mindformers.dataset.handler.base_handler.BaseInstructDataHandler.build_tokenizer',
       return_value=MockTokenizer())
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_load_alpaca_json(mock_tokenizer):
    """
    Feature: HFDataLoader load json dataset
    Description: HFDataLoader load json dataset from generated file
    Expectation: success
    """
    _ = mock_tokenizer
    generate_json()
    GLOBAL_CONFIG.data_loader.path = 'json'
    GLOBAL_CONFIG.data_loader.data_files = JSON_DATASET_PATH
    GLOBAL_CONFIG.data_loader.handler = [
        dict(type='AlpacaInstructDataHandler', seq_length=1024, padding=True)
    ]
    dataloader = HFDataLoader(**GLOBAL_CONFIG.data_loader)
    assert len(dataloader) == 1024  # num_samples

    sample = dataloader.source.dataset[0]
    assert list(sample.keys()) == ['input_ids', 'labels']
    assert len(dataloader.source.dataset[0]['input_ids']) == 1025  # legacy seq_length


@patch('mindformers.dataset.handler.base_handler.BaseInstructDataHandler.build_tokenizer',
       return_value=MockTokenizer())
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_iter_pack_alpaca(mock_tokenizer):
    """
    Feature: HFDataLoader load json dataset with packing
    Description: HFDataLoader load json dataset and pack samples
    Expectation: success
    """
    _ = mock_tokenizer
    generate_json()
    GLOBAL_CONFIG.data_loader.path = 'json'
    GLOBAL_CONFIG.data_loader.data_files = JSON_DATASET_PATH
    GLOBAL_CONFIG.data_loader.handler = [
        dict(type='AlpacaInstructDataHandler', seq_length=512, padding=False),
        dict(type='PackingHandler', pack_strategy='pack', seq_length=4096),
    ]
    dataloader = HFDataLoader(**GLOBAL_CONFIG.data_loader)
    assert len(dataloader) < 1024  # packed num_samples

    sample = dataloader.source.dataset[0]
    assert list(sample.keys()) == ['input_ids', 'labels', 'actual_seq_len']
    sample = dataloader.source[0]
    assert len(sample) == 5  # ['input_ids', 'labels', 'loss_mask', 'position_ids', 'attention_mask']
    assert sample[-1].shape == (1, 4096, 4096)


@patch('mindformers.dataset.handler.base_handler.BaseInstructDataHandler.build_tokenizer',
       return_value=MockTokenizer())
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_iter_truncate_code_alpaca(mock_tokenizer):
    """
    Feature: HFDataLoader load json dataset with packing
    Description: HFDataLoader load json dataset and pack samples
    Expectation: success
    """
    _ = mock_tokenizer
    generate_json()
    GLOBAL_CONFIG.data_loader.path = 'json'
    GLOBAL_CONFIG.data_loader.data_files = JSON_DATASET_PATH
    GLOBAL_CONFIG.data_loader.handler = [
        dict(type='CodeAlpacaInstructDataHandler', seq_length=512, padding=False),
        dict(type='PackingHandler', pack_strategy='truncate', seq_length=4096),
    ]
    dataloader = HFDataLoader(**GLOBAL_CONFIG.data_loader)
    assert len(dataloader) < 1024  # packed num_samples

    sample = dataloader.source.dataset[0]
    assert list(sample.keys()) == ['input_ids', 'labels', 'actual_seq_len']
    sample = dataloader.source[0]
    assert len(sample) == 5  # ['input_ids', 'labels', 'loss_mask', 'position_ids', 'attention_mask']
    assert sample[-1].shape == (1, 4096, 4096)


@patch('mindformers.dataset.handler.base_handler.BaseInstructDataHandler.build_tokenizer',
       return_value=MockTokenizer())
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_iter_pack_alpaca_compress_mask(mock_tokenizer):
    """
    Feature: HFDataLoader load json dataset with packing
    Description: HFDataLoader load json dataset and pack samples
    Expectation: success
    """
    _ = mock_tokenizer
    generate_json()
    GLOBAL_CONFIG.data_loader.path = 'json'
    GLOBAL_CONFIG.data_loader.data_files = JSON_DATASET_PATH
    GLOBAL_CONFIG.data_loader.create_compressed_eod_mask = True
    GLOBAL_CONFIG.data_loader.handler = [
        dict(type='AlpacaInstructDataHandler', seq_length=512, padding=False),
        dict(type='PackingHandler', pack_strategy='pack', seq_length=4096),
    ]
    dataloader = HFDataLoader(**GLOBAL_CONFIG.data_loader)
    assert len(dataloader) < 1024  # packed num_samples

    sample = dataloader.source.dataset[0]
    assert list(sample.keys()) == ['input_ids', 'labels', 'actual_seq_len']
    sample = dataloader.source[0]
    assert len(sample) == 5  # ['input_ids', 'labels', 'loss_mask', 'position_ids', 'actual_seq_len']
    assert sample[-1].size == 128


@patch('mindformers.dataset.handler.base_handler.BaseInstructDataHandler.build_tokenizer',
       return_value=MockTokenizer())
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_remove_columns(mock_tokenizer):
    """
    Feature: HFDataLoader load json dataset with remove_columns
    Description: HFDataLoader load json dataset and remove columns
    Expectation: success
    """
    _ = mock_tokenizer
    generate_json()
    GLOBAL_CONFIG.data_loader.path = 'json'
    GLOBAL_CONFIG.data_loader.data_files = JSON_DATASET_PATH
    GLOBAL_CONFIG.data_loader.create_compressed_eod_mask = True
    GLOBAL_CONFIG.data_loader.handler = [
        dict(type='AlpacaInstructDataHandler', seq_length=512, padding=False),
        dict(type='remove_columns', column_names='input_ids'),
    ]
    dataloader = HFDataLoader(**GLOBAL_CONFIG.data_loader)
    sample = dataloader.source.dataset[0]
    assert list(sample.keys()) == ['labels']


@patch('mindformers.dataset.handler.base_handler.BaseInstructDataHandler.build_tokenizer',
       return_value=MockTokenizer())
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_rename_columns(mock_tokenizer):
    """
    Feature: HFDataLoader load json dataset with rename_column
    Description: HFDataLoader load json dataset and rename columns
    Expectation: ValueError
    """
    _ = mock_tokenizer
    generate_json()
    GLOBAL_CONFIG.data_loader.path = 'json'
    GLOBAL_CONFIG.data_loader.data_files = JSON_DATASET_PATH
    GLOBAL_CONFIG.data_loader.create_compressed_eod_mask = True
    GLOBAL_CONFIG.data_loader.handler = [
        dict(type='AlpacaInstructDataHandler', seq_length=512, padding=False),
        dict(type='rename_column', original_column_name='input_ids', new_column_name='user_input'),
    ]

    with pytest.raises(ValueError):  # 'user_input' not in support columns
        HFDataLoader(**GLOBAL_CONFIG.data_loader)
