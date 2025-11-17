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
"""test hf dataloader streaming"""

import os
import json
from unittest.mock import patch
import pytest

from mindformers.tools.register.config import MindFormerConfig
from mindformers.dataset.dataloader.hf_dataloader import HFDataLoader
from .test_hf_dataloader import MockTokenizer

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_DATASET_PATH = os.path.join(WORK_DIR, 'alpaca_streaming.json')

DATASET_CONFIG = {
    "data_loader": {
        "type": "HFDataLoader",
        "load_func": "load_dataset",
        "path": "",
        "data_files": "",
        "streaming": True,
        "size": 6,
        "dataset_state_dir": f"{WORK_DIR}/saved_state",
        "handler": None,
        "use_broadcast_data": False,
        "create_compressed_eod_mask": False,
        "compressed_eod_mask_length": 128,
        "shuffle": False,
    }
}
GLOBAL_CONFIG = MindFormerConfig(**DATASET_CONFIG)


def generate_json():
    """generate alpaca samples and save to json"""
    num_samples = 6
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
    }] * (num_samples // 2)
    sample += [{
        "instruction": "Give three tips for staying healthy.",
        "input": "",
        "output": "1. Eat a balanced and nutritious diet: Make sure your meals are inclusive of a variety of fruits "
                  "and vegetables, lean protein, whole grains, and healthy fats. This helps to provide your body "
                  "with the essential nutrients to function at its best and can help prevent chronic diseases."
                  "\n\n2. Engage in regular physical activity: Exercise is crucial for maintaining strong bones, "
                  "muscles, and cardiovascular health. Aim for at least 150 minutes of moderate aerobic exercise "
                  "or 75 minutes of vigorous exercise each week.\n\n3. Get enough sleep: Getting enough quality "
                  "sleep is crucial for physical and mental well-being. It helps to regulate mood, improve cognitive "
                  "function, and supports healthy growth and immune function. Aim for 7-9 hours of sleep each night."
    }] * (num_samples // 2)
    with open(JSON_DATASET_PATH, 'w', encoding='utf-8') as fp:
        json.dump(sample, fp, indent=2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_load_streaming_resume():
    """
    Feature: HFDataLoader load dataset with streaming=True
    Description: HFDataLoader save and resume in streaming mode
    Expectation: success
    """
    generate_json()
    GLOBAL_CONFIG.data_loader.path = 'json'
    GLOBAL_CONFIG.data_loader.data_files = JSON_DATASET_PATH
    GLOBAL_CONFIG.data_loader.save_step = 4
    dataloader = HFDataLoader(**GLOBAL_CONFIG.data_loader, python_multiprocessing=False)

    resume_target = None
    for idx, sample in enumerate(dataloader.source):
        resume_target = sample
        if idx >= 3:
            break

    GLOBAL_CONFIG.data_loader.resume_step = 4
    dataloader = HFDataLoader(**GLOBAL_CONFIG.data_loader, python_multiprocessing=False)
    dataloader.set_init_step(4)
    resume_sample = dataloader.source[1]

    assert resume_sample[-1] == resume_target[-1]


@patch('mindformers.dataset.handler.base_handler.BaseInstructDataHandler.build_tokenizer',
       return_value=MockTokenizer())
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_load_streaming_pack(mock_tokenizer):
    """
    Feature: HFDataLoader load dataset with streaming=True
    Description: HFDataLoader pack samples in streaming mode
    Expectation: success
    """
    _ = mock_tokenizer
    generate_json()
    GLOBAL_CONFIG.data_loader.path = 'json'
    GLOBAL_CONFIG.data_loader.data_files = JSON_DATASET_PATH
    GLOBAL_CONFIG.data_loader.handler = [
        {"type": "AlpacaInstructDataHandler", "seq_length": 4096, "padding": False},
        {"type": "PackingHandler", "pack_strategy": "pack", "seq_length": 4096}
    ]
    dataloader = HFDataLoader(**GLOBAL_CONFIG.data_loader, python_multiprocessing=False)

    target_sample = dataloader.source[0]
    assert len(target_sample) == 5  # ['input_ids', 'labels', 'loss_mask', 'position_ids', 'attention_mask']
    assert target_sample[-1].shape == (1, 4096, 4096)

    GLOBAL_CONFIG.data_loader.create_compressed_eod_mask = True
    dataloader = HFDataLoader(**GLOBAL_CONFIG.data_loader, python_multiprocessing=False)

    target_sample = dataloader.source[0]
    assert target_sample[-1].shape == (GLOBAL_CONFIG.data_loader.compressed_eod_mask_length,)
