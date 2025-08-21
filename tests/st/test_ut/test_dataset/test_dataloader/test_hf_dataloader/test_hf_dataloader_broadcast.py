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
"""test HFDataLoader broadcast."""

import os
from unittest.mock import patch
import pytest
import numpy as np
from datasets import Dataset

from mindspore import Tensor

import mindformers
from mindformers.tools.register.config import MindFormerConfig
from mindformers.dataset.dataloader.hf_dataloader import HFDataLoader
from mindformers.dataset.handler.base_handler import BaseInstructDataHandler

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
MOCK_DATASET_PATH = os.path.join(WORK_DIR, 'mock_dataset')
INVALID_DATASET_PATH = os.path.join(WORK_DIR, 'invalid_dataset')

DATASET_CONFIG = dict(
    data_loader=dict(
        type='HFDataLoader',
        load_func='load_from_disk',
        dataset_path='',
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


class MockHandler(BaseInstructDataHandler):

    # pylint: disable=W0221
    def __call__(self, dataset):
        """mock handle func"""
        return dataset


def generate_dataset():
    """generate dataset file."""
    num_samples = 1024
    input_ids = [np.random.randint(0, 1024, size=256).tolist()] * num_samples
    labels = [np.random.randint(0, 1024, size=256).tolist()] * num_samples
    dataset = {'input_ids': input_ids, 'labels': labels}

    dataset = Dataset.from_dict(dataset)
    dataset.save_to_disk(MOCK_DATASET_PATH)


def mock_broadcast(data):
    """mock ops.Broadcast"""
    return data


def mock_broadcast_received(data):
    """mock ops.Broadcast received"""
    _ = data
    dataset_size = Tensor([1024])
    num_columns = Tensor([2])
    seq_length = Tensor([1024])
    return dataset_size, num_columns, seq_length


@patch('mindformers.dataset.dataloader.hf_dataloader.get_real_group_size', return_value=2)
@patch('mindformers.dataset.dataloader.hf_dataloader.skip_barrier_controller', return_value=None)
@patch('mindformers.dataset.dataloader.hf_dataloader.is_dataset_built_on_rank', return_value=True)
@patch('mindformers.dataset.dataloader.hf_dataloader.ops.Broadcast', return_value=mock_broadcast)
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_broadcast_main_rank(
        mock_get_real_group_size, mock_skip_barrier_controller, mock_is_dataset_built_on_rank, mock_ops_broadcast
):
    """
    Feature: HFDataLoader enable mock dataset
    Description: HFDataLoader broadcast mocked data
    Expectation: success
    """
    _ = mock_get_real_group_size, mock_skip_barrier_controller, mock_is_dataset_built_on_rank, mock_ops_broadcast
    setattr(mindformers.dataset.handler, 'MockHandler', MockHandler)

    # prepare dataset
    generate_dataset()

    GLOBAL_CONFIG.data_loader.dataset_path = MOCK_DATASET_PATH
    GLOBAL_CONFIG.data_loader.handler = [dict(type='MockHandler')]
    GLOBAL_CONFIG.data_loader.use_broadcast_data = True

    HFDataLoader(**GLOBAL_CONFIG.data_loader)


@patch('mindformers.dataset.dataloader.hf_dataloader.get_real_group_size', return_value=2)
@patch('mindformers.dataset.dataloader.hf_dataloader.skip_barrier_controller', return_value=None)
@patch('mindformers.dataset.dataloader.hf_dataloader.is_dataset_built_on_rank', return_value=False)
@patch('mindformers.dataset.dataloader.hf_dataloader.ops.Broadcast', return_value=mock_broadcast_received)
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_broadcast_received_rank(
        mock_get_real_group_size, mock_skip_barrier_controller, mock_is_dataset_built_on_rank, mock_ops_broadcast
):
    """
    Feature: HFDataLoader enable mock dataset
    Description: HFDataLoader receive mocked data
    Expectation: success
    """
    _ = mock_get_real_group_size, mock_skip_barrier_controller, mock_is_dataset_built_on_rank, mock_ops_broadcast
    setattr(mindformers.dataset.handler, 'MockHandler', MockHandler)

    # prepare dataset
    generate_dataset()

    GLOBAL_CONFIG.data_loader.dataset_path = MOCK_DATASET_PATH
    GLOBAL_CONFIG.data_loader.handler = [dict(type='MockHandler')]
    GLOBAL_CONFIG.data_loader.use_broadcast_data = True

    HFDataLoader(**GLOBAL_CONFIG.data_loader)


def generate_invalid_dataset():
    """generate invalid dataset file."""
    num_samples = 1024
    input_ids = [np.random.randint(0, 1024, size=256).tolist()] * num_samples
    labels = [np.random.randint(0, 1024, size=256).tolist()] * num_samples
    mask = [np.random.randint(0, 1024, size=(256, 256)).tolist()] * num_samples
    dataset = {'input_ids': input_ids, 'labels': labels, 'mask': mask}

    dataset = Dataset.from_dict(dataset)
    dataset.save_to_disk(INVALID_DATASET_PATH)


@patch('mindformers.dataset.dataloader.hf_dataloader.get_real_group_size', return_value=2)
@patch('mindformers.dataset.dataloader.hf_dataloader.skip_barrier_controller', return_value=None)
@patch('mindformers.dataset.dataloader.hf_dataloader.is_dataset_built_on_rank', return_value=True)
@patch('mindformers.dataset.dataloader.hf_dataloader.ops.Broadcast', return_value=mock_broadcast)
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_broadcast_invalid_dataset(
        mock_get_real_group_size, mock_skip_barrier_controller, mock_is_dataset_built_on_rank, mock_ops_broadcast
):
    """
    Feature: HFDataLoader load invalid dataset
    Description: HFDataLoader raise ValueError loading invalid dataset
    Expectation: ValueError
    """
    _ = mock_get_real_group_size, mock_skip_barrier_controller, mock_is_dataset_built_on_rank, mock_ops_broadcast
    setattr(mindformers.dataset.handler, 'MockHandler', MockHandler)

    # prepare dataset
    generate_invalid_dataset()

    GLOBAL_CONFIG.data_loader.dataset_path = INVALID_DATASET_PATH
    GLOBAL_CONFIG.data_loader.handler = [dict(type='MockHandler')]
    GLOBAL_CONFIG.data_loader.use_broadcast_data = True

    with pytest.raises(ValueError):
        HFDataLoader(**GLOBAL_CONFIG.data_loader)
