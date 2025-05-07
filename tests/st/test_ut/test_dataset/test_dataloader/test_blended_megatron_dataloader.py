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
"""test BlendedMegatronDatasetDataLoader"""

import os
import pytest
import numpy as np
from mindformers.dataset.blended_datasets.indexed_dataset import IndexedDatasetBuilder
from mindformers.dataset.dataloader.blended_megatron_dataloader import BlendedMegatronDatasetDataLoader


WORK_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_megatron_dataset():
    """generate .bin and .idx file."""
    os.makedirs(f"{WORK_DIR}/data", exist_ok=True)
    bin_file = f"{WORK_DIR}/data/megatron_dataset.bin"
    idx_file = f"{WORK_DIR}/data/megatron_dataset.idx"

    data_size = 32
    seq_length = 1024
    random_ids = [np.random.randint(low=1, high=12001, size=seq_length) for _ in range(data_size)]

    builder = IndexedDatasetBuilder(bin_file, dtype=np.int32)
    for random_id in random_ids:
        builder.add_document(random_id, [len(random_id)])
    builder.finalize(idx_file)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_blended_megatron_dataloader():
    """
    Feature: BlendedMegatronDatasetDataLoader
    Description: test BlendedMegatronDatasetDataLoader iteration
    Expectation: data shape correct
    """
    generate_megatron_dataset()

    sizes = [16, 0, 0]
    config = dict(
        seed=1234,
        seq_length=1024,
        split="1, 0, 0",
        reset_position_ids=True,
        reset_attention_mask=True,
        eod_mask_loss=True,
        create_attention_mask=True,
        create_compressed_eod_mask=False,
        eod_pad_length=128,
        num_dataset_builder_threads=1,
        eod=2,
        pad=3,
        data_path=['1', f"{WORK_DIR}/data/megatron_dataset"]
    )
    dataloader = BlendedMegatronDatasetDataLoader(
        datasets_type="GPTDataset",
        sizes=sizes,
        config=config,
        column_names=["input_ids", "labels", "loss_mask", "position_ids", "attention_mask"],
    )

    assert len(dataloader) == 17  # oversampled
    # pylint: disable=E1133
    for input_ids, labels, loss_mask, position_ids, attention_mask in dataloader:
        assert input_ids.shape == (1024,)
        assert labels.shape == (1024,)
        assert loss_mask.shape == (1024,)
        assert position_ids.shape == (1024,)
        assert attention_mask.shape == (1, 1024, 1024)
