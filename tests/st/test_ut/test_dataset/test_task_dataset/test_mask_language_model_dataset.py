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
"""test mask_language_model_dataset"""
import copy
import os
import unittest
import tempfile
import pytest
from mindformers import MindFormerConfig
from mindformers.dataset import MaskLanguageModelDataset
from tests.st.test_ut.test_dataset.get_test_data import get_mindrecord_data


def check_dataset_config(dataset_config, params):
    """Check `dataset_config`, If it is empty, use the input parameter to create a new `dataset_config`."""
    if not dataset_config:
        params.pop("dataset_config")
        kwargs = params.pop("kwargs") if params.get("kwargs") else {}
        params.update(kwargs)
        dataset_config = MindFormerConfig(**params)
    return dataset_config


class TestMaskLanguageModelDataset(unittest.TestCase):
    """A test class for testing MaskLanguageModelDataset"""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.path = cls.temp_dir.name
        get_mindrecord_data(cls.path)
        cls.raw_config = {
            "dataset_config": None,
            "data_loader": {"dataset_files": [os.path.join(cls.path, "test.mindrecord")], "type": "MindDataset"},
            "input_columns": ['input_ids', 'labels'],
            "batch_size": 1,
            "auto_tune": False,
            "profile": False,
            "seed": 0,
            "prefetch_size": 1,
            "numa_enable": False,
            "filepath_prefix": './autotune',
            "autotune_per_step": 10,
            "repeat": 1,
            "num_parallel_workers": 8,
            "drop_remainder": True
        }
        cls.dataset_config = check_dataset_config(None, copy.deepcopy(cls.raw_config))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_default(self):
        """test default logic"""
        dataset = MaskLanguageModelDataset(dataset_config=self.dataset_config)
        for item in dataset:
            assert len(item) == 2
            assert all(item[0].asnumpy()[0] == [44, 47, 53, 0, 3, 59, 3, 39, 9, 19, 21, 50, 36, 23, 6, 24])
            assert all(item[1].asnumpy()[0] == [44, 47, 53, 0, 3, 59, 3, 39, 9, 19, 21, 50, 36, 23, 6, 24])
            break

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_other(self):
        """test other logic"""
        raw_config = copy.deepcopy(self.raw_config)
        raw_config["data_loader"] = {"dataset_dir": self.path, "type": "MindDataset"}
        dataset_config = check_dataset_config(None, raw_config)
        dataset = MaskLanguageModelDataset(dataset_config=dataset_config)
        for item in dataset:
            assert len(item) == 2
            assert all(item[0].asnumpy()[0] == [44, 47, 53, 0, 3, 59, 3, 39, 9, 19, 21, 50, 36, 23, 6, 24])
            assert all(item[1].asnumpy()[0] == [44, 47, 53, 0, 3, 59, 3, 39, 9, 19, 21, 50, 36, 23, 6, 24])
            break
        raw_config = copy.deepcopy(self.raw_config)
        raw_config["data_loader"] = {"dataset_dir": os.path.join(self.path, "test.mindrecord"), "type": "MindDataset"}
        dataset_config = check_dataset_config(None, raw_config)
        dataset = MaskLanguageModelDataset(dataset_config=dataset_config)
        for item in dataset:
            assert len(item) == 2
            assert all(item[0].asnumpy()[0] == [44, 47, 53, 0, 3, 59, 3, 39, 9, 19, 21, 50, 36, 23, 6, 24])
            assert all(item[1].asnumpy()[0] == [44, 47, 53, 0, 3, 59, 3, 39, 9, 19, 21, 50, 36, 23, 6, 24])
            break
        with pytest.raises(ValueError):
            raw_config = copy.deepcopy(self.raw_config)
            raw_config["data_loader"] = {"dataset_files": "",
                                         "type": "MindDataset"}
            dataset_config = check_dataset_config(None, raw_config)
            assert MaskLanguageModelDataset(dataset_config)
        with pytest.raises(NotImplementedError):
            raw_config = copy.deepcopy(self.raw_config)
            raw_config["data_loader"] = {"dataset_files": "",
                                         "type": "mock"}
            dataset_config = check_dataset_config(None, raw_config)
            assert MaskLanguageModelDataset(dataset_config)
