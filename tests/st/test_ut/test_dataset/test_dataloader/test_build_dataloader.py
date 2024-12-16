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
"""test build dataloader"""
import os
import unittest
import tempfile
import pytest
import mindspore.dataset

from mindformers.dataset.dataloader import build_dataset_loader
from tests.st.test_ut.test_dataset.get_test_data import get_mindrecord_data


class TestBuildDataloader(unittest.TestCase):
    """ A test class for testing build dataloader"""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.path = cls.temp_dir.name
        get_mindrecord_data(cls.path)
        cls.data_path = os.path.join(cls.path, "test.mindrecord")

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_default(self):
        """test default logic"""
        dataset_loader_config = {'type': 'MindDataset', 'dataset_files': self.data_path}
        data_loader = build_dataset_loader(dataset_loader_config)
        assert isinstance(data_loader, mindspore.dataset.MindDataset)
        data_loader = build_dataset_loader()
        assert data_loader is None
        data_loader = build_dataset_loader(class_name="MindDataset", dataset_files=self.data_path)
        assert isinstance(data_loader, mindspore.dataset.MindDataset)
