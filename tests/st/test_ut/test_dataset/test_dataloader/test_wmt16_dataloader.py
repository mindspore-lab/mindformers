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
"""test wmt16 dataloader"""
import unittest
import tempfile
from mindformers.dataset.dataloader.wmt16_dataloader import WMT16DataSet, WMT16DataLoader
import pytest
from tests.st.test_ut.test_dataset.get_test_data import get_wmt16_data


class TestWMT16Dataloader(unittest.TestCase):
    """ A test class for testing WMT16 dataloader"""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.path = cls.temp_dir.name
        get_wmt16_data(cls.path)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_default(self):
        """test default logic"""
        dataloader = WMT16DataLoader(dataset_dir=self.path)
        columns = dataloader.column_names
        assert columns == ["source", "target"]
        dataloader = dataloader.batch(1)
        for item in dataloader:
            assert len(item) == 2
            assert item[0].asnumpy()[0] == "Membership of Parliament: see Minutes"
            assert item[1].asnumpy()[0] == "Componenţa Parlamentului: a se vedea procesul-verbal"
            break

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_not_dir(self):
        """test not dir input logic"""
        mock_dir = "not_a_dir"
        with pytest.raises(ValueError):
            assert WMT16DataLoader(dataset_dir=mock_dir)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_column_names(self):
        """test column names logic"""
        with pytest.raises(TypeError):
            column_names = 1
            assert WMT16DataLoader(self.path, column_names=column_names)
        with pytest.raises(Exception):
            column_names = [1]
            assert WMT16DataLoader(self.path, column_names=column_names)
        with pytest.raises(Exception):
            column_names = [1, 2]
            assert WMT16DataLoader(self.path, column_names=column_names)


class TestWMT16DataSet(unittest.TestCase):
    """ A test class for testing WMT16 dataset"""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.path = cls.temp_dir.name
        get_wmt16_data(cls.path)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_default(self):
        """test default logic"""
        dataset = WMT16DataSet(dataset_dir=self.path)
        res = dataset[0]
        assert len(res) == 2
        assert res[0] == 'Membership of Parliament: see Minutes'
        assert res[1] == 'Componenţa Parlamentului: a se vedea procesul-verbal'

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_other(self):
        """test other logic"""
        with pytest.raises(ValueError):
            mock_dir = "not_a_dir"
            assert WMT16DataSet(mock_dir)
