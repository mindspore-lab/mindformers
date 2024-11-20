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
"""import cifar100 dataloader"""
import unittest
import tempfile
from mindformers.dataset.dataloader.cifar100_dataloader import Cifar100DataLoader, Cifar100DataSet
import pytest

from tests.st.test_ut.test_dataset.get_test_data import get_cifar100_data


class TestCifar100Dataloader(unittest.TestCase):
    """ A test class for testing Cifar100 dataloader"""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.path = cls.temp_dir.name
        get_cifar100_data(cls.path)

    def test_default(self):
        """test default logic"""
        dataloader = Cifar100DataLoader(self.path)
        columns = dataloader.column_names
        assert set(columns) == {"image", "text", "label"}
        dataloader = dataloader.batch(1)
        for item in dataloader:
            assert item[0].shape == (1, 32, 32, 3)
            assert all(item[0][0][0][0].asnumpy() == [117, 118, 196])
            assert item[1].shape == (1, 2)
            assert item[2].shape == (1,)
            assert all(item[2].asnumpy() == [0])
            break

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_not_dir(self):
        """test not dir input logic"""
        mock_dir = "not_a_dir"
        with pytest.raises(Exception):
            assert Cifar100DataLoader(mock_dir)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_column_names(self):
        """test columns names logic"""
        with pytest.raises(Exception):
            column_names = 1
            assert Cifar100DataLoader(self.path, column_names=column_names)
        with pytest.raises(Exception):
            column_names = [1]
            assert Cifar100DataLoader(self.path, column_names=column_names)
        with pytest.raises(Exception):
            column_names = [1, 2, 3]
            assert Cifar100DataLoader(self.path, column_names=column_names)


class TestCifar100DataSet(unittest.TestCase):
    """ A test class for testing Cifar100 dataset"""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.path = cls.temp_dir.name
        get_cifar100_data(cls.path)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_default(self):
        """test default logic"""
        dataset = Cifar100DataSet(self.path)
        assert dataset.image.shape == (1, 32, 32, 3)
        assert all(dataset.image[0][0][0] == [117, 118, 196])
        assert dataset.label == [0]
        assert dataset.label_names == ['fine_1', 'fine_2']
        assert dataset.text == ['This is a photo of fine_1.', 'This is a photo of fine_2.']

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_stage(self):
        """test stage logic"""
        with pytest.raises(Exception):
            stage = "mock_stage"
            assert Cifar100DataSet(self.path, stage=stage)
        stage = "test"
        dataset = Cifar100DataSet(self.path, stage=stage)
        assert dataset.image.shape == (1, 32, 32, 3)
        assert all(dataset.image[0][0][0] == [45, 21, 177])
        assert dataset.label == [1]
        assert dataset.label_names == ['fine_1', 'fine_2']
        assert dataset.text == ['This is a photo of fine_1.', 'This is a photo of fine_2.']

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_dataset_dir(self):
        """test dataset dir logic"""
        with pytest.raises(Exception):
            mock_dir = "not_a_dir"
            assert Cifar100DataSet(dataset_dir=mock_dir)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_fine_label(self):
        """test fine label logic"""
        fine_label = False
        dataset = Cifar100DataSet(self.path, fine_label=fine_label)
        assert dataset.label_names == ['coarse_1', 'coarse_2']
        assert dataset.text == ['This is a photo of coarse_1.', 'This is a photo of coarse_2.']

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_other(self):
        """test other logic"""
        stage = "all"
        dataset = Cifar100DataSet(self.path, stage=stage)
        assert dataset.image.shape == (2, 32, 32, 3)
        assert dataset.label == [0, 1]
        fine_label = False
        hypothesis_template = "This is a test prompt of {}."
        dataset = Cifar100DataSet(self.path, stage=stage, fine_label=fine_label,
                                  hypothesis_template=hypothesis_template)
        assert dataset.label_names == ['coarse_1', 'coarse_2']
        assert dataset.text == ['This is a test prompt of coarse_1.', 'This is a test prompt of coarse_2.']

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_getitem(self):
        """test __getitem__ logic"""
        dataset = Cifar100DataSet(self.path)
        item = dataset[0]
        assert all(item[0][0][0] == [117, 118, 196])
        assert item[1] == ['This is a photo of fine_1.', 'This is a photo of fine_2.']
        assert item[2] == 0
        assert len(item) == 3

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_len(self):
        """test __len__ logic"""
        dataset = Cifar100DataSet(self.path)
        assert len(dataset) == 1
