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
"""test flickr8k dataloader"""
import os
import unittest
import tempfile
from mindformers.dataset.dataloader.flickr8k_dataloader import Flickr8kDataLoader, Flickr8kDataSet
import pytest
from tests.st.test_ut.test_dataset.get_test_data import get_flickr8k_data


class TestFlickr8kDataloader(unittest.TestCase):
    """ A test class for testing Flickr8k dataloader"""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.path = cls.temp_dir.name
        cls.phase = "train"
        cls.columns = ["content", "summary"]
        get_flickr8k_data(cls.path)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_default(self):
        """test default logic"""
        dataloader = Flickr8kDataLoader(dataset_dir=self.path)
        columns = dataloader.column_names
        assert columns == ["image", "text"]
        dataloader = dataloader.batch(1)
        for item in dataloader:
            assert len(item) == 2
            assert item[0].shape == (1, 224, 224, 3)
            assert all(item[1].asnumpy()[0] == [
                'A child in a pink dress is climbing up a set of stairs in an entry way .',
                'A girl going into a wooden building .',
                'A little girl climbing into a wooden playhouse .']
                       )
            break

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_not_dir(self):
        """test not dir input logic"""
        mock_dir = "not_a_dir"
        with pytest.raises(ValueError):
            assert Flickr8kDataLoader(dataset_dir=mock_dir)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_column_names(self):
        """test column names logic"""
        with pytest.raises(TypeError):
            column_names = 1
            assert Flickr8kDataLoader(self.path, column_names=column_names)
        with pytest.raises(ValueError):
            column_names = [1]
            assert Flickr8kDataLoader(self.path, column_names=column_names)
        with pytest.raises(ValueError):
            column_names = [1, 2]
            assert Flickr8kDataLoader(self.path, column_names=column_names)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_dataset_dir(self):
        """test dataset dir logic"""
        tmp_path = tempfile.TemporaryDirectory()
        with pytest.raises(ValueError):
            assert Flickr8kDataLoader(tmp_path.name)
        with pytest.raises(ValueError):
            os.makedirs(os.path.join(tmp_path.name, "Flickr8k_Dataset", "Flickr8k_Dataset"), exist_ok=True)
            assert Flickr8kDataLoader(tmp_path.name)


class TestFlickr8kDataSet(unittest.TestCase):
    """ A test class for testing Flickr8k dataset"""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.path = cls.temp_dir.name
        cls.dataset_dir = os.path.join(cls.path, "Flickr8k_Dataset", "Flickr8k_Dataset")
        cls.annotation_dir = os.path.join(cls.path, "Flickr8k_text")
        get_flickr8k_data(cls.path)
        cls.res = [
            'A child in a pink dress is climbing up a set of stairs in an entry way .',
            'A girl going into a wooden building .', 'A little girl climbing into a wooden playhouse .']

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_default(self):
        """test default logic"""
        dataset = Flickr8kDataSet(self.dataset_dir, self.annotation_dir)
        assert hasattr(dataset, "dataset_dict")
        assert dataset.dataset_dict["mock.jpg"] == self.res

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_stage(self):
        """test stage logic"""
        with pytest.raises(ValueError):
            stage = "mock_stage"
            assert Flickr8kDataSet(self.dataset_dir, self.annotation_dir, stage=stage)
        stage = "test"
        dataset = Flickr8kDataSet(self.dataset_dir, self.annotation_dir, stage=stage)
        assert dataset.dataset_dict["mock.jpg"] == self.res
        stage = "dev"
        dataset = Flickr8kDataSet(self.dataset_dir, self.annotation_dir, stage=stage)
        assert dataset.dataset_dict["mock.jpg"] == self.res
        stage = "all"
        dataset = Flickr8kDataSet(self.dataset_dir, self.annotation_dir, stage=stage)
        assert dataset.dataset_dict["mock.jpg"] == self.res

    def test_dataset_dir(self):
        with pytest.raises(ValueError):
            assert Flickr8kDataSet(os.path.join(self.dataset_dir, "mock_dir"), self.annotation_dir)
        with pytest.raises(ValueError):
            assert Flickr8kDataSet(self.dataset_dir, os.path.join(self.annotation_dir, "mock_dir"))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_getitem(self):
        """test __getitem__ logic"""
        dataset = Flickr8kDataSet(self.dataset_dir, self.annotation_dir)
        item = dataset[0]
        assert hasattr(item[0], "height") and item[0].height == 224
        assert item[1] == self.res

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_len(self):
        """test __len__ logic"""
        dataset = Flickr8kDataSet(self.dataset_dir, self.annotation_dir)
        assert len(dataset) == 1
