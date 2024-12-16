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
"""test cluener dataloader"""
import unittest
import tempfile
from mindformers.dataset.dataloader.cluener_dataloader import CLUENERDataSet, CLUENERDataLoader
import pytest

from tests.st.test_ut.test_dataset.get_test_data import get_cluener_data


class TestCluenerDataloader(unittest.TestCase):
    """ A test class for testing Cluener dataloader"""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.path = cls.temp_dir.name
        get_cluener_data(cls.path)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_default(self):
        """test default logic"""
        dataloader = CLUENERDataLoader(self.path)
        columns = dataloader.column_names
        assert set(columns) == {"text", "label_id"}
        dataloader = dataloader.batch(1)
        for item in dataloader:
            assert item[0].asnumpy()[0] == \
                   "浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，"
            assert all(item[1][0].asnumpy() == [23, 0, 0, 0, 0, 0, 0, 0, 0, 7, 17, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                0, 0, 0])
            assert item[1].shape == (1, 50)
            break

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_not_dir(self):
        """test not dir input logic"""
        mock_dir = "not_a_dir"
        with pytest.raises(ValueError):
            assert CLUENERDataLoader(mock_dir)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_column_names(self):
        """test columns names logic"""
        with pytest.raises(TypeError):
            column_names = 1
            assert CLUENERDataLoader(self.path, column_names=column_names)
        with pytest.raises(ValueError):
            column_names = [1]
            assert CLUENERDataLoader(self.path, column_names=column_names)
        with pytest.raises(ValueError):
            column_names = [1, 2]
            assert CLUENERDataLoader(self.path, column_names=column_names)


class TestCluenerDataSet(unittest.TestCase):
    """ A test class for testing Cluener dataset"""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.path = cls.temp_dir.name
        get_cluener_data(cls.path)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_default(self):
        """test default logic"""
        dataset = CLUENERDataSet(self.path)
        assert dataset.texts[0] == '浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，'
        assert dataset.label_ids[0] == [23, 0, 0, 0, 0, 0, 0, 0, 0, 7, 17, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        assert dataset.label2id == {'O': 0, 'B-address': 1, 'B-book': 2, 'B-company': 3, 'B-game': 4,
                                    'B-government': 5, 'B-movie': 6, 'B-name': 7, 'B-organization': 8,
                                    'B-position': 9, 'B-scene': 10, 'I-address': 11, 'I-book': 12, 'I-company': 13,
                                    'I-game': 14, 'I-government': 15, 'I-movie': 16, 'I-name': 17,
                                    'I-organization': 18, 'I-position': 19, 'I-scene': 20, 'S-address': 21,
                                    'S-book': 22, 'S-company': 23, 'S-game': 24, 'S-government': 25, 'S-movie': 26,
                                    'S-name': 27, 'S-organization': 28, 'S-position': 29, 'S-scene': 30}

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_stage(self):
        """test stage logic"""
        with pytest.raises(ValueError):
            stage = "mock_stage"
            assert CLUENERDataSet(self.path, stage=stage)
        stage = "test"
        dataset = CLUENERDataSet(self.path, stage=stage)
        assert dataset.texts[0] == '四川敦煌学”。近年来，丹棱县等地一些不知名的石窟迎来了海内外的游客，他们随身携带着胡文和的著作。'
        assert dataset.label_ids[0] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        stage = "dev"
        dataset = CLUENERDataSet(self.path, stage=stage)
        assert dataset.texts[0] == '彭小军认为，国内银行现在走的是台湾的发卡模式，先通过跑马圈地再在圈的地里面选择客户，'
        assert dataset.label_ids[0] == [7, 17, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 11, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_dataset_dir(self):
        """test not dir input logic"""
        with pytest.raises(Exception):
            mock_dir = "not_a_dir"
            assert CLUENERDataSet(dataset_dir=mock_dir)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_getitem(self):
        """test __getitem__ logic"""
        dataset = CLUENERDataSet(self.path)
        item = dataset[0]
        assert item[0] == '浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，'
        assert item[1] == [23, 0, 0, 0, 0, 0, 0, 0, 0, 7, 17, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_len(self):
        """test __len__ logic"""
        dataset = CLUENERDataSet(self.path)
        assert len(dataset) == 1
