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
"""test adgen dataloader"""
import os
import unittest
import tempfile
from mindformers.dataset.dataloader.adgen_dataloader import ADGenDataset, ADGenDataLoader
import pytest
from tests.st.test_ut.test_dataset.get_test_data import get_adgen_data


class TestAdgenDataloader(unittest.TestCase):
    """ A test class for testing Adgen dataloader"""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.path = cls.temp_dir.name
        cls.phase = "train"
        cls.data_path = os.path.join(cls.path, f"{cls.phase}.json")
        cls.columns = ["content", "summary"]
        get_adgen_data(cls.path)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_default(self):
        """test default logic"""
        dataloader = ADGenDataLoader(dataset_dir=self.data_path, phase=self.phase, shuffle=False,
                                     origin_columns=self.columns)
        columns = dataloader.column_names
        assert columns == ['prompt', 'answer']
        dataloader = dataloader.batch(1)
        for item in dataloader:
            assert len(item) == 2
            assert item[0].asnumpy()[0] == "类型#裤*版型#宽松*风格#性感*图案#线条*裤型#阔腿裤"
            assert item[1].asnumpy()[0] == "宽松的阔腿裤这两年真的吸粉不少，明星时尚达人的心头爱。"
            break

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_not_dir(self):
        """test not dir input logic"""
        mock_dir = os.path.join(self.path, "not_a_dir")
        with pytest.raises(ValueError):
            assert ADGenDataLoader(dataset_dir=mock_dir, phase=self.phase, shuffle=False, origin_columns=self.columns)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_origin_columns(self):
        """test origin columns logic"""
        with pytest.raises(TypeError):
            column_names = [1]
            assert ADGenDataLoader(dataset_dir=self.data_path, phase=self.phase, shuffle=False,
                                   origin_columns=column_names)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_illegal_phase(self):
        """test illegal phase logic"""
        phase = "mock_phase"
        with pytest.raises(ValueError):
            assert ADGenDataLoader(dataset_dir=self.data_path, phase=phase, shuffle=False,
                                   origin_columns=self.columns)


class TestAdgenDataSet(unittest.TestCase):
    """ A test class for testing Adgen dataset"""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.path = cls.temp_dir.name

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_default(self):
        """test default logic"""
        phase = "train"
        data_path = os.path.join(self.path, f"{phase}.json")
        columns = ["content", "summary"]
        get_adgen_data(self.path)
        dataset = ADGenDataset(dataset_dir=data_path, origin_columns=columns)
        res = dataset[0]
        assert len(res) == 2
        assert res[0] == "类型#裤*版型#宽松*风格#性感*图案#线条*裤型#阔腿裤"
        assert res[1] == "宽松的阔腿裤这两年真的吸粉不少，明星时尚达人的心头爱。"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_other(self):
        """test other logic"""
        get_adgen_data(self.path, is_json_error=True)
        phase = "train"
        data_path = os.path.join(self.path, f"json_error_{phase}.json")
        columns = ["content", "summary"]
        with pytest.raises(ValueError):
            mock_dir = "not_a_dir"
            assert ADGenDataset(mock_dir, columns)

        with pytest.raises(ValueError):
            assert ADGenDataset(data_path, columns)
